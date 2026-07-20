import {
  resolveOnnxModelProfile,
  type HardwareTopology,
  type ModelProfile,
  type QuantType,
  type SimulationScenario,
} from "@inference-sim/core";
import type {
  ImportedModelPackage,
  ImportedOnnxModel,
} from "./model-package-import.js";

export interface ModelComponentMetrics {
  readonly fileName: string;
  readonly modelName: string;
  readonly modelType?: string;
  readonly parameterCount: number;
  readonly weightBytes: number;
  readonly graphNodes: number;
  readonly operatorKinds: number;
  readonly topOperators: readonly string[];
  readonly weightDtypes: readonly string[];
  readonly forwardFlopsPerToken?: number;
  readonly activeWeightBytesPerToken?: number;
  readonly weightQuantization?: QuantType;
  readonly architecture?: {
    readonly layers: number;
    readonly hiddenSize: number;
    readonly attentionHeads: number;
    readonly kvHeads: number;
    readonly experts?: number;
    readonly activeExperts?: number;
  };
}

export interface ModelPackageMetrics {
  readonly packageBytes: number;
  readonly modelFileBytes: number;
  readonly weightBytes: number;
  readonly parameterCount: number;
  readonly graphNodes: number;
  readonly components: readonly ModelComponentMetrics[];
  readonly forwardFlopsPerToken?: number;
  readonly activeWeightBytesPerToken?: number;
  readonly hotMemoryBandwidthBytesPerSec: number;
  readonly bandwidthCeilingTokensPerSec?: number;
  readonly completeComputeProfiles: number;
}

export interface IdealRooflineMetrics {
  readonly activeWeightBytesPerToken: number;
  readonly forwardFlopsPerToken: number;
  readonly aggregateMemoryBandwidthBytesPerSec: number;
  readonly aggregatePeakComputeFlops: number;
  readonly bandwidthCeilingTokensPerSec: number;
  readonly computeCeilingTokensPerSec?: number;
  readonly rooflineCeilingTokensPerSec: number;
  readonly limitingResource: "compute" | "memory_bandwidth";
}

export function summarizeModelPackage(
  modelPackage: ImportedModelPackage,
  scenario?: SimulationScenario,
): ModelPackageMetrics {
  const components = modelPackage.models.map(summarizeComponent);
  const complete = components.filter((component) => (
    component.forwardFlopsPerToken !== undefined
    && component.activeWeightBytesPerToken !== undefined
  ));
  const hotMemoryBandwidthBytesPerSec = scenario === undefined
    ? 0
    : aggregateHotMemoryBandwidth(scenario);
  const hasSingleCompleteModel =
    components.length === 1 && complete.length === 1;
  const activeWeightBytesPerToken = hasSingleCompleteModel
    ? sum(complete.map((component) => component.activeWeightBytesPerToken!))
    : undefined;
  const forwardFlopsPerToken = hasSingleCompleteModel
    ? sum(complete.map((component) => component.forwardFlopsPerToken!))
    : undefined;

  return {
    packageBytes: modelPackage.packageByteLength,
    modelFileBytes: sum(modelPackage.models.map(
      (model) => model.manifest.source.modelByteLength,
    )),
    weightBytes: sum(components.map((component) => component.weightBytes)),
    parameterCount: sum(components.map(
      (component) => component.parameterCount,
    )),
    graphNodes: sum(components.map((component) => component.graphNodes)),
    components,
    ...(forwardFlopsPerToken === undefined ? {} : { forwardFlopsPerToken }),
    ...(activeWeightBytesPerToken === undefined
      ? {}
      : { activeWeightBytesPerToken }),
    hotMemoryBandwidthBytesPerSec,
    ...(activeWeightBytesPerToken === undefined
      || activeWeightBytesPerToken === 0
      || hotMemoryBandwidthBytesPerSec === 0
      ? {}
      : {
          bandwidthCeilingTokensPerSec:
            hotMemoryBandwidthBytesPerSec / activeWeightBytesPerToken,
        }),
    completeComputeProfiles: complete.length,
  };
}

export function calculateIdealRoofline(
  model: ModelProfile,
  topology: HardwareTopology,
  maximumDevices?: number,
): IdealRooflineMetrics {
  const modelWork = summarizeProfileWork(model);
  const devices = topology.nodes
    .flatMap((node) => node.devices)
    .slice(0, maximumDevices);
  const aggregateMemoryBandwidthBytesPerSec = sum(devices.map(
    (device) => device.memory.bandwidthBytesPerSec,
  ));
  const aggregatePeakComputeFlops = sum(devices.map((device) => (
    peakComputeForQuantization(device.compute, model.quantization.weights) ?? 0
  )));
  const bandwidthCeilingTokensPerSec =
    aggregateMemoryBandwidthBytesPerSec / modelWork.activeWeightBytesPerToken;
  const computeCeilingTokensPerSec = aggregatePeakComputeFlops > 0
    ? aggregatePeakComputeFlops / modelWork.forwardFlopsPerToken
    : undefined;
  const rooflineCeilingTokensPerSec = computeCeilingTokensPerSec === undefined
    ? bandwidthCeilingTokensPerSec
    : Math.min(computeCeilingTokensPerSec, bandwidthCeilingTokensPerSec);

  return {
    ...modelWork,
    aggregateMemoryBandwidthBytesPerSec,
    aggregatePeakComputeFlops,
    bandwidthCeilingTokensPerSec,
    ...(computeCeilingTokensPerSec === undefined
      ? {}
      : { computeCeilingTokensPerSec }),
    rooflineCeilingTokensPerSec,
    limitingResource: computeCeilingTokensPerSec !== undefined
      && computeCeilingTokensPerSec < bandwidthCeilingTokensPerSec
      ? "compute"
      : "memory_bandwidth",
  };
}

function summarizeComponent(
  model: ImportedOnnxModel,
): ModelComponentMetrics {
  const architecture = model.manifest.architecture;
  const weightDtypes = [...new Set(model.manifest.initializers
    .filter((initializer) => initializer.dimensions.length >= 2)
    .map((initializer) => initializer.dataType))].sort();
  const base = {
    fileName: model.fileName,
    modelName: architecture.modelType
      ?? model.fileName.replace(/\.onnx$/i, ""),
    ...(architecture.modelType === undefined
      ? {}
      : { modelType: architecture.modelType }),
    parameterCount: model.manifest.totals.initializerElements,
    weightBytes: model.manifest.totals.initializerLogicalBytes,
    graphNodes: model.manifest.graph.nodeCount,
    operatorKinds: model.manifest.graph.operators.length,
    topOperators: [...model.manifest.graph.operators]
      .sort((left, right) => (
        right.count - left.count
        || left.opType.localeCompare(right.opType)
      ))
      .slice(0, 3)
      .map((operator) => `${operator.opType} ${operator.count}`),
    weightDtypes: weightDtypes.length === 0 ? ["unknown"] : weightDtypes,
  };
  if (!model.manifest.profileReadiness.ready) {
    return base;
  }
  try {
    const profile = resolveOnnxModelProfile(model.manifest);
    return {
      ...base,
      ...summarizeProfileWork(profile),
      weightQuantization: profile.quantization.weights,
      architecture: {
        layers: profile.architecture.numLayers,
        hiddenSize: profile.architecture.hiddenDim,
        attentionHeads: profile.architecture.numHeads,
        kvHeads: profile.architecture.numKVHeads,
        ...(profile.moe === undefined
          ? {}
          : {
              experts: profile.moe.numExperts,
              activeExperts: profile.moe.activeExpertsPerToken,
            }),
      },
    };
  } catch {
    return base;
  }
}

function summarizeProfileWork(model: ModelProfile): {
  readonly activeWeightBytesPerToken: number;
  readonly forwardFlopsPerToken: number;
} {
  const denseWeightBytes = sum(model.layers.map(
    (layer) => layer.attentionBytes + layer.ffnBytes,
  ));
  const activeExpertBytes = model.moe === undefined
    ? 0
    : model.architecture.numLayers * (
      model.moe.activeExpertsPerToken * model.moe.expertBytesPerLayer
      + model.moe.sharedExpertBytesPerLayer
    );
  const activeWeightBytesPerToken = denseWeightBytes + activeExpertBytes;
  const activeParameters = model.moe === undefined
    ? model.totalParams
    : activeWeightBytesPerToken / bytesPerElement(model.quantization.weights);
  return {
    activeWeightBytesPerToken,
    forwardFlopsPerToken: 2 * activeParameters,
  };
}

function aggregateHotMemoryBandwidth(
  scenario: SimulationScenario,
): number {
  const domainById = new Map(scenario.memoryDomains.map(
    (domain) => [domain.id, domain] as const,
  ));
  const selected = new Set<string>();
  for (const device of scenario.devices) {
    if (!device.capabilities.some((capability) => (
      capability === "attention"
      || capability === "ffn"
      || capability === "draft"
    ))) {
      continue;
    }
    const accessible = device.memoryDomainIds
      .map((id) => domainById.get(id))
      .filter((domain) => domain !== undefined);
    const preferredKinds = device.kind === "cpu"
      ? new Set(["host", "unified"])
      : new Set(["device", "unified"]);
    const preferred = accessible.filter((domain) => (
      preferredKinds.has(domain.kind)
    ));
    for (const domain of preferred.length > 0
      ? preferred
      : accessible.filter((domain) => domain.kind !== "storage")) {
      selected.add(domain.id);
    }
  }
  return sum([...selected].map(
    (id) => domainById.get(id)?.bandwidthBytesPerSec ?? 0,
  ));
}

function peakComputeForQuantization(
  compute: HardwareTopology["nodes"][number]["devices"][number]["compute"],
  quantization: QuantType,
): number | undefined {
  switch (quantization) {
    case "fp16":
      return compute.fp16Flops;
    case "fp8":
      return compute.fp8Flops;
    case "int8":
      return compute.int8Flops;
    case "fp32":
    case "bf16":
    case "int4":
    case "nf4":
      return undefined;
  }
}

function bytesPerElement(quantization: QuantType): number {
  switch (quantization) {
    case "fp32": return 4;
    case "fp16":
    case "bf16": return 2;
    case "fp8":
    case "int8": return 1;
    case "int4":
    case "nf4": return 0.5;
  }
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
