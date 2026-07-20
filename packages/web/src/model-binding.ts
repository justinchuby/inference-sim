import {
  buildModelProfile,
  resolveOnnxModelProfile,
  type ModelProfile,
  type QuantType,
  type SpeculativeProposerFamily,
  type TopologyPipelinePhase,
  type TopologyPipelineWork,
} from "@inference-sim/core";
import type { ImportedModelPackage } from "./model-package-import.js";
import type {
  DashboardModelBinding,
  DashboardModelExecutionProfile,
  DashboardModelFormat,
} from "./types.js";

export const DASHBOARD_MODEL_PRESETS = [
  "llama-3-8b",
  "llama-3-70b",
  "mixtral-8x22b",
  "deepseek-v2",
  "qwen-3-235b",
] as const;

export type DashboardModelPreset = typeof DASHBOARD_MODEL_PRESETS[number];

export function modelSupportsSpeculativeFamily(
  binding: DashboardModelBinding,
  family: SpeculativeProposerFamily,
): boolean {
  return binding.speculativeFamilies.includes(family)
    || (binding.source === "builtin_model" && family === "prompt_lookup");
}

export function createBuiltinModelBinding(
  preset: DashboardModelPreset,
  weightDtype: QuantType = "fp16",
  kvCacheDtype: QuantType = "fp16",
): DashboardModelBinding {
  const model = buildModelProfile(preset, weightDtype, kvCacheDtype);
  const fingerprint = `builtin:${preset}:${weightDtype}:${kvCacheDtype}`;
  const moeLimitations = model.moe === undefined
    ? []
    : ["model_moe_routing_not_bound_to_expert_workload"];
  return {
    source: "builtin_model",
    displayName: model.name,
    modelFingerprints: [fingerprint],
    targetModelFingerprint: fingerprint,
    componentCount: 1,
    totalParameters: model.totalParams,
    weightBytes: totalModelWeightBytes(model),
    modelFormat: {
      weightDtypes: [weightDtype],
      weightQuantization: quantizationKind(weightDtype),
      kvCacheDtype: model.quantization.kvCache,
      activationDtype: model.quantization.activations,
      evidence: "preset_declared",
      runtimeDtypesDefaulted: false,
    },
    executionProfile: executionProfile(model, preset),
    executionCoverage: {
      fidelity: moeLimitations.length === 0 ? "complete" : "partial",
      scope: "full_model",
      modeledComponentIds: [preset],
      unmodeledComponentIds: [],
      limitations: moeLimitations,
    },
    speculativeFamilies: ["prompt_lookup"],
  };
}

export function createImportedModelBinding(
  modelPackage: ImportedModelPackage,
): DashboardModelBinding {
  const target = selectTargetModel(modelPackage);
  const model = target.manifest.profileReadiness.ready
    ? resolveOnnxModelProfile(target.manifest)
    : undefined;
  const fingerprint = target.manifest.manifestFingerprint;
  const pipelineExecution = buildPipelineExecution(modelPackage, target);
  const assessedCoverage = assessImportedModelExecutionCoverage(
    modelPackage,
    pipelineExecution,
  );
  const executionCoverage = model?.moe === undefined
    ? assessedCoverage
    : addCoverageLimitation(
        assessedCoverage,
        "model_moe_routing_not_bound_to_expert_workload",
      );
  const modelFormat = importedModelFormat(modelPackage, model);
  return {
    source: "local_model_package",
    displayName: modelPackage.metadata.components.length > 1
      ? modelPackage.metadata.components.map((component) => component.id)
        .join(" -> ")
      : model?.name
        ?? target.manifest.architecture.modelType
        ?? target.fileName.replace(/\.onnx$/i, ""),
    modelFingerprints: modelPackage.models
      .map((candidate) => candidate.manifest.manifestFingerprint)
      .sort(),
    targetModelFingerprint: fingerprint,
    componentCount: modelPackage.metadata.components.length
      || modelPackage.models.length,
    totalParameters: modelPackage.models.reduce(
      (sum, candidate) => sum + candidate.manifest.totals.initializerElements,
      0,
    ),
    weightBytes: modelPackage.models.reduce(
      (sum, candidate) => (
        sum + candidate.manifest.totals.initializerLogicalBytes
      ),
      0,
    ),
    modelFormat,
    executionProfile: model === undefined
      ? genericExecutionProfile(target)
      : executionProfile(model, fingerprint),
    ...(pipelineExecution === undefined ? {} : { pipelineExecution }),
    executionCoverage,
    ...(modelPackage.metadata.pipelineStrategy === undefined
      ? {}
      : { pipelineStrategy: modelPackage.metadata.pipelineStrategy }),
    speculativeFamilies:
      modelPackage.metadata.speculative.availableFamilies,
  };
}

function importedModelFormat(
  modelPackage: ImportedModelPackage,
  targetProfile: ModelProfile | undefined,
): DashboardModelFormat {
  const bytesByDtype = new Map<string, number>();
  const quantizations = new Set<DashboardModelFormat["weightQuantization"]>();
  for (const importedModel of modelPackage.models) {
    for (const initializer of importedModel.manifest.initializers) {
      if (initializer.dimensions.length < 2) {
        continue;
      }
      bytesByDtype.set(
        initializer.dataType,
        (bytesByDtype.get(initializer.dataType) ?? 0)
          + initializer.logicalByteLength,
      );
      quantizations.add(quantizationKindForOnnxDtype(initializer.dataType));
    }
  }
  const weightDtypes = [...bytesByDtype.entries()]
    .sort((left, right) => (
      right[1] - left[1] || left[0].localeCompare(right[0])
    ))
    .map(([dtype]) => dtype);
  const knownQuantizations = [...quantizations].filter(
    (quantization) => quantization !== "unknown",
  );
  const weightQuantization = knownQuantizations.length === 0
    ? "unknown"
    : new Set(knownQuantizations).size === 1
        && !quantizations.has("unknown")
      ? knownQuantizations[0]!
      : "mixed";
  return {
    weightDtypes: weightDtypes.length === 0 ? ["unknown"] : weightDtypes,
    weightQuantization,
    kvCacheDtype: targetProfile?.quantization.kvCache ?? "unknown",
    activationDtype: targetProfile?.quantization.activations ?? "unknown",
    evidence: "onnx_inferred",
    runtimeDtypesDefaulted: targetProfile !== undefined,
  };
}

function quantizationKind(
  dtype: QuantType,
): DashboardModelFormat["weightQuantization"] {
  switch (dtype) {
    case "fp32":
    case "fp16":
    case "bf16":
      return "none";
    case "fp8":
    case "int8":
    case "int4":
    case "nf4":
      return dtype;
  }
}

function quantizationKindForOnnxDtype(
  dtype: string,
): DashboardModelFormat["weightQuantization"] {
  switch (dtype) {
    case "float":
    case "float16":
    case "bfloat16":
      return "none";
    case "float8e4m3fn":
    case "float8e4m3fnuz":
    case "float8e5m2":
    case "float8e5m2fnuz":
    case "float8e8m0":
      return "fp8";
    case "int8":
    case "uint8":
      return "int8";
    case "int4":
    case "uint4":
      return "int4";
    default:
      return "unknown";
  }
}

function addCoverageLimitation(
  coverage: DashboardModelBinding["executionCoverage"],
  limitation: string,
): DashboardModelBinding["executionCoverage"] {
  return coverage.limitations.includes(limitation)
    ? coverage
    : {
        ...coverage,
        fidelity: "partial",
        limitations: [...coverage.limitations, limitation],
      };
}

export function assessImportedModelExecutionCoverage(
  modelPackage: ImportedModelPackage,
  pipelineExecution = buildPipelineExecution(
    modelPackage,
    selectTargetModel(modelPackage),
  ),
): DashboardModelBinding["executionCoverage"] {
  const allComponentIds = modelPackage.metadata.components.length > 0
    ? modelPackage.metadata.components.map((component) => component.id)
    : modelPackage.models.flatMap((model) => (
        model.componentIds.length > 0 ? model.componentIds : [model.fileName]
      ));
  const modeledComponentIds = pipelineExecution?.components.map(
    (component) => component.id,
  ) ?? allComponentIds;
  const modeled = new Set(modeledComponentIds);
  const unmodeledComponentIds = allComponentIds
    .filter((id) => !modeled.has(id))
    .sort();
  const limitations: string[] = [];
  if (
    pipelineExecution !== undefined
    && ![
      "autoregressive",
      "single_pass",
      "composite",
      "iterative",
    ].includes(pipelineExecution.strategyKind)
  ) {
    limitations.push("pipeline_strategy_not_executable");
  }
  if (pipelineExecution?.components.some(
    (component) => component.phase === "on_demand",
  )) {
    limitations.push("on_demand_components_require_application_invocation");
  }
  if (pipelineExecution?.components.some(
    (component) => component.strategyKind === "iterative",
  )) {
    limitations.push("iterative_scheduler_and_cfg_cost_not_modeled");
  }
  if (pipelineExecution?.components.some(
    (component) => component.strategyKind === "nested_autoregressive",
  )) {
    limitations.push("nested_autoregressive_inner_loop_not_modeled");
  }
  if (modelPackage.metadata.vision !== undefined) {
    limitations.push("vision_request_tile_expansion_not_modeled");
  }
  if (
    modelPackage.metadata.components.some(
      (component) => component.type.toLowerCase() === "draft",
    )
  ) {
    limitations.push("draft_model_profile_not_bound_to_proposer_cost");
  }
  const hardware = modelPackage.metadata.hardware;
  if (
    hardware.minimumMemoryGiB !== undefined
    || hardware.minimumTensorParallelDegree !== undefined
    || hardware.supportsTensorParallel !== undefined
    || hardware.requiredDtypes.length > 0
  ) {
    limitations.push("metadata_hardware_requirements_not_enforced");
  }
  return {
    fidelity: limitations.length === 0 ? "complete" : "partial",
    scope: unmodeledComponentIds.length === 0 ? "full_model" : "target_component_only",
    modeledComponentIds: [...new Set(modeledComponentIds)].sort(),
    unmodeledComponentIds,
    limitations,
  };
}

function selectTargetModel(modelPackage: ImportedModelPackage) {
  if (modelPackage.models.length === 1) {
    return modelPackage.models[0]!;
  }
  const componentType = new Map(modelPackage.metadata.components.map(
    (component) => [component.id, component.type.toLowerCase()] as const,
  ));
  const leafStages = modelPackage.metadata.stages.filter(
    (stage) => stage.componentIds.length > 0,
  );
  const preferredComponentId =
    leafStages.find((stage) => stage.kind === "autoregressive")
      ?.bindings.decoder
    ?? leafStages.find((stage) => stage.kind === "nested_autoregressive")
      ?.bindings.outer
    ?? leafStages.find((stage) => stage.kind === "iterative")
      ?.bindings.denoiser
    ?? leafStages.at(-1)?.bindings.model;
  const scored = modelPackage.models.map((model) => {
    const types = model.componentIds.map((id) => componentType.get(id) ?? "");
    const stageScore = model.componentIds.includes(preferredComponentId ?? "")
      ? 100
      : 0;
    const roleScore = types.includes("decoder")
      ? 4
      : types.includes("target")
        ? 3
        : types.includes("model")
          ? 2
          : types.some((type) => type !== "draft" && type !== "encoder")
            ? 1
            : 0;
    return { model, score: stageScore * 10 + roleScore };
  }).sort((left, right) => right.score - left.score);
  if (
    scored[0] === undefined
    || scored[0].score === scored[1]?.score
  ) {
    throw new Error(
      "multi-model package must identify exactly one executable primary component",
    );
  }
  return scored[0].model;
}

function buildPipelineExecution(
  modelPackage: ImportedModelPackage,
  primaryModel: ImportedModelPackage["models"][number],
): TopologyPipelineWork | undefined {
  const metadata = modelPackage.metadata;
  if (metadata.components.length === 0) {
    return undefined;
  }
  const modelByComponent = new Map(
    modelPackage.models.flatMap((model) => (
      model.componentIds.map((componentId) => [componentId, model] as const)
    )),
  );
  const stageByComponent = new Map<string, typeof metadata.stages[number]>();
  const orderByComponent = new Map<string, number>();
  for (const [index, stage] of metadata.stages.entries()) {
    for (const componentId of stage.componentIds) {
      stageByComponent.set(componentId, stage);
      orderByComponent.set(componentId, index);
    }
  }
  const primaryIds = new Set(primaryModel.componentIds);
  const rerunEveryStepIds = new Set(metadata.edges.filter((edge) => (
    primaryIds.has(edge.toComponent)
    && edge.toPort.toLowerCase().endsWith("inputs_embeds")
  )).map((edge) => edge.fromComponent));
  const strategyKind = metadata.pipelineStrategy ?? "single_pass";
  const replacesTarget = !metadata.stages.some(
    (stage) => stage.kind === "autoregressive"
      || stage.kind === "nested_autoregressive",
  );
  const components = topologicalComponents(metadata.components, metadata.edges)
    .map((component) => {
      const model = modelByComponent.get(component.id);
      if (model === undefined) {
        throw new Error(
          `pipeline component ${component.id} is not bound to an ONNX manifest`,
        );
      }
      const stage = stageByComponent.get(component.id);
      const phase = normalizePhase(
        component.runOn
          ?? stage?.runOn
          ?? (component.type.toLowerCase() === "draft"
            ? "on_demand"
            : undefined)
          ?? (stage?.kind === "iterative"
              || stage?.kind === "nested_autoregressive"
            ? "every_step"
            : undefined)
          ?? (primaryIds.has(component.id) && !replacesTarget
            ? "every_step"
            : "prompt_only"),
      );
      const invocationMultiplier = stage?.kind === "iterative"
        ? Math.max(1, (stage.numSteps ?? 1) - (stage.startStep ?? 0))
        : stage?.kind === "nested_autoregressive"
            && stage.bindings.inner === component.id
          ? stage.numCodeGroups ?? 1
        : 1;
      return {
        id: component.id,
        role: component.type,
        phase,
        strategyKind: stage?.kind ?? strategyKind,
        invocationMultiplier,
        weightBytes: model.manifest.totals.initializerLogicalBytes,
        ...(component.devicePreference === undefined
          ? {}
          : { devicePreference: component.devicePreference.toLowerCase() }),
        isPrimary: primaryIds.has(component.id),
        order: orderByComponent.get(component.id)
          ?? metadata.stages.length + metadata.components.findIndex(
            (candidate) => candidate.id === component.id,
          ),
        ...(rerunEveryStepIds.has(component.id)
          ? { rerunEveryStep: true }
          : {}),
      };
    }).sort((left, right) => left.order - right.order);
  return {
    strategyKind,
    replacesTarget,
    components,
    edges: metadata.edges.map((edge) => ({
      fromComponent: edge.fromComponent,
      toComponent: edge.toComponent,
      ...(edge.deviceTransfer === undefined
        ? {}
        : { deviceTransfer: edge.deviceTransfer }),
    })),
  };
}

function topologicalComponents<T extends { readonly id: string }>(
  components: readonly T[],
  edges: ImportedModelPackage["metadata"]["edges"],
): T[] {
  const remaining = new Map(components.map((component) => (
    [component.id, component] as const
  )));
  const ordered: T[] = [];
  while (remaining.size > 0) {
    const ready = [...remaining.values()].find((component) => (
      edges.every((edge) => (
        edge.toComponent !== component.id
        || edge.fromComponent === component.id
        || !remaining.has(edge.fromComponent)
      ))
    ));
    if (ready === undefined) {
      throw new Error("pipeline dataflow contains a non-iterative cycle");
    }
    ordered.push(ready);
    remaining.delete(ready.id);
  }
  return ordered;
}

function normalizePhase(value: string): TopologyPipelinePhase {
  if (value === "always") {
    return "every_step";
  }
  if (
    value !== "prompt_only"
    && value !== "every_step"
    && value !== "final_only"
    && value !== "on_demand"
  ) {
    throw new Error(`unsupported pipeline phase ${value}`);
  }
  return value;
}

function genericExecutionProfile(
  model: ImportedModelPackage["models"][number],
): DashboardModelExecutionProfile {
  const bytes = Math.max(1, model.manifest.totals.initializerLogicalBytes);
  const parameters = Math.max(1, model.manifest.totals.initializerElements);
  return {
    modelId: model.manifest.manifestFingerprint,
    modelName: model.manifest.architecture.modelType
      ?? model.fileName.replace(/\.onnx$/i, ""),
    attentionWeightBytesPerToken: Math.max(1, Math.floor(bytes / 4)),
    ffnWeightBytesPerToken: Math.max(1, bytes - Math.floor(bytes / 4)),
    forwardFlopsPerToken: 2 * parameters,
  };
}

function executionProfile(
  model: ModelProfile,
  modelId: string,
): DashboardModelExecutionProfile {
  const attentionWeightBytesPerToken = model.layers.reduce(
    (sum, layer) => sum + layer.attentionBytes,
    0,
  );
  const denseFfnBytes = model.layers.reduce(
    (sum, layer) => sum + layer.ffnBytes,
    0,
  );
  const activeExpertBytes = model.moe === undefined
    ? 0
    : model.architecture.numLayers * (
      model.moe.activeExpertsPerToken * model.moe.expertBytesPerLayer
      + model.moe.sharedExpertBytesPerLayer
    );
  const ffnWeightBytesPerToken = denseFfnBytes + activeExpertBytes;
  return {
    modelId,
    modelName: model.name,
    attentionWeightBytesPerToken,
    ffnWeightBytesPerToken,
    forwardFlopsPerToken: 2 * (
      model.moe === undefined
        ? model.totalParams
        : (attentionWeightBytesPerToken + ffnWeightBytesPerToken)
          / bytesPerElement(model.quantization.weights)
    ),
  };
}

function totalModelWeightBytes(model: ModelProfile): number {
  const dense = model.layers.reduce(
    (sum, layer) => sum + layer.attentionBytes + layer.ffnBytes,
    0,
  );
  const experts = model.moe === undefined
    ? 0
    : model.architecture.numLayers * (
      model.moe.numExperts * model.moe.expertBytesPerLayer
      + model.moe.sharedExpertBytesPerLayer
    );
  return dense + experts;
}

function bytesPerElement(
  quantization: ModelProfile["quantization"]["weights"],
): number {
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
