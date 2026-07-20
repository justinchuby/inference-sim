import {
  buildModelProfile,
  resolveOnnxModelProfile,
  type ModelProfile,
} from "@inference-sim/core";
import type { ImportedModelPackage } from "./model-package-import.js";
import type {
  DashboardModelBinding,
  DashboardModelExecutionProfile,
} from "./types.js";

export const DASHBOARD_MODEL_PRESETS = [
  "llama-3-8b",
  "llama-3-70b",
  "mixtral-8x22b",
  "deepseek-v2",
  "qwen-3-235b",
] as const;

export type DashboardModelPreset = typeof DASHBOARD_MODEL_PRESETS[number];

export function createBuiltinModelBinding(
  preset: DashboardModelPreset,
): DashboardModelBinding {
  const model = buildModelProfile(preset);
  const moeLimitations = model.moe === undefined
    ? []
    : ["model_moe_routing_not_bound_to_expert_workload"];
  return {
    source: "builtin_model",
    displayName: model.name,
    modelFingerprints: [`builtin:${preset}`],
    targetModelFingerprint: `builtin:${preset}`,
    componentCount: 1,
    totalParameters: model.totalParams,
    weightBytes: totalModelWeightBytes(model),
    executionProfile: executionProfile(model, `builtin:${preset}`),
    executionCoverage: {
      fidelity: moeLimitations.length === 0 ? "complete" : "partial",
      scope: "full_model",
      modeledComponentIds: [preset],
      unmodeledComponentIds: [],
      limitations: moeLimitations,
    },
    speculativeFamilies: [],
  };
}

export function createImportedModelBinding(
  modelPackage: ImportedModelPackage,
): DashboardModelBinding {
  const target = selectTargetModel(modelPackage);
  const model = resolveOnnxModelProfile(target.manifest);
  const fingerprint = target.manifest.manifestFingerprint;
  const assessedCoverage = assessImportedModelExecutionCoverage(
    modelPackage,
    target.componentIds,
  );
  const executionCoverage = model.moe === undefined
    ? assessedCoverage
    : addCoverageLimitation(
        assessedCoverage,
        "model_moe_routing_not_bound_to_expert_workload",
      );
  return {
    source: "local_model_package",
    displayName: model.name,
    modelFingerprints: modelPackage.models
      .map((candidate) => candidate.manifest.manifestFingerprint)
      .sort(),
    targetModelFingerprint: fingerprint,
    componentCount: modelPackage.metadata.components.length
      || modelPackage.models.length,
    totalParameters: model.totalParams,
    weightBytes: target.manifest.totals.initializerLogicalBytes,
    executionProfile: executionProfile(model, fingerprint),
    executionCoverage,
    ...(modelPackage.metadata.pipelineStrategy === undefined
      ? {}
      : { pipelineStrategy: modelPackage.metadata.pipelineStrategy }),
    speculativeFamilies:
      modelPackage.metadata.speculative.availableFamilies,
  };
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
  targetComponentIds?: readonly string[],
): DashboardModelBinding["executionCoverage"] {
  const allComponentIds = modelPackage.metadata.components.length > 0
    ? modelPackage.metadata.components.map((component) => component.id)
    : modelPackage.models.flatMap((model) => (
        model.componentIds.length > 0 ? model.componentIds : [model.fileName]
      ));
  const componentType = new Map(modelPackage.metadata.components.map(
    (component) => [component.id, component.type.toLowerCase()] as const,
  ));
  const selectedIds = targetComponentIds ?? selectTargetModel(
    modelPackage,
  ).componentIds;
  const modeledComponentIds = selectedIds.filter((id) => {
    const type = componentType.get(id);
    return type === undefined
      || type === "decoder"
      || type === "target"
      || type === "model";
  });
  if (modeledComponentIds.length === 0 && modelPackage.models.length === 1) {
    modeledComponentIds.push(
      selectedIds[0] ?? modelPackage.models[0]!.fileName,
    );
  }
  const modeled = new Set(modeledComponentIds);
  const unmodeledComponentIds = allComponentIds
    .filter((id) => !modeled.has(id))
    .sort();
  const limitations: string[] = [];
  if (unmodeledComponentIds.length > 0) {
    limitations.push("non_target_components_not_scheduled");
    limitations.push("non_target_weights_not_capacity_checked");
  }
  if (modelPackage.metadata.edges.length > 0) {
    limitations.push("pipeline_dataflow_transfers_not_scheduled");
  }
  if (modelPackage.metadata.components.some(
    (component) => component.devicePreference !== undefined,
  )) {
    limitations.push("component_device_preferences_not_enforced");
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
    scope: unmodeledComponentIds.length === 0
      ? "full_model"
      : "target_component_only",
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
  const scored = modelPackage.models.map((model) => {
    const types = model.componentIds.map((id) => componentType.get(id) ?? "");
    const score = types.includes("decoder")
      ? 4
      : types.includes("target")
        ? 3
        : types.includes("model")
          ? 2
          : types.some((type) => type !== "draft" && type !== "encoder")
            ? 1
            : 0;
    return { model, score };
  }).sort((left, right) => right.score - left.score);
  if (
    scored[0] === undefined
    || scored[0].score === 0
    || scored[0].score === scored[1]?.score
  ) {
    throw new Error(
      "multi-model package must identify exactly one decoder or target component",
    );
  }
  return scored[0].model;
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
