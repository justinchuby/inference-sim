import {
  buildTopology,
  resolveOnnxModelProfile,
  searchStaticConfigurations,
} from "@inference-sim/core";
import { parseOnnxManifestFileText } from "./onnx-manifest-import.js";
import type {
  OnnxSearchBrowserConfig,
  OnnxSearchBrowserResult,
  OnnxStaticBrowserConfig,
} from "./types.js";

const ALL_HARDWARE: readonly OnnxStaticBrowserConfig["hardwarePreset"][] = [
  "dgx-h100",
  "dgx-h200",
  "2x-dgx-h100",
  "4x-mac-studio-m4",
  "a100-4x",
  "rtx-4090-2x",
];

export function executeOnnxSearchWorkerRun(
  artifactText: string,
  sourceFileName: string,
  baseConfig: OnnxStaticBrowserConfig,
  searchConfig: OnnxSearchBrowserConfig,
): OnnxSearchBrowserResult {
  const manifest = parseOnnxManifestFileText(artifactText, sourceFileName);
  const model = resolveOnnxModelProfile(manifest, {
    kvCacheQuantization: baseConfig.kvCacheQuantization,
    activationQuantization: baseConfig.activationQuantization,
  });
  const hardware = searchConfig.topologyScope === "all"
    ? ALL_HARDWARE
    : [baseConfig.hardwarePreset];
  const kvCacheQuantizations =
    searchConfig.kvCacheScope === "fp16_fp8"
      ? ["fp16", "fp8"] as const
      : [baseConfig.kvCacheQuantization];
  const batchSizes = searchConfig.batchScope === "common"
    ? [1, 4, 16]
    : [baseConfig.batchSize];
  const parallelism = searchConfig.parallelismScope === "common"
    ? {
        tensorParallel: [1, 2, 4, 8],
        pipelineParallel: [1, 2, 4],
        expertParallel: [1, 2, 4, 8],
      }
    : {
        tensorParallel: [baseConfig.parallelism.tensorParallel],
        pipelineParallel: [baseConfig.parallelism.pipelineParallel],
        expertParallel: [baseConfig.parallelism.expertParallel],
      };
  const offloadStrategies = searchConfig.offloadScope === "none_partial"
    ? ["none", "partial"] as const
    : [baseConfig.memory.offloadStrategy];
  const result = searchStaticConfigurations(model, {
    objective: searchConfig.objective,
    topK: searchConfig.topK,
    maxCandidates: searchConfig.maxCandidates,
    constraints: {
      requireFeasible: true,
      maximumDeviceUsedFraction: searchConfig.maximumDeviceUsedFraction,
    },
    space: {
      topologies: hardware.map((id) => ({
        id,
        topology: buildTopology(id),
      })),
      kvCacheQuantizations,
      activationQuantizations: [baseConfig.activationQuantization],
      batchSizes,
      inputSeqLens: [baseConfig.inputSeqLen],
      outputSeqLens: [baseConfig.outputSeqLen],
      tensorParallel: parallelism.tensorParallel,
      pipelineParallel: parallelism.pipelineParallel,
      expertParallel: parallelism.expertParallel,
      dataParallel: [baseConfig.parallelism.dataParallel],
      memoryPolicies: offloadStrategies.map((offloadStrategy) => ({
        ...baseConfig.memory,
        offloadStrategy,
      })),
    },
  });
  return {
    sourceFileName,
    manifest: {
      fingerprint: manifest.manifestFingerprint,
      modelFileName: manifest.source.modelFileName,
    },
    modelName: model.name,
    baseConfig,
    searchConfig,
    result,
  };
}
