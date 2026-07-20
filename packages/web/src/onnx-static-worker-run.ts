import {
  analyzeStatic,
  buildTopology,
  resolveOnnxModelProfile,
} from "@inference-sim/core";
import { parseOnnxManifestFileText } from "./onnx-manifest-import.js";
import type {
  OnnxStaticBrowserConfig,
  OnnxStaticBrowserResult,
  WorkerRunProgressReporter,
} from "./types.js";

export function executeOnnxStaticWorkerRun(
  artifactText: string,
  sourceFileName: string,
  config: OnnxStaticBrowserConfig,
  reportProgress: WorkerRunProgressReporter = () => {},
): OnnxStaticBrowserResult {
  reportProgress({ progress: 12, phase: "Validating ONNX manifest" });
  const manifest = parseOnnxManifestFileText(artifactText, sourceFileName);
  reportProgress({ progress: 35, phase: "Resolving model profile" });
  const model = resolveOnnxModelProfile(manifest, {
    kvCacheQuantization: config.kvCacheQuantization,
    activationQuantization: config.activationQuantization,
  });
  reportProgress({ progress: 55, phase: "Building hardware topology" });
  const topology = buildTopology(config.hardwarePreset);
  reportProgress({ progress: 72, phase: "Analyzing capacity and roofline" });
  const analysis = analyzeStatic(topology, model, {
    batchSize: config.batchSize,
    inputSeqLen: config.inputSeqLen,
    outputSeqLen: config.outputSeqLen,
    parallelism: config.parallelism,
    memory: config.memory,
  });
  reportProgress({ progress: 92, phase: "Preparing model report" });
  return {
    sourceFileName,
    manifest: {
      revision: manifest.revision,
      fingerprint: manifest.manifestFingerprint,
      modelFileName: manifest.source.modelFileName,
      modelSha256: manifest.source.sha256,
      graphName: manifest.graph.name,
      nodeCount: manifest.graph.nodeCount,
      initializerCount: manifest.graph.initializerCount,
      initializerLogicalBytes: manifest.totals.initializerLogicalBytes,
      externalDataFiles: manifest.externalDataFiles.length,
      architectureSource: manifest.architecture.source,
      profileReadiness: manifest.profileReadiness,
    },
    config,
    model,
    analysis,
  };
}
