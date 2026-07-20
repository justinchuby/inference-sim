import {
  analyzeStatic,
  buildTopology,
  resolveOnnxModelProfile,
} from "@inference-sim/core";
import { parseOnnxManifestFileText } from "./onnx-manifest-import.js";
import type {
  OnnxStaticBrowserConfig,
  OnnxStaticBrowserResult,
} from "./types.js";

export function executeOnnxStaticWorkerRun(
  artifactText: string,
  sourceFileName: string,
  config: OnnxStaticBrowserConfig,
): OnnxStaticBrowserResult {
  const manifest = parseOnnxManifestFileText(artifactText, sourceFileName);
  const model = resolveOnnxModelProfile(manifest, {
    kvCacheQuantization: config.kvCacheQuantization,
    activationQuantization: config.activationQuantization,
  });
  const topology = buildTopology(config.hardwarePreset);
  const analysis = analyzeStatic(topology, model, {
    batchSize: config.batchSize,
    inputSeqLen: config.inputSeqLen,
    outputSeqLen: config.outputSeqLen,
    parallelism: config.parallelism,
    memory: config.memory,
  });
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
