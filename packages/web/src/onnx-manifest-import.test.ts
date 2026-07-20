import { describe, expect, it } from "vitest";
import {
  ONNX_MODEL_MANIFEST_KIND,
  ONNX_MODEL_MANIFEST_REVISION,
  createOnnxModelManifest,
  serializeOnnxModelManifest,
} from "@inference-sim/core";
import { parseOnnxManifestFileText } from "./onnx-manifest-import.js";
import { executeOnnxStaticWorkerRun } from "./onnx-static-worker-run.js";
import { executeOnnxSearchWorkerRun } from "./onnx-search-worker-run.js";
import type {
  OnnxSearchBrowserConfig,
  OnnxStaticBrowserConfig,
  WorkerRunProgress,
} from "./types.js";

function manifestText(): string {
  return serializeOnnxModelManifest(createOnnxModelManifest({
    kind: ONNX_MODEL_MANIFEST_KIND,
    revision: ONNX_MODEL_MANIFEST_REVISION,
    source: {
      modelFileName: "tiny.onnx",
      modelByteLength: 100,
      sha256: "a".repeat(64),
    },
    model: {
      irVersion: "11",
      producerName: "test",
      producerVersion: "",
      domain: "",
      modelVersion: "0",
    },
    graph: {
      name: "tiny",
      nodeCount: 1,
      initializerCount: 1,
      inputNames: ["input"],
      outputNames: ["output"],
      operators: [{ domain: "ai.onnx", opType: "MatMul", count: 1 }],
    },
    initializers: [{
      name: "weight",
      dataType: "float16",
      dimensions: [16, 16],
      elementCount: 256,
      logicalByteLength: 512,
      storage: { kind: "inline", byteLength: 512 },
    }],
    externalDataFiles: [],
    architecture: {
      source: "onnx_genai_manifest",
      modelType: "TinyLM",
      hiddenSize: 16,
      intermediateSize: 32,
      numHiddenLayers: 1,
      numAttentionHeads: 2,
      numKeyValueHeads: 2,
      headDimension: 8,
      vocabSize: 32,
    },
    totals: {
      initializerElements: 256,
      initializerLogicalBytes: 512,
      inlineInitializerBytes: 512,
      externalInitializerBytes: 0,
    },
    profileReadiness: { ready: true, missingFields: [] },
  }));
}

const config: OnnxStaticBrowserConfig = {
  hardwarePreset: "rtx-4090-2x",
  kvCacheQuantization: "fp8",
  activationQuantization: "fp16",
  batchSize: 1,
  inputSeqLen: 128,
  outputSeqLen: 32,
  parallelism: {
    tensorParallel: 1,
    pipelineParallel: 1,
    expertParallel: 1,
    dataParallel: 1,
  },
  memory: {
    kvCacheBudgetFraction: 0.3,
    expertCacheBudgetFraction: 0,
    pinnedPoolFraction: 0.05,
    offloadStrategy: "none",
    prefetchAhead: 0,
    pressureThreshold: 0.9,
    reclaimBatchSize: 4,
  },
};

const searchConfig: OnnxSearchBrowserConfig = {
  objective: "decode_throughput",
  topologyScope: "all",
  kvCacheScope: "fp16_fp8",
  batchScope: "common",
  parallelismScope: "common",
  offloadScope: "none_partial",
  maximumDeviceUsedFraction: 0.9,
  topK: 10,
  maxCandidates: 10_000,
};

describe("ONNX manifest browser import", () => {
  it("validates the shared manifest and runs static analysis", () => {
    const text = manifestText();
    const manifest = parseOnnxManifestFileText(text, "tiny.json");
    const result = executeOnnxStaticWorkerRun(text, "tiny.json", config);

    expect(result.manifest.fingerprint).toBe(manifest.manifestFingerprint);
    expect(result.model.name).toBe("TinyLM");
    expect(result.model.quantization.kvCache).toBe("fp8");
    expect(result.analysis.feasible).toBe(true);
  });

  it("reports static-analysis phases without changing the result", () => {
    const text = manifestText();
    const progress: WorkerRunProgress[] = [];
    const baseline = executeOnnxStaticWorkerRun(text, "tiny.json", config);
    const observed = executeOnnxStaticWorkerRun(
      text,
      "tiny.json",
      config,
      (update) => progress.push(update),
    );

    expect(observed).toEqual(baseline);
    assertMonotonicProgress(progress, 92, "Preparing model report");
  });

  it("rejects stale revisions and non-JSON files", () => {
    const stale = JSON.parse(manifestText()) as Record<string, unknown>;
    stale.revision = 1;
    expect(() => parseOnnxManifestFileText(
      JSON.stringify(stale),
      "tiny.json",
    )).toThrow("revision");
    expect(() => parseOnnxManifestFileText(manifestText(), "tiny.yaml"))
      .toThrow("must use .json");
  });

  it("runs deterministic bounded search through the Worker boundary", () => {
    const text = manifestText();
    const first = executeOnnxSearchWorkerRun(
      text,
      "tiny.json",
      config,
      searchConfig,
    );
    const second = executeOnnxSearchWorkerRun(
      text,
      "tiny.json",
      config,
      searchConfig,
    );

    expect(first).toEqual(second);
    expect(first.result.declaredCandidateCount).toBe(3456);
    expect(first.result.returnedCandidateCount).toBe(10);
    expect(first.result.rejectionCounts).toMatchObject({
      expert_parallel_requires_moe: expect.any(Number),
      pipeline_parallel_exceeds_layers: expect.any(Number),
    });
    expect(first.result.candidates.every((candidate) => candidate.feasible))
      .toBe(true);
  });

  it("reports bounded candidate progress without changing search ranking", () => {
    const text = manifestText();
    const progress: WorkerRunProgress[] = [];
    const baseline = executeOnnxSearchWorkerRun(
      text,
      "tiny.json",
      config,
      searchConfig,
    );
    const observed = executeOnnxSearchWorkerRun(
      text,
      "tiny.json",
      config,
      searchConfig,
      (update) => progress.push(update),
    );

    expect(observed).toEqual(baseline);
    assertMonotonicProgress(progress, 94, "Preparing candidate ranking");
    expect(progress.some((entry) => (
      entry.progress === 88
      && entry.phase.startsWith("Evaluating candidates ")
    ))).toBe(true);
  });
});

function assertMonotonicProgress(
  progress: readonly WorkerRunProgress[],
  finalProgress: number,
  finalPhase: string,
): void {
  expect(progress.length).toBeGreaterThan(0);
  expect(progress.at(-1)).toEqual({
    progress: finalProgress,
    phase: finalPhase,
  });
  expect(progress.every((entry, index) => (
    entry.phase.length > 0
    && entry.progress >= 0
    && entry.progress <= 99
    && (index === 0 || entry.progress > progress[index - 1]!.progress)
  ))).toBe(true);
}
