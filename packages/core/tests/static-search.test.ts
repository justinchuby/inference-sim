import { describe, expect, it } from "vitest";
import {
  buildModelProfile,
  buildTopology,
  searchStaticConfigurations,
  type MemoryPolicyConfig,
  type StaticSearchRequest,
} from "../src/index.js";

const memory: MemoryPolicyConfig = {
  kvCacheBudgetFraction: 0.3,
  expertCacheBudgetFraction: 0.2,
  pinnedPoolFraction: 0.05,
  offloadStrategy: "none",
  prefetchAhead: 0,
  pressureThreshold: 0.9,
  reclaimBatchSize: 4,
};

function request(): StaticSearchRequest {
  return {
    objective: "decode_throughput",
    topK: 5,
    maxCandidates: 100,
    constraints: { requireFeasible: true },
    space: {
      topologies: [
        { id: "dgx-h100", topology: buildTopology("dgx-h100") },
        { id: "rtx-4090-2x", topology: buildTopology("rtx-4090-2x") },
      ],
      kvCacheQuantizations: ["fp16", "fp8"],
      activationQuantizations: ["fp16"],
      batchSizes: [1, 4],
      inputSeqLens: [2048],
      outputSeqLens: [512],
      tensorParallel: [1, 2],
      pipelineParallel: [1],
      expertParallel: [1, 2],
      dataParallel: [1],
      memoryPolicies: [memory],
    },
  };
}

describe("static configuration search", () => {
  it("exhaustively ranks a declared space with stable IDs and rejection counts", () => {
    const model = buildModelProfile("llama-3-8b", "fp16", "fp16");
    const first = searchStaticConfigurations(model, request());
    const second = searchStaticConfigurations(model, request());

    expect(first).toEqual(second);
    expect(first.exhaustive).toBe(true);
    expect(first.declaredCandidateCount).toBe(32);
    expect(first.evaluatedCandidateCount).toBe(16);
    expect(first.rejectionCounts.expert_parallel_requires_moe).toBe(16);
    expect(first.candidates).toHaveLength(5);
    expect(new Set(first.candidates.map((entry) => entry.candidateId)).size)
      .toBe(first.candidates.length);
    expect(first.candidates.every((entry) => (
      entry.parallelism.expertParallel === 1 && entry.feasible
    ))).toBe(true);
    expect(first.candidates.map((entry) => entry.score)).toEqual(
      [...first.candidates.map((entry) => entry.score)]
        .sort((left, right) => right - left),
    );
  });

  it("applies runtime KV quantization without changing model weights", () => {
    const model = buildModelProfile("llama-3-8b", "fp16", "fp16");
    const searchRequest = request();
    searchRequest.objective = "device_headroom";
    searchRequest.space = {
      ...searchRequest.space,
      topologies: [{
        id: "rtx-4090-2x",
        topology: buildTopology("rtx-4090-2x"),
      }],
      batchSizes: [16],
      tensorParallel: [1],
      expertParallel: [1],
    };
    const result = searchStaticConfigurations(model, searchRequest);

    expect(result.candidates[0].kvCacheQuantization).toBe("fp8");
    expect(result.candidates[0].analysis.memoryBreakdown[0].weights)
      .toBe(result.candidates[1].analysis.memoryBreakdown[0].weights);
    expect(result.candidates[0].analysis.memoryBreakdown[0].kvCache)
      .toBeLessThan(result.candidates[1].analysis.memoryBreakdown[0].kvCache);
  });

  it("fails closed for oversized or duplicate candidate spaces", () => {
    const model = buildModelProfile("llama-3-8b");
    const oversized = request();
    oversized.maxCandidates = 31;
    expect(() => searchStaticConfigurations(model, oversized))
      .toThrow("exceeding maxCandidates");

    const duplicated = request();
    duplicated.space = {
      ...duplicated.space,
      batchSizes: [1, 1],
    };
    expect(() => searchStaticConfigurations(model, duplicated))
      .toThrow("batchSizes values must be unique");
  });

  it("reports monotonic candidate progress without changing the result", () => {
    const model = buildModelProfile("llama-3-8b");
    const progress: Array<{
      completedCandidates: number;
      totalCandidates: number;
    }> = [];
    const withProgress = searchStaticConfigurations(
      model,
      request(),
      (update) => progress.push(update),
    );
    const withoutProgress = searchStaticConfigurations(model, request());

    expect(withProgress).toEqual(withoutProgress);
    expect(progress[0]).toEqual({
      completedCandidates: 0,
      totalCandidates: 32,
    });
    expect(progress.at(-1)).toEqual({
      completedCandidates: 32,
      totalCandidates: 32,
    });
    expect(progress.every((entry, index) => (
      index === 0
      || entry.completedCandidates
        > progress[index - 1].completedCandidates
    ))).toBe(true);
  });
});
