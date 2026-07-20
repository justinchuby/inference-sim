import { describe, it, expect } from "vitest";
import { buildTopology, buildModelProfile, analyzeStatic } from "../src/index.js";
import type { PipelineConfig } from "../src/index.js";

describe("static analysis", () => {
  it("Mixtral-8x22B on 8×H100 (FP8) is feasible", () => {
    const topology = buildTopology("dgx-h100");
    const model = buildModelProfile("mixtral-8x22b", "fp8", "fp8");
    const pipeline: PipelineConfig = {
      batchSize: 32,
      inputSeqLen: 4096,
      outputSeqLen: 2048,
      parallelism: { tensorParallel: 4, pipelineParallel: 1, expertParallel: 2, dataParallel: 1 },
      memory: {
        kvCacheBudgetFraction: 0.4,
        expertCacheBudgetFraction: 0.3,
        pinnedPoolFraction: 0.1,
        offloadStrategy: "none",
        prefetchAhead: 2,
        pressureThreshold: 0.85,
        reclaimBatchSize: 4,
      },
    };

    const result = analyzeStatic(topology, model, pipeline);
    expect(result.feasible).toBe(true);
    expect(result.bottleneck).toBeDefined();
    expect(result.estimatedThroughput.decodeToksPerSec).toBeGreaterThan(0);
    expect(result.memoryBreakdown.length).toBe(8);
  });

  it("Llama-3-70B on single RTX 4090 is NOT feasible without offload", () => {
    const topology = buildTopology("rtx-4090-2x");
    const model = buildModelProfile("llama-3-70b", "fp16", "fp16");
    const pipeline: PipelineConfig = {
      batchSize: 1,
      inputSeqLen: 2048,
      outputSeqLen: 512,
      parallelism: { tensorParallel: 2, pipelineParallel: 1, expertParallel: 1, dataParallel: 1 },
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

    const result = analyzeStatic(topology, model, pipeline);
    expect(result.feasible).toBe(false);
    expect(result.recommendations.length).toBeGreaterThan(0);
  });

  it("Llama-3-8B on single 4090 fits easily", () => {
    const topology = buildTopology("rtx-4090-2x");
    const model = buildModelProfile("llama-3-8b", "fp16", "fp16");
    const pipeline: PipelineConfig = {
      batchSize: 1,
      inputSeqLen: 4096,
      outputSeqLen: 1024,
      parallelism: { tensorParallel: 1, pipelineParallel: 1, expertParallel: 1, dataParallel: 1 },
      memory: {
        kvCacheBudgetFraction: 0.4,
        expertCacheBudgetFraction: 0,
        pinnedPoolFraction: 0.05,
        offloadStrategy: "none",
        prefetchAhead: 0,
        pressureThreshold: 0.9,
        reclaimBatchSize: 4,
      },
    };

    const result = analyzeStatic(topology, model, pipeline);
    expect(result.feasible).toBe(true);
    expect(result.bottleneck).toBe("memory_bandwidth"); // decode is always mem-bw bound at batch=1
    expect(result.estimatedThroughput.decodeToksPerSec).toBeGreaterThan(30);
  });

  it("Apple Silicon M4 Max with DeepSeek-V2 (INT4) uses offload", () => {
    const topology = buildTopology("4x-mac-studio-m4");
    const model = buildModelProfile("deepseek-v2", "int4", "fp8");
    const pipeline: PipelineConfig = {
      batchSize: 1,
      inputSeqLen: 4096,
      outputSeqLen: 2048,
      parallelism: { tensorParallel: 1, pipelineParallel: 4, expertParallel: 1, dataParallel: 1 },
      memory: {
        kvCacheBudgetFraction: 0.3,
        expertCacheBudgetFraction: 0.4,
        pinnedPoolFraction: 0.05,
        offloadStrategy: "partial",
        prefetchAhead: 3,
        pressureThreshold: 0.8,
        reclaimBatchSize: 8,
      },
    };

    const result = analyzeStatic(topology, model, pipeline);
    // Should be feasible with offload
    expect(result.feasible).toBe(true);
    expect(result.estimatedThroughput.decodeToksPerSec).toBeGreaterThan(0);
  });

  it("does not make offload feasible when host capacity is insufficient", () => {
    const topology = buildTopology("rtx-4090-2x");
    topology.nodes[0].hostMemory.capacityBytes = 1;
    const model = buildModelProfile("llama-3-70b", "fp16", "fp16");
    const pipeline: PipelineConfig = {
      batchSize: 1,
      inputSeqLen: 2048,
      outputSeqLen: 512,
      parallelism: {
        tensorParallel: 2,
        pipelineParallel: 1,
        expertParallel: 1,
        dataParallel: 1,
      },
      memory: {
        kvCacheBudgetFraction: 0.3,
        expertCacheBudgetFraction: 0,
        pinnedPoolFraction: 0.05,
        offloadStrategy: "full",
        prefetchAhead: 0,
        pressureThreshold: 0.9,
        reclaimBatchSize: 4,
      },
    };

    const result = analyzeStatic(topology, model, pipeline);
    expect(result.feasible).toBe(false);
    expect(result.bottleneck).toBe("capacity");
    expect(result.hostMemoryBreakdown.free).toBeLessThan(0);
  });

  it("checks every device capacity instead of trusting the first device", () => {
    const topology = buildTopology("rtx-4090-2x");
    expect(topology.nodes[0].devices[0].memory).not.toBe(
      topology.nodes[0].devices[1].memory,
    );
    topology.nodes[0].devices[0].memory.capacityBytes = 1024 ** 4;
    topology.nodes[0].devices[1].memory.capacityBytes = 1;
    const model = buildModelProfile("llama-3-8b", "fp16", "fp16");
    const pipeline: PipelineConfig = {
      batchSize: 1,
      inputSeqLen: 1024,
      outputSeqLen: 128,
      parallelism: {
        tensorParallel: 2,
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

    const result = analyzeStatic(topology, model, pipeline);
    expect(result.feasible).toBe(false);
    expect(result.memoryBreakdown[0].free).toBeGreaterThan(0);
    expect(result.memoryBreakdown[1].free).toBeLessThan(0);
  });
});
