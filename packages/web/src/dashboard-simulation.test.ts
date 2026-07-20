import { describe, expect, it } from "vitest";
import { simulateDashboard } from "./dashboard-simulation.js";
import type { DashboardRunConfig } from "./types.js";

const base: DashboardRunConfig = {
  scenarioName: "multi-gpu",
  mode: "speculative",
  seed: 42,
  speculative: {
    family: "mtp",
    outputTokens: 64,
    draftWidth: 4,
    firstPositionAcceptance: 0.82,
  },
  serving: {
    requestCount: 8,
    arrivalGapUs: 100,
    promptTokens: 128,
    outputTokens: 16,
    maxBatchSize: 4,
    maxBatchTokens: 64,
    prefillChunkTokens: 32,
  },
  expertCache: {
    tokenCount: 32,
    topK: 2,
    expertCount: 12,
    hotSlots: 4,
    warmSlots: 6,
  },
};

describe("simulateDashboard", () => {
  it("runs bounded speculative simulation with paged KV metrics", () => {
    const result = simulateDashboard(base);

    expect(result.scenario.deviceCount).toBe(3);
    expect(result.speculative?.finalTokenLength).toBe(2112);
    expect(result.speculative?.family).toBe("mtp");
    expect(result.speculative?.metrics.kvPagesAllocated).toBeGreaterThan(0);
    expect(result.topology.confidence).toBe("heuristic");
    expect(result.topology.metrics.totalDurationNs).toBeGreaterThan(0);
    expect(result.topology.operationCounts.collective).toBeGreaterThan(0);
  });

  it("runs every selectable proposer family through the shared core contract", () => {
    const families = [
      "prompt_lookup",
      "draft_model",
      "mtp",
      "eagle3",
      "shared_kv",
      "self_speculative",
    ] as const;
    for (const family of families) {
      const result = simulateDashboard({
        ...base,
        speculative: { ...base.speculative, family },
      });
      expect(result.speculative?.family).toBe(family);
      expect(result.speculative?.finalTokenLength).toBe(2112);
    }
  });

  it("runs deterministic expert cache simulation", () => {
    const config = { ...base, mode: "expert-cache" as const };
    const first = simulateDashboard(config);
    const second = simulateDashboard(config);

    expect(first).toEqual(second);
    expect(first.expertCache?.routes).toHaveLength(32);
    expect(first.expertCache?.metrics.hotHitRate).toBeGreaterThanOrEqual(0);
    expect(first.topology.operationCounts.transfer).toBeGreaterThan(0);
  });

  it("runs continuous serving with replayed request timing", () => {
    const config = { ...base, mode: "serving" as const };
    const result = simulateDashboard(config);

    expect(result.serving?.requests).toHaveLength(8);
    expect(result.serving?.metrics.outputTokens).toBe(128);
    expect(result.serving?.metrics.p95TimeToFirstTokenNs).toBeGreaterThan(0);
    expect(result.serving?.metrics.kvHighWaterTokens).toBeGreaterThan(0);
    expect(result.serving?.batches.length).toBeGreaterThan(1);
    expect(result.topology.planSteps).toBeGreaterThan(0);
  });

  it("changes modeled latency when the same workload changes topology", () => {
    const multiGpu = simulateDashboard(base);
    const cpu = simulateDashboard({ ...base, scenarioName: "cpu-only" });

    expect(cpu.topology.metrics.totalDurationNs).toBeGreaterThan(
      multiGpu.topology.metrics.totalDurationNs,
    );
    expect(cpu.topology.metrics.tokensPerSecond).toBeLessThan(
      multiGpu.topology.metrics.tokensPerSecond,
    );
  });
});
