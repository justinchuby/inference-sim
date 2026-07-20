import { describe, expect, it } from "vitest";
import { simulateDashboard } from "./dashboard-simulation.js";
import type { DashboardRunConfig } from "./types.js";

const base: DashboardRunConfig = {
  scenarioName: "multi-gpu",
  mode: "speculative",
  seed: 42,
  speculative: {
    outputTokens: 64,
    draftWidth: 4,
    firstPositionAcceptance: 0.82,
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
    expect(result.speculative?.metrics.kvPagesAllocated).toBeGreaterThan(0);
    expect(result.topology.confidence).toBe("heuristic");
    expect(result.topology.metrics.totalDurationNs).toBeGreaterThan(0);
    expect(result.topology.operationCounts.collective).toBeGreaterThan(0);
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
