import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { parseCalibrationFileText } from "./calibration-import.js";
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
    compareTopologies: false,
    decodeMode: "mtp",
    draftWidth: 4,
    firstPositionAcceptance: 0.82,
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
    expect(result.serving?.decodeMode).toBe("mtp");
    expect(result.serving?.metrics.proposedDraftTokens).toBeGreaterThan(0);
    expect(result.serving?.metrics.acceptedDraftTokens).toBeGreaterThan(0);
    expect(result.serving?.batches.length).toBeGreaterThan(1);
    expect(result.topology.planSteps).toBeGreaterThan(0);
  });

  it("preserves an explicit target-only serving baseline", () => {
    const result = simulateDashboard({
      ...base,
      mode: "serving",
      serving: { ...base.serving, decodeMode: "target_only" },
    });

    expect(result.serving?.support).toBe("target_only");
    expect(result.serving?.metrics.proposedDraftTokens).toBe(0);
    expect(result.serving?.metrics.acceptedDraftTokens).toBe(0);
    expect(result.serving?.metrics.committedTokensPerTargetForward).toBe(1);
  });

  it("compares the same serving workload across all six topologies", () => {
    const config: DashboardRunConfig = {
      ...base,
      mode: "serving",
      serving: { ...base.serving, compareTopologies: true },
    };
    const first = simulateDashboard(config);
    const second = simulateDashboard(config);

    expect(first).toEqual(second);
    expect(first.comparison).toHaveLength(6);
    expect(first.comparison?.map((entry) => entry.rank))
      .toEqual([1, 2, 3, 4, 5, 6]);
    expect(first.comparison?.[0]).toMatchObject({
      scenarioId: "multi-gpu",
      relativeToFastest: 1,
    });
    expect(first.comparison?.at(-1)?.scenarioId).toBe("cpu-only");
    expect(first.scenario.id).toBe("multi-gpu");
    expect(first.comparison?.every((entry, index, entries) => (
      index === 0
      || entry.totalDurationNs >= entries[index - 1].totalDurationNs
    ))).toBe(true);
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

  it("runs an imported calibration through the dashboard worker contract", async () => {
    const text = await readFile(new URL(
      "../../../examples/calibration-synthetic.yaml",
      import.meta.url,
    ), "utf8");
    const calibration = await parseCalibrationFileText(
      text,
      "calibration-synthetic.yaml",
    );
    const result = simulateDashboard({
      ...base,
      calibration: calibration.dataset,
    });

    expect(result.calibration).toMatchObject({
      datasetId: "synthetic-linear-example",
      datasetFingerprint: calibration.fit.datasetFingerprint,
      evidenceKind: "synthetic",
      fitConfidence: "heuristic",
    });
    expect(result.calibration?.diagnostics).toHaveLength(15);
    expect(result.topology.assumptions[0]).toContain(
      calibration.fit.datasetFingerprint,
    );
  });

  it("rejects dashboard work outside imported interpolation ranges", async () => {
    const text = await readFile(new URL(
      "../../../examples/calibration-synthetic.yaml",
      import.meta.url,
    ), "utf8");
    const calibration = await parseCalibrationFileText(
      text,
      "calibration-synthetic.yaml",
    );

    expect(() => simulateDashboard({
      ...base,
      mode: "serving",
      calibration: calibration.dataset,
      serving: {
        ...base.serving,
        decodeMode: "target_only",
        promptTokens: 512,
        maxBatchTokens: 512,
        prefillChunkTokens: 512,
      },
    })).toThrow("outside calibrated range 1..128");
  });
});
