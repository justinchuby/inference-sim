import { describe, expect, it } from "vitest";
import {
  SCENARIO_PRESET_NAMES,
  buildScenarioPreset,
  simulateTopologyServingWorkload,
  type ServingSchedulerConfig,
} from "../src/index.js";

const workload: ServingSchedulerConfig = {
  requests: [
    { id: "r0", arrivalNs: 0, promptTokens: 8, outputTokens: 3 },
    { id: "r1", arrivalNs: 100_000, promptTokens: 4, outputTokens: 2 },
    { id: "r2", arrivalNs: 100_000, promptTokens: 12, outputTokens: 3 },
  ],
  maxBatchSize: 3,
  maxBatchTokens: 8,
  prefillChunkTokens: 4,
  maxKvTokens: 32,
};

describe("topology-aware serving", () => {
  it("executes and replays dynamic batches on every topology family", () => {
    for (const scenarioName of SCENARIO_PRESET_NAMES) {
      const result = simulateTopologyServingWorkload(
        buildScenarioPreset(scenarioName),
        workload,
      );

      expect(result.serving.replay.completedRequests, scenarioName).toBe(3);
      expect(result.serving.metrics.outputTokens, scenarioName).toBe(8);
      expect(result.batches.length, scenarioName).toBeGreaterThan(1);
      expect(result.batches.every((batch) => (
        batch.topology.execution.status === "succeeded"
      )), scenarioName).toBe(true);
      expect(result.metrics.planSteps, scenarioName).toBeGreaterThan(0);
      expect(result.metrics.totalDurationNs, scenarioName).toBeGreaterThan(0);
      expect(result.metrics.idleNs, scenarioName).toBeGreaterThanOrEqual(0);
    }
  });

  it("preserves deterministic topology-relative latency", () => {
    const gpu = simulateTopologyServingWorkload(
      buildScenarioPreset("multi-gpu"),
      workload,
    );
    const cpu = simulateTopologyServingWorkload(
      buildScenarioPreset("cpu-only"),
      workload,
    );
    const repeat = simulateTopologyServingWorkload(
      buildScenarioPreset("multi-gpu"),
      workload,
    );

    expect(cpu.metrics.totalDurationNs).toBeGreaterThan(
      gpu.metrics.totalDurationNs,
    );
    expect(gpu).toEqual(repeat);
  });

  it("distinguishes intermediate prefill from output-producing batches", () => {
    const result = simulateTopologyServingWorkload(
      buildScenarioPreset("single-gpu-cpu"),
      {
        ...workload,
        requests: [
          { id: "long", arrivalNs: 0, promptTokens: 10, outputTokens: 1 },
        ],
        maxBatchTokens: 4,
        prefillChunkTokens: 4,
        maxKvTokens: 10,
      },
    );

    expect(result.batches.map((batch) => batch.work.expectedOutputTokens))
      .toEqual([0, 0, 1]);
    expect(result.batches.map((batch) => (
      batch.topology.metrics.committedTokens
    ))).toEqual([0, 0, 1]);
  });
});
