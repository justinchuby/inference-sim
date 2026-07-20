import { describe, expect, it } from "vitest";
import {
  SCENARIO_PRESET_NAMES,
  buildScenarioPreset,
  compareTopologyServingWorkloads,
  defaultSpeculativeEligibility,
  simulateTopologyServingWorkload,
  type ServingSchedulerConfig,
  type SpeculativeProposerFamily,
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

const FAMILIES: readonly SpeculativeProposerFamily[] = [
  "prompt_lookup",
  "draft_model",
  "mtp",
  "eagle3",
  "shared_kv",
  "self_speculative",
];

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

  it("executes speculative serving for every proposer and topology family", () => {
    const requests = [
      { id: "r0", arrivalNs: 0, promptTokens: 4, outputTokens: 6 },
      { id: "r1", arrivalNs: 50_000, promptTokens: 6, outputTokens: 6 },
    ];
    for (const family of FAMILIES) {
      const config: ServingSchedulerConfig = {
        requests,
        maxBatchSize: 2,
        maxBatchTokens: 8,
        prefillChunkTokens: 4,
        maxKvTokens: 22,
        speculative: {
          family,
          eligibility: defaultSpeculativeEligibility(family),
          maxAdditionalTokens: 2,
          acceptance: {
            kind: "conditional_empirical",
            matchProbabilityByPosition: [0.8, 0.6],
            seed: 17,
          },
        },
      };
      for (const scenarioName of SCENARIO_PRESET_NAMES) {
        const result = simulateTopologyServingWorkload(
          buildScenarioPreset(scenarioName),
          config,
        );
        expect(
          result.serving.metrics.outputTokens,
          `${family}:${scenarioName}`,
        ).toBe(12);
        expect(
          result.serving.metrics.proposedDraftTokens,
          `${family}:${scenarioName}`,
        ).toBeGreaterThan(0);
        expect(
          result.serving.replay.finalKvTokens,
          `${family}:${scenarioName}`,
        ).toBe(0);
        expect(result.batches.every((batch) => (
          batch.topology.execution.status === "succeeded"
        )), `${family}:${scenarioName}`).toBe(true);
      }
    }
  });

  it("can trade proposer work for fewer target forwards", () => {
    const base: ServingSchedulerConfig = {
      requests: [
        { id: "r0", arrivalNs: 0, promptTokens: 8, outputTokens: 16 },
        { id: "r1", arrivalNs: 0, promptTokens: 8, outputTokens: 16 },
      ],
      maxBatchSize: 2,
      maxBatchTokens: 8,
      prefillChunkTokens: 8,
      maxKvTokens: 46,
    };
    const scenario = buildScenarioPreset("multi-gpu");
    const targetOnly = simulateTopologyServingWorkload(scenario, base);
    const speculative = simulateTopologyServingWorkload(scenario, {
      ...base,
      speculative: {
        family: "mtp",
        eligibility: defaultSpeculativeEligibility("mtp"),
        maxAdditionalTokens: 3,
        acceptance: {
          kind: "conditional_empirical",
          matchProbabilityByPosition: [1, 1, 1],
          seed: 1,
        },
      },
    });

    expect(speculative.serving.metrics.targetForwards).toBeLessThan(
      targetOnly.serving.metrics.targetForwards,
    );
    expect(speculative.serving.metrics.proposedDraftTokens).toBeGreaterThan(0);
    expect(speculative.metrics.totalDurationNs).toBeLessThan(
      targetOnly.metrics.totalDurationNs,
    );
  });

  it("ranks the same serving workload across every topology", () => {
    const scenarios = SCENARIO_PRESET_NAMES.map(buildScenarioPreset);
    const config: ServingSchedulerConfig = {
      ...workload,
      speculative: {
        family: "mtp",
        eligibility: defaultSpeculativeEligibility("mtp"),
        maxAdditionalTokens: 2,
        acceptance: {
          kind: "conditional_empirical",
          matchProbabilityByPosition: [0.8, 0.6],
          seed: 19,
        },
      },
    };
    const first = compareTopologyServingWorkloads(scenarios, config);
    const second = compareTopologyServingWorkloads(scenarios, config);

    expect(first).toEqual(second);
    expect(first.runs).toHaveLength(6);
    expect(first.runs.map((run) => run.rank)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(first.runs[0].relativeToFastest).toBe(1);
    expect(first.runs.every((run) => (
      run.result.serving.replay.completedRequests === 3
      && run.result.serving.metrics.outputTokens === 8
      && run.result.serving.replay.finalKvTokens === 0
    ))).toBe(true);
    expect(new Set(first.runs.map((run) => (
      run.result.scenarioId
    )))).toEqual(new Set(SCENARIO_PRESET_NAMES));
    expect(first.runs.at(-1)?.result.scenarioId).toBe("cpu-only");
  });

  it("rejects empty and duplicate serving comparison scenarios", () => {
    expect(() => compareTopologyServingWorkloads([], workload))
      .toThrow("at least one scenario");
    const scenario = buildScenarioPreset("multi-gpu");
    expect(() => compareTopologyServingWorkloads(
      [scenario, scenario],
      workload,
    )).toThrow("scenario id must be unique");
  });
});
