import { describe, expect, it } from "vitest";
import {
  SCENARIO_PRESET_NAMES,
  SERVING_EXPERT_CACHE_CONTRACT_REVISION,
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

  it("composes speculative serving with persistent routed-expert cache state", () => {
    const expertBytes = 64 * 1024 ** 2;
    const result = simulateTopologyServingWorkload(
      buildScenarioPreset("multi-gpu"),
      {
        requests: [
          { id: "moe", arrivalNs: 0, promptTokens: 4, outputTokens: 5 },
        ],
        maxBatchSize: 1,
        maxBatchTokens: 4,
        prefillChunkTokens: 4,
        maxKvTokens: 12,
        speculative: {
          family: "mtp",
          eligibility: defaultSpeculativeEligibility("mtp"),
          maxAdditionalTokens: 2,
          acceptance: {
            kind: "conditional_empirical",
            matchProbabilityByPosition: [1, 1],
            seed: 3,
          },
        },
      },
      undefined,
      {
        contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
        cache: {
          experts: [{ id: "e0", bytes: expertBytes }],
          hotCapacityBytes: expertBytes,
          warmCapacityBytes: expertBytes,
          warmToHotLatencyNs: 2_000,
          coldToHotLatencyNs: 20_000,
          coldToWarmLatencyNs: 10_000,
          routingSeed: 7,
        },
        topK: 1,
      },
    );
    const expertCache = result.expertCache;
    if (expertCache === undefined) {
      throw new Error("missing composed expert cache");
    }

    expect(result.serving.metrics.outputTokens).toBe(5);
    expect(result.serving.metrics.proposedDraftTokens).toBeGreaterThan(0);
    expect(expertCache.routes).toHaveLength(
      result.batches.reduce((sum, batch) => sum + batch.work.tokenWork, 0),
    );
    expect(expertCache.snapshot.metrics.coldMisses).toBe(1);
    expect(expertCache.snapshot.metrics.hotHits)
      .toBe(expertCache.routes.length - 1);
    expect(expertCache.replay.snapshot).toEqual(expertCache.snapshot);
    expect(result.physical?.execution.trace.admissions).toHaveLength(
      result.batches.length,
    );
    expect(result.physical?.replay.appliedEvents).toBeGreaterThan(0);
    expect(result.physical?.replay.completedAtNs).toBe(
      result.physical?.execution.completedAtNs,
    );
    expect(result.batches.every((batch) => (
      batch.physicalExecution?.executionId
        === batch.topology.plan.executionId
      && batch.foregroundCompletedAtNs
        <= batch.startedAtNs + batch.durationNs
    ))).toBe(true);
    expect(result.batches.length).toBeGreaterThan(1);
    expect(result.batches[0].expertRoutes[0].sourceTiers).toEqual(["cold"]);
    expect(result.batches.slice(1).every((batch) => (
      batch.expertRoutes.every((route) => (
        route.sourceTiers.every((tier) => tier === "hot")
      ))
    ))).toBe(true);
    expect(result.batches.every((batch) => (
      batch.durationNs === Math.max(
        batch.cacheConstraintNs,
        batch.topology.metrics.totalDurationNs,
      )
    ))).toBe(true);
    expect(result.metrics.allToAllOperations).toBeGreaterThan(0);
  });

  it("executes composed serving on every required device topology", () => {
    const expertBytes = 64 * 1024 ** 2;
    const config: ServingSchedulerConfig = {
      requests: [
        { id: "matrix", arrivalNs: 0, promptTokens: 2, outputTokens: 3 },
      ],
      maxBatchSize: 1,
      maxBatchTokens: 2,
      prefillChunkTokens: 2,
      maxKvTokens: 8,
      speculative: {
        family: "mtp",
        eligibility: defaultSpeculativeEligibility("mtp"),
        maxAdditionalTokens: 1,
        acceptance: {
          kind: "conditional_empirical",
          matchProbabilityByPosition: [1],
          seed: 5,
        },
      },
    };
    for (const scenarioName of SCENARIO_PRESET_NAMES) {
      const result = simulateTopologyServingWorkload(
        buildScenarioPreset(scenarioName),
        config,
        undefined,
        {
          contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
          cache: {
            experts: [{ id: "e0", bytes: expertBytes }],
            hotCapacityBytes: expertBytes,
            warmCapacityBytes: expertBytes,
            warmToHotLatencyNs: 2_000,
            coldToHotLatencyNs: 20_000,
            coldToWarmLatencyNs: 10_000,
            routingSeed: 7,
          },
          topK: 1,
        },
      );

      expect(result.serving.replay.completedRequests, scenarioName).toBe(1);
      expect(result.expertCache?.replay.snapshot, scenarioName)
        .toEqual(result.expertCache?.snapshot);
      expect(result.batches.every((batch) => (
        batch.topology.execution.status === "succeeded"
      )), scenarioName).toBe(true);
      expect(
        result.metrics.allToAllOperations > 0,
        scenarioName,
      ).toBe(scenarioName === "multi-gpu" || scenarioName === "multi-node");
    }
  });

  it("fails closed for unsupported composed expert-cache contracts", () => {
    const scenario = buildScenarioPreset("single-gpu-cpu");
    const expertBytes = 64 * 1024 ** 2;
    const cache = {
      experts: [{ id: "e0", bytes: expertBytes }],
      hotCapacityBytes: expertBytes,
      warmCapacityBytes: expertBytes,
      warmToHotLatencyNs: 2_000,
      coldToHotLatencyNs: 20_000,
      coldToWarmLatencyNs: 10_000,
      routingSeed: 7,
    } as const;

    expect(() => simulateTopologyServingWorkload(
      scenario,
      workload,
      undefined,
      {
        contractRevision: 99 as 2,
        cache,
        topK: 1,
      },
    )).toThrow("unsupported serving expert-cache contract revision");
    expect(() => simulateTopologyServingWorkload(
      scenario,
      workload,
      undefined,
      {
        contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
        cache,
        topK: 2,
      },
    )).toThrow("topK 2 exceeds 1 experts");
  });

  it("retimes adaptive prefetch from the shared physical storage trace", () => {
    const expertBytes = 64 * 1024 ** 2;
    const result = simulateTopologyServingWorkload(
      buildScenarioPreset("multi-gpu"),
      {
        requests: [
          { id: "adaptive", arrivalNs: 0, promptTokens: 4, outputTokens: 3 },
        ],
        maxBatchSize: 1,
        maxBatchTokens: 2,
        prefillChunkTokens: 2,
        maxKvTokens: 10,
      },
      undefined,
      {
        contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
        cache: {
          experts: Array.from({ length: 4 }, (_, index) => ({
            id: `e${index}`,
            bytes: expertBytes,
          })),
          hotCapacityBytes: 2 * expertBytes,
          warmCapacityBytes: 2 * expertBytes,
          warmToHotLatencyNs: 400_000,
          coldToHotLatencyNs: 2_200_000,
          coldToWarmLatencyNs: 1_500_000,
          routingSeed: 7,
          adaptivePrefetch: {
            targetTier: "warm",
            minObservations: 1,
            intervalTokens: 1,
            maxExpertsPerDecision: 1,
          },
        },
        topK: 1,
      },
    );
    const trace = result.expertCache?.trace ?? [];
    const prefetchLoads = trace.filter((event) => (
      event.kind === "load_start" && event.load.kind === "prefetch"
    ));
    const retimes = trace.filter((event) => event.kind === "load_retime");
    const physicalRetimes = retimes.filter(
      (event) => event.physicalCompletesAtNs !== undefined,
    );
    const storageTransfers = result.physical?.execution.trace.operations.filter(
      ({ event }) => (
        event.kind === "transfer"
        && event.resources.some((resource) => (
          resource.resourceId.endsWith(":storage-read")
        ))
      ),
    ) ?? [];

    expect(prefetchLoads.length).toBeGreaterThan(0);
    expect(retimes.length).toBe(prefetchLoads.length * 2);
    expect(physicalRetimes).toHaveLength(prefetchLoads.length);
    expect(storageTransfers.length).toBeGreaterThan(0);
    for (const retime of physicalRetimes) {
      const physicalTerminals = result.batches.flatMap((batch) => (
        batch.topology.backgroundPrefetchTerminals
          .filter((terminal) => terminal.prefetchId === retime.loadId)
          .map((terminal) => result.physical?.execution.trace.operations.find(
            ({ event }) => (
              event.executionId === batch.topology.plan.executionId
              && event.stepId === terminal.stepId
            ),
          )?.event)
      ));
      expect(physicalTerminals.length, retime.loadId).toBeGreaterThan(0);
      expect(
        physicalTerminals.every((event) => event?.kind === "transfer"),
        retime.loadId,
      ).toBe(true);
      expect(
        retime.physicalCompletesAtNs,
        retime.loadId,
      ).toBe(Math.max(...physicalTerminals.map((event) => event?.finishNs ?? 0)));
      expect(
        retime.completesAtNs,
        retime.loadId,
      ).toBeGreaterThanOrEqual(retime.physicalCompletesAtNs ?? 0);
    }
    expect(result.expertCache?.replay.snapshot)
      .toEqual(result.expertCache?.snapshot);
    expect(result.physical?.replay.completedAtNs)
      .toBe(result.physical?.execution.completedAtNs);
  });

  it("closes adaptive prefetch reservations on every required topology", () => {
    const expertBytes = 64 * 1024 ** 2;
    for (const scenarioName of SCENARIO_PRESET_NAMES) {
      const result = simulateTopologyServingWorkload(
        buildScenarioPreset(scenarioName),
        {
          requests: [
            {
              id: "adaptive-matrix",
              arrivalNs: 0,
              promptTokens: 4,
              outputTokens: 2,
            },
          ],
          maxBatchSize: 1,
          maxBatchTokens: 2,
          prefillChunkTokens: 2,
          maxKvTokens: 9,
        },
        undefined,
        {
          contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
          cache: {
            experts: Array.from({ length: 4 }, (_, index) => ({
              id: `e${index}`,
              bytes: expertBytes,
            })),
            hotCapacityBytes: 2 * expertBytes,
            warmCapacityBytes: 2 * expertBytes,
            warmToHotLatencyNs: 400_000,
            coldToHotLatencyNs: 2_200_000,
            coldToWarmLatencyNs: 1_500_000,
            routingSeed: 7,
            adaptivePrefetch: {
              targetTier: "warm",
              minObservations: 1,
              intervalTokens: 1,
              maxExpertsPerDecision: 1,
            },
          },
          topK: 1,
        },
      );
      const trace = result.expertCache?.trace ?? [];
      const prefetchLoads = trace.filter((event) => (
        event.kind === "load_start" && event.load.kind === "prefetch"
      ));
      const physicalRetimes = trace.filter((event) => (
        event.kind === "load_retime"
        && event.physicalCompletesAtNs !== undefined
      ));

      expect(prefetchLoads.length, scenarioName).toBeGreaterThan(0);
      expect(physicalRetimes, scenarioName).toHaveLength(
        prefetchLoads.length,
      );
      expect(result.expertCache?.snapshot.pendingLoads, scenarioName)
        .toHaveLength(0);
      expect(result.expertCache?.snapshot.warmReservedBytes, scenarioName)
        .toBe(0);
      expect(result.expertCache?.replay.snapshot, scenarioName)
        .toEqual(result.expertCache?.snapshot);
      expect(result.physical?.replay.completedAtNs, scenarioName)
        .toBe(result.physical?.execution.completedAtNs);
    }
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
