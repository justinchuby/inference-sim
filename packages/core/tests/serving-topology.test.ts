import { describe, expect, it } from "vitest";
import {
  SCENARIO_PRESET_NAMES,
  SERVING_EXPERT_CACHE_CONTRACT_REVISION,
  buildScenarioPreset,
  compareTopologyServingWorkloads,
  defaultSpeculativeEligibility,
  expertCacheConfigForTopology,
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
      expect(result.metrics.resourceObservationNs, scenarioName).toBe(
        result.metrics.totalDurationNs,
      );
      expect(result.metrics.backgroundDrainNs, scenarioName).toBe(0);
      expect(result.metrics.resourceUtilization.every((resource) => (
        resource.utilization >= 0 && resource.utilization <= 1
      )), scenarioName).toBe(true);
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

  it("reuses immutable stateless batch topology templates", () => {
    const result = simulateTopologyServingWorkload(
      buildScenarioPreset("multi-gpu"),
      {
        requests: [
          { id: "long", arrivalNs: 0, promptTokens: 8, outputTokens: 32 },
        ],
        maxBatchSize: 1,
        maxBatchTokens: 8,
        prefillChunkTokens: 8,
        maxKvTokens: 40,
      },
    );

    expect(result.metrics.reusedTopologyBatches).toBeGreaterThan(0);
    expect(
      result.metrics.compiledTopologyTemplates
        + result.metrics.reusedTopologyBatches,
    ).toBe(result.batches.length);
    const templateCounts = new Map<object, number>();
    for (const batch of result.batches) {
      templateCounts.set(
        batch.topology,
        (templateCounts.get(batch.topology) ?? 0) + 1,
      );
    }
    expect(Math.max(...templateCounts.values())).toBeGreaterThan(1);
    expect([...templateCounts.keys()].every((topology) => (
      (topology as typeof result.batches[number]["topology"])
        .plan.executionId.includes("serving-template:")
    ))).toBe(true);
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
    expect(result.metrics.reusedTopologyBatches).toBe(0);
    expect(result.metrics.compiledTopologyTemplates).toBe(
      result.batches.length,
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
    const demandLoads = expertCache.trace.filter((event) => (
      event.kind === "load_start" && event.load.kind === "demand"
    ));
    expect(result.physical?.execution.trace.admissions).toHaveLength(
      result.batches.length + demandLoads.length,
    );
    expect(result.physical?.replay.appliedEvents).toBeGreaterThan(0);
    expect(result.physical?.replay.completedAtNs).toBe(
      result.physical?.execution.completedAtNs,
    );
    expect(result.metrics.resourceObservationNs).toBe(Math.max(
      result.metrics.totalDurationNs,
      result.physical?.execution.completedAtNs ?? 0,
    ));
    expect(result.metrics.backgroundDrainNs).toBe(
      result.metrics.resourceObservationNs - result.metrics.totalDurationNs,
    );
    for (const resource of result.metrics.resourceUtilization) {
      const expectedBusyNs =
        result.physical?.execution.trace.operations.reduce(
          (sum, { event }) => (
            event.resources.some((reservation) => (
              reservation.resourceId === resource.resourceId
            ))
              ? sum + event.finishNs - event.startNs
              : sum
          ),
          0,
        ) ?? 0;
      expect(resource.busyNs, resource.resourceId).toBe(expectedBusyNs);
      expect(resource.utilization, resource.resourceId)
        .toBeLessThanOrEqual(1);
    }
    expect(result.batches.every((batch) => (
      batch.physicalExecution?.executionId
        === batch.topology.plan.executionId
      && batch.foregroundCompletedAtNs
        === batch.startedAtNs + batch.durationNs
    ))).toBe(true);
    expect(result.batches.length).toBeGreaterThan(1);
    expect(result.batches[0].expertRoutes[0].sourceTiers).toEqual(["cold"]);
    expect(result.batches.slice(1).every((batch) => (
      batch.expertRoutes.every((route) => (
        route.sourceTiers.every((tier) => tier === "hot")
      ))
    ))).toBe(true);
    expect(result.batches.every((batch) => (
      batch.durationNs >= batch.cacheConstraintNs
      && batch.foregroundCompletedAtNs
        >= batch.startedAtNs + batch.cacheConstraintNs
    ))).toBe(true);
    expect(demandLoads).toHaveLength(1);
    const demandLoad = demandLoads[0].load;
    const demandRetime = expertCache.trace.find((event) => (
      event.kind === "load_retime"
      && event.loadId === demandLoad.loadId
      && event.physicalCompletesAtNs !== undefined
    ));
    const demandExecutionId =
      `multi-gpu:expert-load:${demandLoad.loadId}`;
    const demandOperations =
      result.physical?.execution.trace.operations.filter(({ event }) => (
        event.executionId === demandExecutionId
      )) ?? [];
    expect(demandOperations.length).toBeGreaterThan(0);
    expect(demandOperations.some(({ event }) => (
      event.kind === "transfer"
      && event.resources.some((resource) => (
        resource.resourceId.endsWith(":storage-read")
      ))
    ))).toBe(true);
    expect(demandOperations.some(({ event }) => (
      event.writes.some((allocation) => (
        allocation.startsWith("expert-hot-cache:")
      ))
    ))).toBe(true);
    expect(demandRetime?.physicalCompletesAtNs).toBe(
      Math.max(...demandOperations.map(({ event }) => event.finishNs)),
    );
    expect(result.batches.every((batch) => (
      batch.topology.plan.steps.every((step) => (
        step.operation.kind !== "transfer"
        || !step.operation.linkId.endsWith(":storage-read")
      ))
    ))).toBe(true);
    expect(result.batches.every((batch) => (
      batch.topology.plan.steps.filter((step) => (
        step.operation.kind === "compute"
        && step.operation.capability === "ffn"
      )).every((step) => (
        step.reads.some((allocation) => (
          allocation.startsWith("expert-hot-cache:")
        ))
      ))
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
      const demandLoads = result.expertCache?.trace.filter((event) => (
        event.kind === "load_start" && event.load.kind === "demand"
      )) ?? [];
      expect(demandLoads.length, scenarioName).toBeGreaterThan(0);
      for (const demand of demandLoads) {
        expect(demand.load.sourceTier, scenarioName).toBe("cold");
        const retime = result.expertCache?.trace.find((event) => (
          event.kind === "load_retime"
          && event.loadId === demand.load.loadId
          && event.physicalCompletesAtNs !== undefined
        ));
        const executionId =
          `${result.scenarioId}:expert-load:${demand.load.loadId}`;
        const operations =
          result.physical?.execution.trace.operations.filter(({ event }) => (
            event.executionId === executionId
          )) ?? [];
        expect(operations.length, `${scenarioName}:${demand.load.loadId}`)
          .toBeGreaterThan(0);
        expect(retime?.physicalCompletesAtNs, scenarioName).toBe(
          Math.max(...operations.map(({ event }) => event.finishNs)),
        );
      }
      for (const batch of result.batches) {
        const admission = result.physical?.execution.trace.admissions.find(
          (event) => (
            event.executionId === batch.topology.plan.executionId
          ),
        );
        expect(admission?.arrivalNs, `${scenarioName}:batch-${batch.batchId}`)
          .toBe(batch.startedAtNs + batch.cacheConstraintNs);
      }
      expect(
        result.metrics.allToAllOperations > 0,
        scenarioName,
      ).toBe(scenarioName === "multi-gpu" || scenarioName === "multi-node");
    }
  });

  it("distinguishes local and transported warm demand on every topology", () => {
    const expertBytes = 64 * 1024 ** 2;
    const localWarmScenarios = new Set(["cpu-only", "unified-memory"]);
    for (const scenarioName of SCENARIO_PRESET_NAMES) {
      const result = simulateTopologyServingWorkload(
        buildScenarioPreset(scenarioName),
        {
          requests: [
            { id: "warm", arrivalNs: 0, promptTokens: 1, outputTokens: 1 },
          ],
          maxBatchSize: 1,
          maxBatchTokens: 1,
          prefillChunkTokens: 1,
          maxKvTokens: 1,
        },
        undefined,
        {
          contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
          cache: {
            experts: [{ id: "e0", bytes: expertBytes }],
            hotCapacityBytes: expertBytes,
            warmCapacityBytes: expertBytes,
            warmToHotLatencyNs: 9_999_999,
            coldToHotLatencyNs: 20_000,
            coldToWarmLatencyNs: 10_000,
            routingSeed: 7,
            initialWarmExpertIds: ["e0"],
          },
          topK: 1,
        },
      );
      const demand = result.expertCache?.trace.find((event) => (
        event.kind === "load_start" && event.load.kind === "demand"
      ));
      if (demand?.kind !== "load_start") {
        throw new Error(`missing warm demand for ${scenarioName}`);
      }
      expect(demand.load.sourceTier, scenarioName).toBe("warm");
      const retime = result.expertCache?.trace.find((event) => (
        event.kind === "load_retime"
        && event.loadId === demand.load.loadId
        && event.physicalCompletesAtNs !== undefined
      ));
      expect(retime, scenarioName).toBeDefined();
      expect(
        (retime?.physicalCompletesAtNs ?? 0) - demand.load.startedAtNs,
        scenarioName,
      ).not.toBe(9_999_999);
      const executionId =
        `${scenarioName}:expert-load:${demand.load.loadId}`;
      const operations =
        result.physical?.execution.trace.operations.filter(({ event }) => (
          event.executionId === executionId
        )) ?? [];
      if (localWarmScenarios.has(scenarioName)) {
        expect(operations, scenarioName).toHaveLength(0);
        expect(retime?.physicalCompletesAtNs, scenarioName)
          .toBe(demand.load.startedAtNs);
      } else {
        expect(operations.length, scenarioName).toBeGreaterThan(0);
        expect(operations.every(({ event }) => (
          event.kind === "transfer"
          && !event.resources.some((resource) => (
            resource.resourceId.endsWith(":storage-read")
          ))
        )), scenarioName).toBe(true);
        expect(retime?.physicalCompletesAtNs, scenarioName).toBe(
          Math.max(...operations.map(({ event }) => event.finishNs)),
        );
      }
      expect(result.expertCache?.replay.snapshot, scenarioName)
        .toEqual(result.expertCache?.snapshot);
      expect(result.physical?.replay.completedAtNs, scenarioName)
        .toBe(result.physical?.execution.completedAtNs);
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
    const largeExperts = Array.from({ length: 3 }, (_, index) => ({
      id: `large-${index}`,
      bytes: 3 * 1024 ** 3,
    }));

    expect(() => simulateTopologyServingWorkload(
      scenario,
      workload,
      undefined,
      {
        contractRevision: 99 as 3,
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
    expect(() => simulateTopologyServingWorkload(
      scenario,
      workload,
      undefined,
      {
        contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
        cache,
        topK: 1,
        placementStrategy: "hashed" as "contiguous",
      },
    )).toThrow("invalid placement strategy hashed");
    expect(() => simulateTopologyServingWorkload(
      scenario,
      workload,
      undefined,
      {
        contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
        cache: {
          ...cache,
          experts: largeExperts,
          hotCapacityBytes: 9 * 1024 ** 3,
        },
        topK: 1,
      },
    )).toThrow("exceeds physical hot allocations");
    expect(() => simulateTopologyServingWorkload(
      scenario,
      workload,
      undefined,
      {
        contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
        cache: {
          ...cache,
          experts: largeExperts,
          hotCapacityBytes: 3 * 1024 ** 3,
          warmCapacityBytes: 9 * 1024 ** 3,
        },
        topK: 1,
      },
    )).toThrow("exceeds physical warm allocations");
  });

  it("fails closed when an EP owner is undersized despite aggregate capacity", () => {
    const base = buildScenarioPreset("multi-gpu");
    const scenario = {
      ...base,
      placements: base.placements.map((placement) => (
        placement.partitionId !== "target-shard-0"
          ? placement
          : {
              ...placement,
              allocations: placement.allocations.map((allocation) => (
                allocation.physicalAllocationId
                  !== "expert-hot-cache:target-shard-0"
                  ? allocation
                  : { ...allocation, bytes: 1024 ** 3 }
              )),
            }
      )),
    };

    expect(() => simulateTopologyServingWorkload(
      scenario,
      workload,
      undefined,
      {
        contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
        cache: {
          experts: Array.from({ length: 4 }, (_, index) => ({
            id: `e${index}`,
            bytes: 1024 ** 3,
          })),
          hotCapacityBytes: 2 * 1024 ** 3,
          warmCapacityBytes: 2 * 1024 ** 3,
          warmToHotLatencyNs: 2_000,
          coldToHotLatencyNs: 20_000,
          coldToWarmLatencyNs: 10_000,
          routingSeed: 7,
        },
        topK: 1,
      },
    )).toThrow(
      "hot requirement 2147483648 on owner target-shard-0 exceeds physical hot allocations 1073741824",
    );
  });

  it("derives independent hot-owner and warm-node cache partitions", () => {
    const scenario = buildScenarioPreset("multi-node");
    const expertBytes = 64 * 1024 ** 2;
    const cache = {
      experts: Array.from({ length: 4 }, (_, index) => ({
        id: `e${index}`,
        bytes: expertBytes,
      })),
      hotCapacityBytes: expertBytes,
      warmCapacityBytes: expertBytes,
      warmToHotLatencyNs: 2_000,
      coldToHotLatencyNs: 20_000,
      coldToWarmLatencyNs: 10_000,
      routingSeed: 7,
      initialHotExpertIds: ["e0", "e1"],
    } as const;
    const partitioned = expertCacheConfigForTopology(
      scenario,
      cache,
      {
        strategy: "round_robin",
        expertIds: cache.experts.map((expert) => expert.id),
      },
    );

    expect(partitioned.hotCapacityBytes).toBe(2 * expertBytes);
    expect(partitioned.warmCapacityBytes).toBe(2 * expertBytes);
    expect(partitioned.hotPartitions).toEqual([
      {
        id: "target-shard-0",
        expertIds: ["e0", "e2"],
        capacityBytes: expertBytes,
      },
      {
        id: "target-shard-1",
        expertIds: ["e1", "e3"],
        capacityBytes: expertBytes,
      },
    ]);
    expect(partitioned.warmPartitions).toEqual([
      {
        id: "node0",
        expertIds: ["e0", "e2"],
        capacityBytes: expertBytes,
      },
      {
        id: "node1",
        expertIds: ["e1", "e3"],
        capacityBytes: expertBytes,
      },
    ]);

    const result = simulateTopologyServingWorkload(
      scenario,
      {
        requests: [
          { id: "partitioned", arrivalNs: 0, promptTokens: 1, outputTokens: 1 },
        ],
        maxBatchSize: 1,
        maxBatchTokens: 1,
        prefillChunkTokens: 1,
        maxKvTokens: 1,
      },
      undefined,
      {
        contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
        cache,
        topK: 1,
        placementStrategy: "round_robin",
      },
    );
    expect(result.expertCache?.snapshot.hotPartitions).toEqual([
      {
        id: "target-shard-0",
        capacityBytes: expertBytes,
        residentBytes: expertBytes,
        reservedBytes: 0,
        expertIds: ["e0"],
      },
      {
        id: "target-shard-1",
        capacityBytes: expertBytes,
        residentBytes: expertBytes,
        reservedBytes: 0,
        expertIds: ["e1"],
      },
    ]);
    expect(result.expertCache?.replay.snapshot)
      .toEqual(result.expertCache?.snapshot);
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
    const prefetchLoadIds = new Set(prefetchLoads.map(
      (event) => event.load.loadId,
    ));
    const physicalRetimes = retimes.filter((event) => (
      event.physicalCompletesAtNs !== undefined
      && prefetchLoadIds.has(event.loadId)
    ));
    const storageTransfers = result.physical?.execution.trace.operations.filter(
      ({ event }) => (
        event.kind === "transfer"
        && event.resources.some((resource) => (
          resource.resourceId.endsWith(":storage-read")
        ))
      ),
    ) ?? [];

    expect(prefetchLoads.length).toBeGreaterThan(0);
    expect(retimes.length).toBeGreaterThanOrEqual(prefetchLoads.length * 2);
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
        && prefetchLoads.some((load) => (
          load.load.loadId === event.loadId
        ))
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
      expect(result.metrics.resourceObservationNs, scenarioName).toBe(
        Math.max(
          result.metrics.totalDurationNs,
          result.physical?.execution.completedAtNs ?? 0,
        ),
      );
      expect(result.metrics.resourceUtilization.every((resource) => (
        resource.utilization >= 0 && resource.utilization <= 1
      )), scenarioName).toBe(true);
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
