import { describe, expect, it } from "vitest";
import {
  SCENARIO_PRESET_NAMES,
  buildScenarioPreset,
  buildSpeculativeStateGroups,
  compareTopologyWorkloads,
  compileTopologyExpertLoadPlan,
  defaultSpeculativeEligibility,
  executeFrozenPlan,
  replayPlanTrace,
  simulateExpertCacheWorkload,
  simulateSpeculativeWorkload,
  simulateTopologyWorkload,
  targetOnlyTopologyProfile,
  topologyProfileFromExpertCache,
  topologyProfileFromSpeculative,
} from "../src/index.js";

function expertPlacement(
  strategy: "contiguous" | "round_robin" = "contiguous",
  expertIds: readonly string[] = ["e0", "e1", "e2", "e3"],
) {
  return { strategy, expertIds };
}

describe("topology-aware workload execution", () => {
  it("rejects an empty target-only profile before plan execution", () => {
    expect(() => targetOnlyTopologyProfile(0)).toThrow(
      "target-only token count must be a positive safe integer",
    );
  });

  it("executes and independently replays all required topology families", () => {
    const profile = targetOnlyTopologyProfile(8);
    for (const name of SCENARIO_PRESET_NAMES) {
      const scenario = buildScenarioPreset(name);
      const result = simulateTopologyWorkload(scenario, profile);

      expect(result.execution.status).toBe("succeeded");
      expect(result.metrics.committedTokens).toBe(8);
      expect(result.metrics.totalDurationNs).toBeGreaterThan(0);
      expect(result.metrics.tokensPerSecond).toBeGreaterThan(0);
      expect(result.plan.steps.length).toBeGreaterThanOrEqual(8);
    }
  });

  it("compiles expert demand loads into replayable physical plans", () => {
    const localWarmScenarios = new Set(["cpu-only", "unified-memory"]);
    for (const name of SCENARIO_PRESET_NAMES) {
      const scenario = buildScenarioPreset(name);
      const cold = compileTopologyExpertLoadPlan(scenario, {
        id: `cold-${name}`,
        expertId: "e0",
        sourceTier: "cold",
        bytes: 64 * 1024 ** 2,
        placement: expertPlacement(),
      });

      expect(cold.plan).toBeDefined();
      expect(cold.terminalStepIds.length).toBeGreaterThan(0);
      expect(cold.plan?.steps.every((step) => (
        step.operation.kind === "transfer"
      ))).toBe(true);
      expect(cold.plan?.steps.some((step) => (
        step.operation.kind === "transfer"
        && step.operation.linkId.endsWith(":storage-read")
      ))).toBe(true);
      const coldExecution = executeFrozenPlan(scenario, cold.plan!);
      expect(coldExecution.status).toBe("succeeded");
      expect(replayPlanTrace(
        scenario,
        cold.plan!,
        coldExecution.trace,
      ).status).toBe("succeeded");
      expect(cold.terminalStepIds.every((stepId) => (
        coldExecution.trace.operations.some((event) => (
          event.stepId === stepId
          && event.writes.some((allocation) => (
            allocation.startsWith("expert-hot-cache:")
          ))
        ))
      ))).toBe(true);

      const warm = compileTopologyExpertLoadPlan(scenario, {
        id: `warm-${name}`,
        expertId: "e0",
        sourceTier: "warm",
        bytes: 64 * 1024 ** 2,
        placement: expertPlacement(),
      });
      if (localWarmScenarios.has(name)) {
        expect(warm.plan).toBeUndefined();
        expect(warm.terminalStepIds).toEqual([]);
      } else {
        expect(warm.plan).toBeDefined();
        expect(warm.terminalStepIds.length).toBeGreaterThan(0);
        expect(warm.plan?.steps.every((step) => (
          step.operation.kind === "transfer"
          && !step.operation.linkId.endsWith(":storage-read")
        ))).toBe(true);
        const warmExecution = executeFrozenPlan(scenario, warm.plan!);
        expect(warmExecution.status).toBe("succeeded");
        expect(replayPlanTrace(
          scenario,
          warm.plan!,
          warmExecution.trace,
        ).status).toBe("succeeded");
      }
    }
  });

  it("rejects malformed physical expert loads", () => {
    const scenario = buildScenarioPreset("single-gpu-cpu");

    expect(() => compileTopologyExpertLoadPlan(scenario, {
      id: "",
      expertId: "e0",
      sourceTier: "cold",
      bytes: 1,
      placement: expertPlacement(),
    })).toThrow("expert load id must be non-empty");
    expect(() => compileTopologyExpertLoadPlan(scenario, {
      id: "zero",
      expertId: "e0",
      sourceTier: "cold",
      bytes: 0,
      placement: expertPlacement(),
    })).toThrow("expert load zero bytes must be a positive safe integer");
    expect(() => compileTopologyExpertLoadPlan(scenario, {
      id: "invalid-tier",
      expertId: "e0",
      sourceTier: "remote" as "cold",
      bytes: 1,
      placement: expertPlacement(),
    })).toThrow("expert load invalid-tier has invalid source tier remote");
    expect(() => compileTopologyExpertLoadPlan(scenario, {
      id: "unknown-expert",
      expertId: "e9",
      sourceTier: "cold",
      bytes: 1,
      placement: expertPlacement(),
    })).toThrow("expert e9 is absent from the placement universe");
  });

  it("executes speculative and expert profiles on every topology family", () => {
    const profiles = [
      {
        id: "speculative-matrix",
        batchSize: 1,
        units: [{
          id: "verify-0",
          targetTokenWidth: 3,
          committedTokens: 2,
          draftTokens: 2,
          activeExperts: 1,
          warmLoadBytes: 0,
          coldLoadBytes: 0,
        }],
      },
      {
        id: "expert-matrix",
        batchSize: 1,
        units: [{
          id: "route-0",
          targetTokenWidth: 1,
          committedTokens: 1,
          draftTokens: 0,
          activeExperts: 2,
          warmLoadBytes: 64 * 1024 ** 2,
          coldLoadBytes: 0,
        }],
      },
    ];
    for (const name of SCENARIO_PRESET_NAMES) {
      for (const profile of profiles) {
        expect(
          simulateTopologyWorkload(buildScenarioPreset(name), profile)
            .execution.status,
        ).toBe("succeeded");
      }
    }
  });

  it("models pipeline transfers and tensor collectives as link resources", () => {
    const profile = targetOnlyTopologyProfile(2);
    const heterogeneous = simulateTopologyWorkload(
      buildScenarioPreset("gpu-npu"),
      profile,
    );
    const tensor = simulateTopologyWorkload(
      buildScenarioPreset("multi-gpu"),
      profile,
    );

    expect(heterogeneous.execution.trace.operations.some(
      (event) => event.kind === "transfer",
    )).toBe(true);
    expect(heterogeneous.metrics.transferServiceNs).toBeGreaterThan(0);
    expect(tensor.execution.trace.operations.some(
      (event) => event.kind === "collective",
    )).toBe(true);
    expect(tensor.metrics.collectiveServiceNs).toBeGreaterThan(0);
  });

  it("orders TP attention around EP dispatch and gather without double sharding FFN", () => {
    const profile = {
      id: "routed-moe",
      batchSize: 1,
      expertPlacement: expertPlacement(),
      units: [{
        id: "moe-0",
        targetTokenWidth: 3,
        committedTokens: 3,
        draftTokens: 0,
        activeExperts: 2,
        expertRouted: true,
        routedExperts: [
          { expertId: "e0", sourceTier: "hot" as const, loadBytes: 0 },
          { expertId: "e2", sourceTier: "hot" as const, loadBytes: 0 },
          { expertId: "e0", sourceTier: "hot" as const, loadBytes: 0 },
          { expertId: "e2", sourceTier: "hot" as const, loadBytes: 0 },
          { expertId: "e0", sourceTier: "hot" as const, loadBytes: 0 },
          { expertId: "e2", sourceTier: "hot" as const, loadBytes: 0 },
        ],
        warmLoadBytes: 0,
        coldLoadBytes: 0,
      }],
    };
    for (const name of ["multi-gpu", "multi-node"] as const) {
      const result = simulateTopologyWorkload(
        buildScenarioPreset(name),
        profile,
      );
      const collectives = result.plan.steps.filter((step) => (
        step.operation.kind === "collective"
      ));
      expect(collectives.map((step) => (
        step.operation.kind === "collective"
          ? step.operation.algorithm
          : "invalid"
      ))).toEqual([
        "all_reduce_ring",
        "all_to_all_v",
        "all_to_all_v",
      ]);
      expect(collectives.map((step) => (
        step.operation.kind === "collective"
          ? step.operation.commSequenceId
          : -1
      ))).toEqual([0, 1, 2]);
      expect(collectives.every((step) => (
        step.operation.kind === "collective"
        && step.operation.linkIds.length === (name === "multi-node" ? 6 : 2)
      ))).toBe(true);
      const ffn = result.plan.steps.filter((step) => (
        step.operation.kind === "compute"
        && step.operation.capability === "ffn"
      ));
      expect(ffn).toHaveLength(2);
      expect(ffn.every((step) => (
        step.operation.kind === "compute"
        && step.operation.durationNs === 234_000
      ))).toBe(true);
      expect(ffn.every((step) => (
        step.reads.some((allocation) => (
          allocation.startsWith("expert-hot-cache:")
        ))
      ))).toBe(true);
      expect(ffn.every((step) => (
        step.dependencies.includes(collectives[1].id)
      ))).toBe(true);
      expect(collectives[2].dependencies).toEqual(ffn.map((step) => step.id));
      expect(result.execution.status).toBe("succeeded");
    }
  });

  it("executes skewed routed FFN work only on the owning EP rank", () => {
    const result = simulateTopologyWorkload(
      buildScenarioPreset("multi-gpu"),
      {
        id: "skewed-routed-moe",
        batchSize: 1,
        expertPlacement: expertPlacement(),
        units: [{
          id: "skewed-0",
          targetTokenWidth: 3,
          committedTokens: 3,
          draftTokens: 0,
          activeExperts: 1,
          expertRouted: true,
          routedExperts: Array.from({ length: 3 }, () => ({
            expertId: "e0",
            sourceTier: "hot" as const,
            loadBytes: 0,
          })),
          warmLoadBytes: 0,
          coldLoadBytes: 0,
        }],
      },
    );
    const ffn = result.plan.steps.filter((step) => (
      step.operation.kind === "compute"
      && step.operation.capability === "ffn"
    ));
    const gather = result.plan.steps.filter((step) => (
      step.operation.kind === "collective"
      && step.operation.algorithm === "all_to_all_v"
    ))[1];

    expect(ffn).toHaveLength(1);
    expect(ffn[0].participants).toEqual(["rank-0"]);
    expect(ffn[0].reads).toContain("expert-hot-cache:target-shard-0");
    expect(gather.reads).toEqual(["workspace-0"]);
  });

  it("resolves contiguous and round-robin expert owners by communicator rank", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const compile = (strategy: "contiguous" | "round_robin") => (
      compileTopologyExpertLoadPlan(scenario, {
        id: `${strategy}-e1`,
        expertId: "e1",
        sourceTier: "cold",
        bytes: 64 * 1024 ** 2,
        placement: expertPlacement(strategy),
      })
    );
    const targetCache = (strategy: "contiguous" | "round_robin") => (
      compile(strategy).plan?.steps.flatMap((step) => step.writes)
        .find((allocation) => allocation.startsWith("expert-hot-cache:"))
    );

    expect(targetCache("contiguous"))
      .toBe("expert-hot-cache:target-shard-0");
    expect(targetCache("round_robin"))
      .toBe("expert-hot-cache:target-shard-1");
  });

  it("keeps routed warm and cold loads on distinct physical source paths", () => {
    const expertBytes = 64 * 1024 ** 2;
    const result = simulateTopologyWorkload(
      buildScenarioPreset("multi-gpu"),
      {
        id: "mixed-tier-routes",
        batchSize: 1,
        expertPlacement: expertPlacement(),
        units: [{
          id: "mixed-0",
          targetTokenWidth: 1,
          committedTokens: 1,
          draftTokens: 0,
          activeExperts: 2,
          expertRouted: true,
          routedExperts: [
            {
              expertId: "e0",
              sourceTier: "warm",
              loadBytes: expertBytes,
            },
            {
              expertId: "e2",
              sourceTier: "cold",
              loadBytes: expertBytes,
            },
          ],
          warmLoadBytes: expertBytes,
          coldLoadBytes: expertBytes,
        }],
      },
    );
    const transfers = result.plan.steps.filter((step) => (
      step.operation.kind === "transfer"
    ));
    const warm = transfers.find((step) => (
      step.reads.includes("expert-warm-cache:node0")
      && step.writes.includes("expert-hot-cache:target-shard-0")
    ));
    const coldStart = transfers.find((step) => (
      step.reads.includes("expert-backing:node0")
      && step.operation.kind === "transfer"
      && step.operation.linkId === "node0:storage-read"
    ));
    const coldFinish = transfers.find((step) => (
      step.writes.includes("expert-hot-cache:target-shard-1")
    ));

    expect(warm).toBeDefined();
    expect(coldStart).toBeDefined();
    expect(coldFinish).toBeDefined();
    expect(coldFinish?.dependencies).toContain(coldStart?.id);
    expect(warm?.dependencies).not.toContain(coldStart?.id);
  });

  it("rejects incomplete, duplicate, and unplaced routed assignments", () => {
    const base = {
      id: "invalid-routes",
      batchSize: 1,
      expertPlacement: expertPlacement(),
    };
    const unit = {
      id: "route-0",
      targetTokenWidth: 1,
      committedTokens: 1,
      draftTokens: 0,
      activeExperts: 2,
      expertRouted: true,
      warmLoadBytes: 0,
      coldLoadBytes: 0,
    } as const;
    const scenario = buildScenarioPreset("multi-gpu");

    expect(() => simulateTopologyWorkload(scenario, {
      ...base,
      units: [{
        ...unit,
        routedExperts: [
          { expertId: "e0", sourceTier: "hot", loadBytes: 0 },
        ],
      }],
    })).toThrow("has 1/2 expert assignments");
    expect(() => simulateTopologyWorkload(scenario, {
      ...base,
      units: [{
        ...unit,
        routedExperts: [
          { expertId: "e0", sourceTier: "hot", loadBytes: 0 },
          { expertId: "e0", sourceTier: "hot", loadBytes: 0 },
        ],
      }],
    })).toThrow("same expert more than once");
    expect(() => simulateTopologyWorkload(scenario, {
      ...base,
      units: [{
        ...unit,
        routedExperts: [
          { expertId: "e0", sourceTier: "hot", loadBytes: 0 },
          { expertId: "e9", sourceTier: "hot", loadBytes: 0 },
        ],
      }],
    })).toThrow("unknown expert e9");
  });

  it("does not emit expert collectives for dense target-only work", () => {
    const result = simulateTopologyWorkload(
      buildScenarioPreset("multi-gpu"),
      targetOnlyTopologyProfile(2),
    );
    const algorithms = result.plan.steps.flatMap((step) => (
      step.operation.kind === "collective"
        ? [step.operation.algorithm]
        : []
    ));

    expect(algorithms).toEqual(["all_reduce_ring", "all_reduce_ring"]);
  });

  it("rejects ambiguous communicator ownership for the same placements", () => {
    const base = buildScenarioPreset("multi-gpu");
    const scenario = {
      ...base,
      groups: [
        ...base.groups,
        {
          ...base.groups[0],
          id: "duplicate-membership",
        },
      ],
    };

    expect(() => simulateTopologyWorkload(
      scenario,
      targetOnlyTopologyProfile(1),
    )).toThrow(
      "requires exactly one communicator for tensor placements; found 2",
    );
  });

  it("turns speculative iterations into draft and multi-token target work", () => {
    const speculative = simulateSpeculativeWorkload({
      family: "mtp",
      eligibility: defaultSpeculativeEligibility("mtp"),
      initialTokenLength: 16,
      outputTokenCount: 8,
      maxAdditionalTokens: 2,
      acceptance: {
        kind: "replay",
        acceptedDraftTokens: [2, 2, 1],
      },
      stateGroups: buildSpeculativeStateGroups("mtp", 32, 2),
    });
    const result = simulateTopologyWorkload(
      buildScenarioPreset("single-gpu-cpu"),
      topologyProfileFromSpeculative(speculative),
    );

    expect(result.metrics.committedTokens).toBe(8);
    expect(result.execution.trace.operations.some((event) => (
      event.kind === "compute"
      && result.plan.steps[event.stepId].operation.kind === "compute"
      && result.plan.steps[event.stepId].operation.capability === "draft"
    ))).toBe(true);
  });

  it("amortizes fixed forward cost across a multi-token verification", () => {
    const unit = (id: string, targetTokenWidth: number) => ({
      id,
      targetTokenWidth,
      committedTokens: targetTokenWidth,
      draftTokens: 0,
      activeExperts: 1,
      warmLoadBytes: 0,
      coldLoadBytes: 0,
    });
    const scenario = buildScenarioPreset("single-gpu-cpu");
    const wide = simulateTopologyWorkload(scenario, {
      id: "wide-forward",
      batchSize: 1,
      units: [unit("wide", 4)],
    });
    const narrow = simulateTopologyWorkload(scenario, {
      id: "narrow-forwards",
      batchSize: 1,
      units: Array.from({ length: 4 }, (_, index) => unit(`narrow-${index}`, 1)),
    });

    expect(wide.metrics.committedTokens).toBe(narrow.metrics.committedTokens);
    expect(wide.metrics.totalDurationNs).toBeLessThan(
      narrow.metrics.totalDurationNs,
    );
  });

  it("maps expert misses to topology transfers and active FFN cost", () => {
    const expertBytes = 64 * 1024 ** 2;
    const cache = simulateExpertCacheWorkload({
      cache: {
        experts: [
          { id: "e0", bytes: expertBytes },
          { id: "e1", bytes: expertBytes },
          { id: "e2", bytes: expertBytes },
        ],
        hotCapacityBytes: 2 * expertBytes,
        warmCapacityBytes: expertBytes,
        warmToHotLatencyNs: 5,
        coldToHotLatencyNs: 20,
        coldToWarmLatencyNs: 10,
        routingSeed: 4,
        initialHotExpertIds: ["e0"],
        initialWarmExpertIds: ["e1"],
      },
      tokenCount: 4,
      topK: 2,
      tokenIntervalNs: 1,
    });
    const result = simulateTopologyWorkload(
      buildScenarioPreset("single-gpu-cpu"),
      topologyProfileFromExpertCache(cache, expertBytes),
    );

    expect(result.metrics.committedTokens).toBe(4);
    expect(result.metrics.transferServiceNs).toBeGreaterThan(0);
    expect(result.metrics.linkUtilization.length).toBeGreaterThan(0);
  });

  it("projects adaptive warm prefetch onto storage links across all topologies", () => {
    const expertBytes = 64 * 1024 ** 2;
    const cache = simulateExpertCacheWorkload({
      cache: {
        experts: Array.from({ length: 4 }, (_, index) => ({
          id: `e${index}`,
          bytes: expertBytes,
        })),
        hotCapacityBytes: 2 * expertBytes,
        warmCapacityBytes: 2 * expertBytes,
        warmToHotLatencyNs: 5,
        coldToHotLatencyNs: 20,
        coldToWarmLatencyNs: 12,
        routingSeed: 7,
        adaptivePrefetch: {
          targetTier: "warm",
          minObservations: 1,
          intervalTokens: 1,
          maxExpertsPerDecision: 2,
        },
      },
      tokenCount: 4,
      topK: 2,
      tokenIntervalNs: 1,
    });
    const profile = topologyProfileFromExpertCache(cache, expertBytes);
    const hasWarmDemand = profile.units.some((unit) => (
      unit.routedExperts?.some((routed) => routed.sourceTier === "warm")
    ));
    expect(profile.backgroundPrefetches?.length).toBeGreaterThan(0);

    for (const name of SCENARIO_PRESET_NAMES) {
      const result = simulateTopologyWorkload(
        buildScenarioPreset(name),
        profile,
      );
      const storageTransfers = result.execution.trace.operations.filter(
        (event) => (
          event.kind === "transfer"
          && result.plan.steps[event.stepId].operation.kind === "transfer"
          && result.plan.steps[event.stepId].operation.linkId.endsWith(
            ":storage-read",
          )
        ),
      );
      const prefetchStorageTransfers = storageTransfers.filter(
        (event) => event.writes.some(
          (allocation) => allocation.startsWith("expert-warm-cache:"),
        ),
      );
      expect(storageTransfers.length).toBeGreaterThan(0);
      expect(prefetchStorageTransfers.every((event) => event.writes.some(
        (allocation) => allocation.startsWith("expert-warm-cache:"),
      ))).toBe(true);
      const compute = result.execution.trace.operations.filter(
        (event) => event.kind === "compute",
      );
      expect(compute.length).toBeGreaterThan(0);
      const warmConsumers = result.execution.trace.operations.filter(
        (event) => event.reads.some((allocation) => (
          allocation.startsWith("expert-warm-cache:")
        )),
      );
      if (
        !hasWarmDemand
        || name === "cpu-only"
        || name === "unified-memory"
      ) {
        expect(warmConsumers).toHaveLength(0);
      } else {
        expect(warmConsumers.length, name).toBeGreaterThan(0);
      }
      expect(prefetchStorageTransfers.every((producer) => (
        warmConsumers
          .filter((consumer) => producer.writes.some(
            (allocation) => consumer.reads.includes(allocation),
          ))
          .every((consumer) => (
            producer.finishNs <= consumer.startNs
            || consumer.finishNs <= producer.startNs
          ))
      ))).toBe(true);
      const foreground = result.execution.trace.operations.find(
        (event) => event.stepId === result.foregroundTerminalStepId,
      );
      expect(foreground?.finishNs).toBe(
        result.metrics.foregroundDurationNs,
      );
      expect(result.metrics.backgroundDrainNs).toBe(
        result.metrics.totalDurationNs - result.metrics.foregroundDurationNs,
      );
      expect(result.metrics.backgroundDrainNs).toBeGreaterThanOrEqual(0);
      if (name === "multi-node") {
        const storageLinkIds = new Set(prefetchStorageTransfers.map((event) => (
          result.plan.steps[event.stepId].operation.kind === "transfer"
            ? result.plan.steps[event.stepId].operation.linkId
            : ""
        )));
        expect(storageLinkIds).toEqual(new Set([
          "node0:storage-read",
          "node1:storage-read",
        ]));
        expect(prefetchStorageTransfers).toHaveLength(
          profile.backgroundPrefetches?.length,
        );
      }
      expect(result.execution.status).toBe("succeeded");
    }
  });

  it("blocks a warm consumer on its physical prefetch producer", () => {
    const expertBytes = 64 * 1024 ** 2;
    const expertIds = ["e0", "e1", "e2", "e3"];
    const cache = simulateExpertCacheWorkload({
      cache: {
        experts: expertIds.map((id) => ({ id, bytes: expertBytes })),
        hotCapacityBytes: 2 * expertBytes,
        warmCapacityBytes: 4 * expertBytes,
        warmToHotLatencyNs: 5,
        coldToHotLatencyNs: 20,
        coldToWarmLatencyNs: 12,
        routingSeed: 7,
      },
      tokenCount: 1,
      topK: 2,
      tokenIntervalNs: 1,
      initialPrefetch: {
        expertIds,
        targetTier: "warm",
        leadTimeNs: 12,
      },
    });
    const profile = topologyProfileFromExpertCache(cache, expertBytes);
    const required = profile.units[0].requiredPrefetchIds ?? [];
    expect(required).toHaveLength(2);

    const result = simulateTopologyWorkload(
      buildScenarioPreset("single-gpu-cpu"),
      profile,
    );
    const storageSteps = result.plan.steps.filter((step) => (
      step.operation.kind === "transfer"
      && step.operation.linkId === "node0:storage-read"
    ));
    const storageStepByPrefetch = new Map(
      profile.backgroundPrefetches?.map((prefetch, index) => (
        [prefetch.id, storageSteps[index].id] as const
      )),
    );
    const warmDemand = result.plan.steps.find((step) => (
      step.operation.kind === "transfer"
      && step.operation.linkId === "node0:pcie0:forward"
    ));

    expect(warmDemand).toBeDefined();
    expect(required.every((prefetchId) => (
      warmDemand?.dependencies.includes(
        storageStepByPrefetch.get(prefetchId) ?? -1,
      )
    ))).toBe(true);
  });

  it("loads sharded expert bytes into every FFN placement", () => {
    const profile = {
      id: "expert-placement",
      batchSize: 1,
      units: [{
        id: "route-0",
        targetTokenWidth: 1,
        committedTokens: 1,
        draftTokens: 0,
        activeExperts: 2,
        warmLoadBytes: 128 * 1024 ** 2,
        coldLoadBytes: 0,
      }],
    };
    const heterogeneous = simulateTopologyWorkload(
      buildScenarioPreset("gpu-npu"),
      profile,
    );
    const heterogeneousLinks = heterogeneous.execution.trace.operations
      .filter((event) => event.kind === "transfer")
      .map((event) => (
        heterogeneous.plan.steps[event.stepId].operation.kind === "transfer"
          ? heterogeneous.plan.steps[event.stepId].operation.linkId
          : ""
      ));
    expect(heterogeneousLinks.filter(
      (linkId) => linkId === "node0:gpu-pcie:forward",
    )).toHaveLength(2);

    const multiNode = simulateTopologyWorkload(
      buildScenarioPreset("multi-node"),
      profile,
    );
    const multiNodeLinks = multiNode.execution.trace.operations
      .filter((event) => event.kind === "transfer")
      .map((event) => (
        multiNode.plan.steps[event.stepId].operation.kind === "transfer"
          ? multiNode.plan.steps[event.stepId].operation.linkId
          : ""
      ));
    expect(multiNodeLinks).toEqual(expect.arrayContaining([
      "node0:pcie:forward",
      "node1:pcie:forward",
    ]));
  });

  it("compares the same workload deterministically across scenarios", () => {
    const scenarios = SCENARIO_PRESET_NAMES.map(buildScenarioPreset);
    const profile = targetOnlyTopologyProfile(16);
    const first = compareTopologyWorkloads(scenarios, profile);
    const second = compareTopologyWorkloads(scenarios, profile);

    expect(first).toEqual(second);
    expect(first).toHaveLength(6);
    expect(first[0].relativeToFastest).toBe(1);
    expect(new Set(first.map((entry) => entry.durationNs)).size)
      .toBeGreaterThan(2);
    expect(
      first.find((entry) => entry.scenarioId === "cpu-only")?.rank,
    ).toBe(6);
    const duration = (scenarioId: string) => (
      first.find((entry) => entry.scenarioId === scenarioId)?.durationNs
      ?? Number.POSITIVE_INFINITY
    );
    expect(duration("multi-gpu")).toBeLessThan(duration("single-gpu-cpu"));
    expect(duration("multi-node")).toBeGreaterThan(duration("multi-gpu"));
  });
});
