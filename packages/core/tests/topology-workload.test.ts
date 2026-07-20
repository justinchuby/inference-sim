import { describe, expect, it } from "vitest";
import {
  SCENARIO_PRESET_NAMES,
  buildScenarioPreset,
  compareTopologyWorkloads,
  simulateExpertCacheWorkload,
  simulateSpeculativeWorkload,
  simulateTopologyWorkload,
  targetOnlyTopologyProfile,
  topologyProfileFromExpertCache,
  topologyProfileFromSpeculative,
} from "../src/index.js";

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

  it("turns speculative iterations into draft and multi-token target work", () => {
    const speculative = simulateSpeculativeWorkload({
      family: "mtp",
      initialTokenLength: 16,
      outputTokenCount: 8,
      maxAdditionalTokens: 2,
      acceptance: {
        kind: "replay",
        acceptedDraftTokens: [2, 2, 1],
      },
      stateGroups: [
        {
          id: "target",
          owner: "target",
          capacityTokens: 32,
          rollbackProtection: { kind: "non_destructive_tail" },
        },
      ],
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
