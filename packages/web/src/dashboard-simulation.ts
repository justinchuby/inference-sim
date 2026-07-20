import {
  buildScenarioPreset,
  calculateScenarioMemoryLedger,
  simulateExpertCacheWorkload,
  simulateSpeculativeWorkload,
  simulateTopologyWorkload,
  topologyProfileFromExpertCache,
  topologyProfileFromSpeculative,
  type ScenarioPresetName,
  type TopologyWorkloadResult,
} from "@inference-sim/core";
import type { DashboardResult, DashboardRunConfig } from "./types.js";

export function simulateDashboard(
  config: DashboardRunConfig,
): Omit<DashboardResult, "durationMs"> {
  const scenario = buildScenarioPreset(
    config.scenarioName as ScenarioPresetName,
  );
  const scenarioSummary = {
    id: scenario.id,
    family: scenario.family,
    deviceCount: scenario.devices.length,
    linkCount: scenario.links.length,
    memoryLedger: calculateScenarioMemoryLedger(scenario),
  };
  if (config.mode === "speculative") {
    const workload = runSpeculative(config);
    return {
      scenario: scenarioSummary,
      mode: config.mode,
      topology: summarizeTopology(simulateTopologyWorkload(
        scenario,
        topologyProfileFromSpeculative(workload.result),
      )),
      speculative: workload.dashboard,
    };
  }
  const workload = runExpertCache(config);
  return {
    scenario: scenarioSummary,
    mode: config.mode,
    topology: summarizeTopology(simulateTopologyWorkload(
      scenario,
      topologyProfileFromExpertCache(workload.result, workload.expertBytes),
    )),
    expertCache: workload.dashboard,
  };
}

function runSpeculative(
  config: DashboardRunConfig,
) {
  const initialTokenLength = 2048;
  const outputTokens = clampInteger(
    config.speculative.outputTokens,
    1,
    512,
  );
  const draftWidth = clampInteger(config.speculative.draftWidth, 1, 8);
  const capacityTokens = initialTokenLength + outputTokens + draftWidth;
  const first = clamp(config.speculative.firstPositionAcceptance, 0.05, 0.99);
  const result = simulateSpeculativeWorkload({
    family: "mtp",
    initialTokenLength,
    outputTokenCount: outputTokens,
    maxAdditionalTokens: draftWidth,
    acceptance: {
      kind: "conditional_heuristic",
      matchProbabilityByPosition: Array.from(
        { length: draftWidth },
        (_, index) => Math.max(0.05, first * 0.86 ** index),
      ),
      seed: clampInteger(config.seed, 0, 0xffff_ffff),
    },
    stateGroups: [
      {
        id: "target-kv",
        owner: "target",
        capacityTokens,
        rollbackProtection: { kind: "non_destructive_tail" },
      },
      {
        id: "mtp-state",
        owner: "proposer",
        capacityTokens,
        rollbackProtection: {
          kind: "bounded_snapshot",
          maxRollbackTokens: draftWidth,
        },
      },
    ],
    pagedKv: {
      pageSizeTokens: 16,
      bytesPerToken: 64 * 1024,
      capacityBytes: 256 * 1024 * 1024,
    },
  });
  return {
    result,
    dashboard: {
      metrics: result.metrics,
      iterations: result.iterations,
      finalTokenLength: result.finalTokenLength,
    },
  };
}

function runExpertCache(
  config: DashboardRunConfig,
) {
  const expertCount = clampInteger(config.expertCache.expertCount, 4, 64);
  const topK = clampInteger(config.expertCache.topK, 1, expertCount);
  const hotSlots = clampInteger(
    config.expertCache.hotSlots,
    topK,
    expertCount,
  );
  const warmSlots = clampInteger(
    config.expertCache.warmSlots,
    0,
    expertCount,
  );
  const expertBytes = 64 * 1024 * 1024;
  const experts = Array.from({ length: expertCount }, (_, index) => ({
    id: `expert-${index}`,
    bytes: expertBytes,
    routingWeight: Math.max(0.2, 1.5 - index / expertCount),
  }));
  const result = simulateExpertCacheWorkload({
    cache: {
      experts,
      hotCapacityBytes: hotSlots * expertBytes,
      warmCapacityBytes: warmSlots * expertBytes,
      warmToHotLatencyNs: 400_000,
      coldToHotLatencyNs: 2_200_000,
      coldToWarmLatencyNs: 1_500_000,
      routingSeed: clampInteger(config.seed, 0, 0xffff_ffff),
      initialHotExpertIds: experts.slice(0, hotSlots).map((expert) => expert.id),
      initialWarmExpertIds: experts
        .slice(hotSlots, hotSlots + warmSlots)
        .map((expert) => expert.id),
    },
    tokenCount: clampInteger(config.expertCache.tokenCount, 1, 512),
    topK,
    tokenIntervalNs: 250_000,
  });
  return {
    result,
    expertBytes,
    dashboard: {
      metrics: result.snapshot.metrics,
      routes: result.routes,
      hotResidentBytes: result.snapshot.hotResidentBytes,
      warmResidentBytes: result.snapshot.warmResidentBytes,
      hotCapacityBytes: result.snapshot.hotCapacityBytes,
      warmCapacityBytes: result.snapshot.warmCapacityBytes,
    },
  };
}

function summarizeTopology(
  result: TopologyWorkloadResult,
): DashboardResult["topology"] {
  const operationCounts = {
    compute: 0,
    transfer: 0,
    collective: 0,
  };
  for (const event of result.execution.trace.operations) {
    operationCounts[event.kind]++;
  }
  return {
    confidence: result.confidence,
    assumptions: result.assumptions,
    planSteps: result.plan.steps.length,
    operationCounts,
    metrics: result.metrics,
    topResources: [
      ...result.metrics.computeUtilization,
      ...result.metrics.linkUtilization,
    ]
      .sort((left, right) => (
        right.utilization - left.utilization
        || left.resourceId.localeCompare(right.resourceId)
      ))
      .slice(0, 8),
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function clampInteger(value: number, minimum: number, maximum: number): number {
  return Math.round(clamp(value, minimum, maximum));
}
