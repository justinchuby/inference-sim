import {
  buildScenarioPreset,
  buildSpeculativeStateGroups,
  calculateScenarioMemoryLedger,
  defaultSpeculativeEligibility,
  simulateExpertCacheWorkload,
  simulateSpeculativeWorkload,
  simulateTopologyServingWorkload,
  simulateTopologyWorkload,
  speculativeFamilyContract,
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
  if (config.mode === "serving") {
    const serving = runServing(config, scenario);
    return {
      scenario: scenarioSummary,
      mode: config.mode,
      topology: summarizeServingTopology(serving),
      serving: {
        decodeMode: config.serving.decodeMode,
        support: config.serving.decodeMode === "target_only"
          ? "target_only"
          : speculativeFamilyContract(config.serving.decodeMode).support,
        metrics: serving.serving.metrics,
        requests: serving.serving.requests,
        batches: serving.batches.map((batch) => ({
          batchId: batch.batchId,
          sequenceCount: batch.work.sequenceCount,
          tokenWork: batch.work.tokenWork,
          prefillSequences: batch.work.prefill.length,
          decodeSequences: batch.work.decode.length,
          durationNs: batch.topology.metrics.totalDurationNs,
        })),
      },
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

function runServing(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
) {
  const requestCount = clampInteger(config.serving.requestCount, 1, 32);
  const promptTokens = clampInteger(config.serving.promptTokens, 16, 4096);
  const outputTokens = clampInteger(config.serving.outputTokens, 1, 512);
  const peakPerRequest = promptTokens + outputTokens - 1;
  const draftWidth = clampInteger(config.serving.draftWidth, 1, 8);
  const first = clamp(
    config.serving.firstPositionAcceptance,
    0.05,
    0.99,
  );
  return simulateTopologyServingWorkload(
    scenario,
    {
      requests: Array.from({ length: requestCount }, (_, index) => ({
        id: `request-${index}`,
        arrivalNs: index * clampInteger(
          config.serving.arrivalGapUs,
          0,
          10_000,
        ) * 1_000,
        promptTokens,
        outputTokens,
      })),
      maxBatchSize: clampInteger(config.serving.maxBatchSize, 1, 16),
      maxBatchTokens: clampInteger(
        config.serving.maxBatchTokens,
        8,
        512,
      ),
      prefillChunkTokens: clampInteger(
        config.serving.prefillChunkTokens,
        8,
        512,
      ),
      maxKvTokens: requestCount * peakPerRequest,
      ...(config.serving.decodeMode === "target_only"
        ? {}
        : {
            speculative: {
              family: config.serving.decodeMode,
              eligibility: defaultSpeculativeEligibility(
                config.serving.decodeMode,
              ),
              maxAdditionalTokens: draftWidth,
              acceptance: {
                kind: "conditional_heuristic" as const,
                matchProbabilityByPosition: Array.from(
                  { length: draftWidth },
                  (_, index) => Math.max(0.05, first * 0.86 ** index),
                ),
                seed: clampInteger(config.seed, 0, 0xffff_ffff),
              },
            },
          }),
    },
  );
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
    family: config.speculative.family,
    eligibility: defaultSpeculativeEligibility(config.speculative.family),
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
    stateGroups: buildSpeculativeStateGroups(
      config.speculative.family,
      capacityTokens,
      draftWidth,
    ),
    pagedKv: {
      pageSizeTokens: 16,
      bytesPerToken: 64 * 1024,
      capacityBytes: 256 * 1024 * 1024,
    },
  });
  return {
    result,
    dashboard: {
      family: result.family,
      support: result.familyContract.support,
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

function summarizeServingTopology(
  result: ReturnType<typeof simulateTopologyServingWorkload>,
): DashboardResult["topology"] {
  const computeUtilization = result.metrics.resourceUtilization.filter(
    (resource) => resource.resourceId.startsWith("compute:"),
  );
  const linkUtilization = result.metrics.resourceUtilization.filter(
    (resource) => resource.resourceId.startsWith("link:"),
  );
  return {
    confidence: result.confidence,
    assumptions: result.assumptions,
    planSteps: result.metrics.planSteps,
    operationCounts: {
      compute: result.metrics.computeOperations,
      transfer: result.metrics.transferOperations,
      collective: result.metrics.collectiveOperations,
    },
    metrics: {
      totalDurationNs: result.metrics.totalDurationNs,
      committedTokens: result.serving.metrics.outputTokens,
      tokensPerSecond: result.serving.metrics.throughputTokensPerSecond,
      computeServiceNs: result.metrics.computeServiceNs,
      transferServiceNs: result.metrics.transferServiceNs,
      collectiveServiceNs: result.metrics.collectiveServiceNs,
      computeUtilization,
      linkUtilization,
    },
    topResources: result.metrics.resourceUtilization.slice(0, 8),
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function clampInteger(value: number, minimum: number, maximum: number): number {
  return Math.round(clamp(value, minimum, maximum));
}
