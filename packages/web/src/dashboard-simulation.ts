import {
  DEFAULT_TOPOLOGY_COST_MODEL,
  SCENARIO_PRESET_NAMES,
  SERVING_EXPERT_CACHE_CONTRACT_REVISION,
  buildScenarioPreset,
  buildSpeculativeStateGroups,
  calculateScenarioMemoryLedger,
  compareTopologyServingWorkloads,
  defaultSpeculativeEligibility,
  fitTopologyCostModel,
  simulateExpertCacheWorkload,
  simulateSpeculativeWorkload,
  simulateSpeculativeTokenTrace,
  simulateTopologyServingWorkload,
  simulateTopologyWorkload,
  speculativeFamilyContract,
  topologyProfileFromExpertCache,
  topologyProfileFromSpeculative,
  type ScenarioPresetName,
  type ServingSchedulerConfig,
  type TopologyWorkloadResult,
  type TopologyCostModel,
  type TopologyServingExpertCacheConfig,
} from "@inference-sim/core";
import type { DashboardResult, DashboardRunConfig } from "./types.js";

export function simulateDashboard(
  config: DashboardRunConfig,
): Omit<DashboardResult, "durationMs"> {
  const calibration = config.calibration === undefined
    ? undefined
    : fitTopologyCostModel(config.calibration);
  const costModel = calibration?.costModel ?? DEFAULT_TOPOLOGY_COST_MODEL;
  const attachCalibration = (
    result: Omit<DashboardResult, "durationMs" | "calibration">,
  ): Omit<DashboardResult, "durationMs"> => ({
    ...result,
    ...(calibration === undefined
      ? {}
      : {
          calibration: {
            datasetId: calibration.datasetId,
            datasetFingerprint: calibration.datasetFingerprint,
            evidenceKind: config.calibration!.provenance.kind,
            fitConfidence: calibration.confidence,
            diagnostics: calibration.diagnostics,
            transportDiagnostics: calibration.transportDiagnostics,
          },
        }),
  });
  if (config.mode === "serving" && config.serving.compareTopologies) {
    const comparison = compareTopologyServingWorkloads(
      SCENARIO_PRESET_NAMES.map(buildScenarioPreset),
      buildServingConfig(config),
      costModel,
      buildServingExpertCacheConfig(config),
    );
    const fastest = comparison.runs[0];
    if (!fastest) {
      throw new Error("serving comparison produced no topology runs");
    }
    const scenario = buildScenarioPreset(
      fastest.result.scenarioId as ScenarioPresetName,
    );
    return attachCalibration(servingDashboardResult(
      config,
      scenario,
      fastest.result,
      comparison,
    ));
  }
  const scenario = buildScenarioPreset(
    config.scenarioName as ScenarioPresetName,
  );
  const scenarioSummary = summarizeScenario(scenario);
  if (config.mode === "speculative") {
    const workload = runSpeculative(config);
    return attachCalibration({
      scenario: scenarioSummary,
      mode: config.mode,
      topology: summarizeTopology(simulateTopologyWorkload(
        scenario,
        topologyProfileFromSpeculative(workload.result),
        costModel,
      )),
      speculative: workload.dashboard,
    });
  }
  if (config.mode === "serving") {
    const serving = runServing(config, scenario, costModel);
    return attachCalibration(servingDashboardResult(config, scenario, serving));
  }
  const workload = runExpertCache(config);
  return attachCalibration({
    scenario: scenarioSummary,
    mode: config.mode,
    topology: summarizeTopology(simulateTopologyWorkload(
      scenario,
      topologyProfileFromExpertCache(
        workload.result,
        config.expertCache.placementStrategy,
      ),
      costModel,
    )),
    expertCache: workload.dashboard,
  });
}

function runServing(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
  costModel: TopologyCostModel,
) {
  return simulateTopologyServingWorkload(
    scenario,
    buildServingConfig(config),
    costModel,
    buildServingExpertCacheConfig(config),
  );
}

function buildServingConfig(
  config: DashboardRunConfig,
): ServingSchedulerConfig {
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
  return {
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
  };
}

function servingDashboardResult(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
  serving: ReturnType<typeof simulateTopologyServingWorkload>,
  comparison?: ReturnType<typeof compareTopologyServingWorkloads>,
): Omit<DashboardResult, "durationMs"> {
  return {
    scenario: summarizeScenario(scenario),
    mode: "serving",
    topology: summarizeServingTopology(serving),
    serving: {
      decodeMode: config.serving.decodeMode,
      support: config.serving.decodeMode === "target_only"
        ? "target_only"
        : speculativeFamilyContract(config.serving.decodeMode).support,
      metrics: serving.serving.metrics,
      requests: serving.serving.requests,
      ...(serving.physical === undefined
        ? {}
        : {
            physicalReplayEvents: serving.physical.replay.appliedEvents,
            maximumConcurrentPlans:
              serving.physical.execution.maximumConcurrentExecutions,
            physicalDrainNs: Math.max(
              0,
              serving.metrics.backgroundDrainNs,
            ),
          }),
      batches: serving.batches.map((batch) => ({
        batchId: batch.batchId,
        sequenceCount: batch.work.sequenceCount,
        tokenWork: batch.work.tokenWork,
        prefillSequences: batch.work.prefill.length,
        decodeSequences: batch.work.decode.length,
        durationNs: batch.durationNs,
        cacheConstraintNs: batch.cacheConstraintNs,
        expertRoutes: batch.expertRoutes.length,
      })),
    },
    ...(serving.expertCache === undefined
      ? {}
      : {
          expertCache: {
            metrics: serving.expertCache.snapshot.metrics,
            routes: serving.expertCache.routes,
            hotResidentBytes: serving.expertCache.snapshot.hotResidentBytes,
            warmResidentBytes: serving.expertCache.snapshot.warmResidentBytes,
            hotCapacityBytes: serving.expertCache.snapshot.hotCapacityBytes,
            warmCapacityBytes: serving.expertCache.snapshot.warmCapacityBytes,
          },
        }),
    ...(comparison
      ? {
          comparison: comparison.runs.map((run) => ({
            rank: run.rank,
            scenarioId: run.result.scenarioId,
            relativeToFastest: run.relativeToFastest,
            totalDurationNs: run.result.metrics.totalDurationNs,
            throughputTokensPerSecond:
              run.result.serving.metrics.throughputTokensPerSecond,
            p95TimeToFirstTokenNs:
              run.result.serving.metrics.p95TimeToFirstTokenNs,
            p95InterTokenLatencyNs:
              run.result.serving.metrics.p95InterTokenLatencyNs,
            averageRequestLatencyNs:
              run.result.serving.metrics.averageRequestLatencyNs,
            kvHighWaterTokens:
              run.result.serving.metrics.kvHighWaterTokens,
            batches: run.result.serving.metrics.batches,
            confidence: run.result.confidence,
          })),
        }
      : {}),
  };
}

function summarizeScenario(
  scenario: ReturnType<typeof buildScenarioPreset>,
): DashboardResult["scenario"] {
  return {
    id: scenario.id,
    family: scenario.family,
    deviceCount: scenario.devices.length,
    linkCount: scenario.links.length,
    memoryLedger: calculateScenarioMemoryLedger(scenario),
  };
}

function runSpeculative(
  config: DashboardRunConfig,
) {
  if (config.speculative.trace) {
    const tokenTrace = simulateSpeculativeTokenTrace(config.speculative.trace);
    const result = tokenTrace.workload;
    return {
      result,
      dashboard: {
        family: result.family,
        support: result.familyContract.support,
        metrics: result.metrics,
        iterations: result.iterations,
        finalTokenLength: result.finalTokenLength,
        tokenTrace: {
          traceId: tokenTrace.traceId,
          source: tokenTrace.provenance.source,
          runtimeRevision: tokenTrace.provenance.runtimeRevision,
          modelFingerprint: tokenTrace.provenance.modelFingerprint,
          targetOnlyRunId: tokenTrace.provenance.targetOnlyRunId,
          speculativeRunId: tokenTrace.provenance.speculativeRunId,
          promptTokenCount: tokenTrace.promptTokenCount,
          comparedTokenCount: tokenTrace.differential.comparedTokenCount,
          matchesTargetOnly: tokenTrace.differential.matchesTargetOnly,
          ...(tokenTrace.differential.firstMismatch
            ? { firstMismatch: tokenTrace.differential.firstMismatch }
            : {}),
          expectedOutputTokenIds: tokenTrace.expectedOutputTokenIds,
          committedOutputTokenIds: tokenTrace.committedOutputTokenIds,
        },
      },
    };
  }
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
  const expert = buildDashboardExpertCache(config, true);
  const result = simulateExpertCacheWorkload({
    cache: expert.cache,
    tokenCount: clampInteger(config.expertCache.tokenCount, 1, 512),
    topK: expert.topK,
    tokenIntervalNs: 250_000,
  });
  return {
    result,
    expertBytes: expert.expertBytes,
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

function buildServingExpertCacheConfig(
  config: DashboardRunConfig,
): TopologyServingExpertCacheConfig | undefined {
  if (!config.serving.useExpertCache) {
    return undefined;
  }
  const expert = buildDashboardExpertCache(config, true);
  return {
    contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
    cache: expert.cache,
    topK: expert.topK,
    placementStrategy: config.expertCache.placementStrategy,
  };
}

function buildDashboardExpertCache(
  config: DashboardRunConfig,
  includeAdaptivePrefetch: boolean,
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
  return {
    expertBytes,
    topK,
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
      ...(includeAdaptivePrefetch
        && config.expertCache.adaptivePrefetch
        && warmSlots > 0
        ? {
            adaptivePrefetch: {
              targetTier: "warm" as const,
              minObservations: 2,
              intervalTokens: 2,
              maxExpertsPerDecision: Math.min(topK, warmSlots),
            },
          }
        : {}),
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
    allReduce: 0,
    allToAll: 0,
  };
  for (const event of result.execution.trace.operations) {
    operationCounts[event.kind]++;
    if (event.collectiveAlgorithm === "all_reduce_ring") {
      operationCounts.allReduce++;
    } else if (event.collectiveAlgorithm === "all_to_all_v") {
      operationCounts.allToAll++;
    }
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
      allReduce: result.metrics.allReduceOperations,
      allToAll: result.metrics.allToAllOperations,
    },
    metrics: {
      totalDurationNs: result.metrics.totalDurationNs,
      foregroundDurationNs: result.batches.reduce(
        (sum, batch) => sum + batch.topology.metrics.foregroundDurationNs,
        0,
      ),
      backgroundDrainNs: result.metrics.backgroundDrainNs,
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
