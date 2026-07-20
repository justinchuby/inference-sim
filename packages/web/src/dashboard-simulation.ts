import {
  DEFAULT_TOPOLOGY_COST_MODEL,
  SCENARIO_PRESET_NAMES,
  SERVING_EXPERT_CACHE_CONTRACT_REVISION,
  buildMultiGpuRingScenario,
  buildScenarioPreset,
  buildSpeculativeStateGroups,
  calculateScenarioMemoryLedger,
  compareTopologyServingWorkloads,
  defaultSpeculativeEligibility,
  expertCacheConfigForTopology,
  fitTopologyCostModel,
  parseSimulationScenario,
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
import type {
  DashboardArtifactOutput,
  DashboardResult,
  DashboardRunConfig,
  WorkerRunProgressReporter,
} from "./types.js";

export function simulateDashboard(
  config: DashboardRunConfig,
): Omit<DashboardResult, "durationMs"> {
  return simulateDashboardExecution(config).summary;
}

export function simulateDashboardExecution(
  config: DashboardRunConfig,
  reportProgress: WorkerRunProgressReporter = () => {},
): DashboardArtifactOutput {
  reportProgress({ progress: 10, phase: "Validating dashboard input" });
  validateModelCapabilityBinding(config);
  if (config.calibration !== undefined) {
    reportProgress({ progress: 18, phase: "Fitting calibration evidence" });
  }
  const calibration = config.calibration === undefined
    ? undefined
    : fitTopologyCostModel(config.calibration);
  const costModel = calibration?.costModel ?? DEFAULT_TOPOLOGY_COST_MODEL;
  const configuredScenario = buildSelectedScenario(config);
  validateModelCapacity(config, configuredScenario);
  validateResourceManager(config, configuredScenario);
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
    reportProgress({ progress: 30, phase: "Building comparison workloads" });
    const comparison = compareTopologyServingWorkloads(
      SCENARIO_PRESET_NAMES.map((name) => {
        const scenario = buildScenarioPreset(name);
        validateModelCapacity(config, scenario);
        validateResourceManager(config, scenario);
        return scenario;
      }),
      buildServingConfig(config),
      costModel,
      buildServingExpertCacheConfig(config),
      config.modelBinding?.executionProfile,
    );
    reportProgress({ progress: 74, phase: "Ranking topology replays" });
    const fastest = comparison.runs[0];
    if (!fastest) {
      throw new Error("serving comparison produced no topology runs");
    }
    const scenario = buildScenarioPreset(
      fastest.result.scenarioId as ScenarioPresetName,
    );
    reportProgress({ progress: 80, phase: "Summarizing serving comparison" });
    return {
      summary: attachCalibration(servingDashboardResult(
        config,
        scenario,
        fastest.result,
        comparison,
      )),
      evidence: {
        kind: "serving_comparison",
        comparison,
      },
    };
  }
  reportProgress({ progress: 26, phase: "Building selected scenario" });
  const scenario = configuredScenario;
  const scenarioSummary = summarizeScenario(scenario, config);
  if (config.mode === "speculative") {
    reportProgress({
      progress: 38,
      phase: config.speculative.trace
        ? "Verifying speculative token trace"
        : "Simulating speculative iterations",
    });
    const workload = runSpeculative(config);
    reportProgress({ progress: 62, phase: "Replaying topology workload" });
    const topology = simulateTopologyWorkload(
      scenario,
      topologyProfileFromSpeculative(
        workload.result,
        config.modelBinding?.executionProfile,
      ),
      costModel,
    );
    reportProgress({ progress: 78, phase: "Summarizing speculative evidence" });
    return {
      summary: attachCalibration({
        scenario: scenarioSummary,
        ...(modelSummary(config) === undefined
          ? {}
          : { model: modelSummary(config)! }),
        mode: config.mode,
        topology: summarizeTopology(topology),
        speculative: workload.dashboard,
      }),
      evidence: {
        kind: "speculative",
        workload: workload.result,
        topology,
      },
    };
  }
  if (config.mode === "serving") {
    reportProgress({ progress: 38, phase: "Simulating continuous batches" });
    const serving = runServing(config, scenario, costModel);
    reportProgress({ progress: 78, phase: "Summarizing serving evidence" });
    return {
      summary: attachCalibration(servingDashboardResult(
        config,
        scenario,
        serving,
      )),
      evidence: {
        kind: "serving",
        serving,
      },
    };
  }
  reportProgress({ progress: 38, phase: "Simulating expert cache routes" });
  const workload = runExpertCache(config, scenario);
  reportProgress({ progress: 62, phase: "Replaying topology workload" });
  const topology = simulateTopologyWorkload(
    scenario,
    topologyProfileFromExpertCache(
      workload.result,
      config.expertCache.placementStrategy,
    ),
    costModel,
  );
  reportProgress({ progress: 78, phase: "Summarizing expert cache evidence" });
  return {
    summary: attachCalibration({
      scenario: scenarioSummary,
      mode: config.mode,
      topology: summarizeTopology(topology),
      expertCache: workload.dashboard,
    }),
    evidence: {
      kind: "expert_cache",
      workload: workload.result,
      topology,
    },
  };
}

function validateModelCapacity(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
): void {
  const binding = config.modelBinding;
  if (binding === undefined || config.mode === "expert-cache") {
    return;
  }
  const targetDomains = new Set(scenario.placements
    .filter((placement) => (
      placement.requiredCapabilities.includes("attention")
      || placement.requiredCapabilities.includes("ffn")
    ))
    .flatMap((placement) => placement.allocations
      .filter((allocation) => allocation.purpose === "weights")
      .map((allocation) => allocation.domainId)));
  const capacityBytes = scenario.memoryDomains
    .filter((domain) => targetDomains.has(domain.id))
    .reduce((sum, domain) => sum + domain.resourceLimitBytes, 0);
  const seenAllocations = new Set<string>();
  const allocationBytes = allocationBytesForDashboard(config, scenario);
  const reservedNonWeightBytes = scenario.placements
    .flatMap((placement) => placement.allocations)
    .filter((allocation) => (
      targetDomains.has(allocation.domainId)
      && allocation.purpose !== "weights"
      && (
        allocation.purpose !== "cache"
        || isExpertCacheEnabled(config)
      )
      && !seenAllocations.has(allocation.physicalAllocationId)
      && seenAllocations.add(allocation.physicalAllocationId)
    ))
    .reduce(
      (sum, allocation) => (
        sum
        + (
          allocationBytes[allocation.physicalAllocationId]
          ?? allocation.bytes
        )
      ),
      0,
    );
  const availableBytes = capacityBytes - reservedNonWeightBytes;
  if (binding.weightBytes > availableBytes) {
    throw new Error(
      `model ${binding.displayName} requires ${formatGiB(binding.weightBytes)} GiB of weights but topology ${scenario.id} has ${formatGiB(availableBytes)} GiB available in target memory domains`,
    );
  }
}

function validateResourceManager(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
): void {
  const expertCacheEnabled = isExpertCacheEnabled(config);
  if (!expertCacheEnabled) {
    assertLedgerWithinResourceLimits(config, scenario);
    return;
  }
  const expertCount = clampInteger(config.expertCache.expertCount, 4, 64);
  const hotSlots = clampInteger(
    config.expertCache.hotSlots,
    config.expertCache.topK,
    expertCount,
  );
  const warmSlots = clampInteger(
    config.expertCache.warmSlots,
    0,
    expertCount,
  );
  const coldExperts = Math.max(0, expertCount - hotSlots - warmSlots);
  if (
    coldExperts > 0
    && !scenario.execution.features.ssdStreaming
  ) {
    throw new Error(
      `expert cache leaves ${coldExperts} experts cold but SSD streaming is disabled`,
    );
  }
  if (coldExperts === 0) {
    assertLedgerWithinResourceLimits(config, scenario);
    return;
  }
  const requiredBackingBytes = expertCount * 64 * 1024 * 1024;
  const availableBackingBytes = scenario.memoryDomains
    .filter((domain) => domain.kind === "storage")
    .reduce((sum, domain) => sum + domain.resourceLimitBytes, 0);
  if (requiredBackingBytes > availableBackingBytes) {
    throw new Error(
      `expert backing requires ${formatGiB(requiredBackingBytes)} GiB but the resource manager allows ${formatGiB(availableBackingBytes)} GiB of SSD`,
    );
  }
  assertLedgerWithinResourceLimits(config, scenario);
}

function assertLedgerWithinResourceLimits(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
): void {
  const overcommitted = calculateScenarioMemoryLedger(scenario, {
    allocationBytes: allocationBytesForDashboard(config, scenario),
  }).find((entry) => entry.freeBytes < 0);
  if (overcommitted !== undefined) {
    throw new Error(
      `resource manager allows ${formatGiB(overcommitted.capacityBytes)} GiB on ${overcommitted.domainId} but active allocations require ${formatGiB(overcommitted.reservedBytes)} GiB`,
    );
  }
}

function isExpertCacheEnabled(config: DashboardRunConfig): boolean {
  return config.mode === "expert-cache"
    || (config.mode === "serving" && config.serving.useExpertCache);
}

function formatGiB(bytes: number): string {
  return (bytes / 1024 ** 3).toFixed(1);
}

function validateModelCapabilityBinding(config: DashboardRunConfig): void {
  const binding = config.modelBinding;
  if (binding === undefined) {
    return;
  }
  const selectedFamily = config.mode === "speculative"
    ? config.speculative.family
    : config.mode === "serving" && config.serving.decodeMode !== "target_only"
      ? config.serving.decodeMode
      : undefined;
  if (
    selectedFamily !== undefined
    && !binding.speculativeFamilies.includes(selectedFamily)
  ) {
    throw new Error(
      `model package does not declare speculative family ${selectedFamily}`,
    );
  }
}

function buildSelectedScenario(config: DashboardRunConfig) {
  if (config.scenarioName === "custom") {
    if (config.customScenario === undefined) {
      throw new Error("dashboard custom scenario is missing");
    }
    return parseSimulationScenario(config.customScenario);
  }
  if (config.customScenario !== undefined) {
    throw new Error(
      "dashboard custom scenario must only be set when scenarioName is custom",
    );
  }
  if (config.scenarioName === "multi-gpu") {
    if (
      config.multiGpuRanks !== 2
      && config.multiGpuRanks !== 4
      && config.multiGpuRanks !== 8
    ) {
      throw new Error(
        `dashboard multi-GPU ranks must be 2, 4, or 8; got ${String(config.multiGpuRanks)}`,
      );
    }
    if (config.multiGpuRanks !== 2) {
      return buildMultiGpuRingScenario(config.multiGpuRanks);
    }
  }
  return buildScenarioPreset(config.scenarioName as ScenarioPresetName);
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
    config.modelBinding?.executionProfile,
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
    scenario: summarizeScenario(scenario, config),
    ...(modelSummary(config) === undefined
      ? {}
      : { model: modelSummary(config)! }),
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
            hotPartitions: serving.expertCache.snapshot.hotPartitions,
            warmPartitions: serving.expertCache.snapshot.warmPartitions,
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

function modelSummary(
  config: DashboardRunConfig,
): DashboardResult["model"] | undefined {
  const binding = config.modelBinding;
  return binding === undefined
    ? undefined
    : {
        name: binding.displayName,
        source: binding.source,
        fingerprint: binding.targetModelFingerprint,
        totalParameters: binding.totalParameters,
        weightBytes: binding.weightBytes,
      };
}

function summarizeScenario(
  scenario: ReturnType<typeof buildScenarioPreset>,
  config: DashboardRunConfig,
): DashboardResult["scenario"] {
  return {
    id: scenario.id,
    family: scenario.family,
    deviceCount: scenario.devices.length,
    linkCount: scenario.links.length,
    memoryLedger: calculateScenarioMemoryLedger(scenario, {
      allocationBytes: allocationBytesForDashboard(config, scenario),
    }),
  };
}

function allocationBytesForDashboard(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
): Readonly<Record<string, number>> {
  const allocations = scenario.placements.flatMap(
    (placement) => placement.allocations,
  );
  const result: Record<string, number> = {};
  const weightAllocations = allocations.filter(
    (allocation) => allocation.purpose === "weights",
  );
  distributeAllocationBytes(
    result,
    weightAllocations,
    config.mode === "expert-cache"
      ? 0
      : config.modelBinding?.weightBytes,
  );

  const cacheAllocations = allocations.filter(
    (allocation) => allocation.purpose === "cache",
  );
  distributeAllocationBytes(result, cacheAllocations, 0);
  const backingAllocations = allocations.filter(
    (allocation) => allocation.purpose === "backing",
  );
  distributeAllocationBytes(result, backingAllocations, 0);
  if (!isExpertCacheEnabled(config)) {
    return result;
  }

  const expert = buildDashboardExpertCache(config, true);
  const placement = {
    strategy: config.expertCache.placementStrategy,
    expertIds: expert.cache.experts.map((candidate) => candidate.id),
  } as const;
  const topologyCache = expertCacheConfigForTopology(
    scenario,
    expert.cache,
    placement,
  );
  distributeAllocationBytes(
    result,
    cacheAllocations.filter((allocation) => (
      allocation.physicalAllocationId.startsWith("expert-hot-cache:")
    )),
    topologyCache.hotCapacityBytes,
  );
  distributeAllocationBytes(
    result,
    cacheAllocations.filter((allocation) => (
      allocation.physicalAllocationId.startsWith("expert-warm-cache:")
    )),
    topologyCache.warmCapacityBytes,
  );
  const coldExperts = Math.max(
    0,
    clampInteger(config.expertCache.expertCount, 4, 64)
      - clampInteger(
        config.expertCache.hotSlots,
        config.expertCache.topK,
        clampInteger(config.expertCache.expertCount, 4, 64),
      )
      - clampInteger(
        config.expertCache.warmSlots,
        0,
        clampInteger(config.expertCache.expertCount, 4, 64),
      ),
  );
  distributeAllocationBytes(
    result,
    backingAllocations,
    coldExperts === 0
      ? 0
      : expert.cache.experts.reduce(
          (sum, candidate) => sum + candidate.bytes,
          0,
        ),
  );
  return result;
}

function distributeAllocationBytes(
  target: Record<string, number>,
  allocations: readonly {
    readonly physicalAllocationId: string;
    readonly bytes: number;
  }[],
  totalBytes: number | undefined,
): void {
  if (totalBytes === undefined || allocations.length === 0) {
    return;
  }
  const ordered = [...allocations].sort((left, right) => (
    left.physicalAllocationId.localeCompare(right.physicalAllocationId)
  ));
  const declaredTotal = ordered.reduce(
    (sum, allocation) => sum + allocation.bytes,
    0,
  );
  let remaining = totalBytes;
  ordered.forEach((allocation, index) => {
    const bytes = index === ordered.length - 1
      ? remaining
      : Math.floor(totalBytes * (allocation.bytes / declaredTotal));
    target[allocation.physicalAllocationId] = bytes;
    remaining -= bytes;
  });
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
  scenario: ReturnType<typeof buildScenarioPreset>,
) {
  const expert = buildDashboardExpertCache(config, true);
  const placement = {
    strategy: config.expertCache.placementStrategy,
    expertIds: expert.cache.experts.map((candidate) => candidate.id),
  } as const;
  const result = simulateExpertCacheWorkload({
    cache: expertCacheConfigForTopology(
      scenario,
      expert.cache,
      placement,
    ),
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
      hotPartitions: result.snapshot.hotPartitions,
      warmPartitions: result.snapshot.warmPartitions,
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
