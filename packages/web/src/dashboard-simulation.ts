import { compareIds } from "@inference-sim/core";
import {
  DEFAULT_TOPOLOGY_COST_MODEL,
  SCENARIO_PRESET_NAMES,
  SERVING_EXPERT_CACHE_CONTRACT_REVISION,
  buildMultiGpuRingScenario,
  buildMultiNodeLanScenario,
  buildScenarioPreset,
  buildSpeculativeStateGroups,
  calculateScenarioMemoryLedger,
  compareTopologyServingWorkloads,
  compileTopologyWorkloadPlan,
  defaultSpeculativeEligibility,
  runSeededConcurrentNodeFailureCampaign,
  simulateMultiModelWorkload,
  targetOnlyTopologyProfile,
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
  topologyProfileFromPipeline,
  topologyProfileFromSpeculative,
  buildModelProfile,
  planExpertResidency,
  type ExpertResidencyPlan,
  type ModelProfile,
  type ScenarioPresetName,
  type ServingSchedulerConfig,
  type TopologyWorkloadResult,
  type TopologyCostModel,
  type TopologyServingExpertCacheConfig,
} from "@inference-sim/core";
import type {
  DashboardArtifactOutput,
  DashboardResult,
  DashboardFaultResult,
  DashboardModelBinding,
  DashboardRunConfig,
  WorkerRunProgressReporter,
} from "./types.js";
import type {
  MediaInputProfile,
  MultiModelResult,
  TopologyResourceUtilization,
} from "@inference-sim/core";
import {
  createBuiltinModelBinding,
  modelSupportsSpeculativeFamily,
  type DashboardModelPreset,
} from "./model-binding.js";
import { buildDashboardRoofline } from "./roofline.js";

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
      // Comparison spans topologies, and how much spills differs per topology.
      // The profile is therefore resolved per scenario inside the comparison
      // rather than shared, so a machine that holds every expert is not timed
      // as though it streamed.
      config.modelBinding?.executionProfile,
      config.modelBinding?.pipelineExecution,
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
        costModel,
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
        executionProfileFor(config, scenario),
        config.modelBinding?.pipelineExecution,
      ),
      costModel,
    );
    reportProgress({ progress: 78, phase: "Summarizing speculative evidence" });
    return {
      summary: attachCalibration({
        scenario: scenarioSummary,
        ...(modelSummary(config, scenario) === undefined
          ? {}
          : { model: modelSummary(config, scenario)! }),
        mode: config.mode,
        topology: summarizeTopology(topology),
        roofline: buildDashboardRoofline({
          scenario,
          model: config.modelBinding,
          costModel,
          topology,
          speculative: workload.result,
          mode: config.mode,
        }),
        ...pipelineExecutionSummary([topology]),
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
        costModel,
      )),
      evidence: {
        kind: "serving",
        serving,
      },
    };
  }
  if (config.mode === "pipeline") {
    const pipeline = config.modelBinding?.pipelineExecution;
    if (pipeline === undefined || !pipeline.replacesTarget) {
      throw new Error(
        "pipeline mode requires an imported single-pass, composite, or iterative pipeline",
      );
    }
    reportProgress({ progress: 50, phase: "Compiling component pipeline" });
    const topology = simulateTopologyWorkload(
      scenario,
      topologyProfileFromPipeline(
        pipeline,
        clampInteger(config.serving.requestCount, 1, 32),
      ),
      costModel,
    );
    reportProgress({ progress: 78, phase: "Summarizing pipeline evidence" });
    return {
      summary: attachCalibration({
        scenario: scenarioSummary,
        ...(modelSummary(config, scenario) === undefined
          ? {}
          : { model: modelSummary(config, scenario)! }),
        mode: config.mode,
        topology: summarizeTopology(topology),
        roofline: buildDashboardRoofline({
          scenario,
          model: config.modelBinding,
          costModel,
          topology,
          mode: config.mode,
        }),
        ...pipelineExecutionSummary([topology]),
      }),
      evidence: { kind: "pipeline", topology },
    };
  }
  if (config.mode === "co-residency") {
    reportProgress({ progress: 40, phase: "Placing models on the device" });
    const co = runCoResidency(config, scenario, costModel);
    reportProgress({ progress: 78, phase: "Summarizing residency evidence" });
    return {
      summary: attachCalibration({
        scenario: scenarioSummary,
        ...(modelSummary(config, scenario) === undefined
          ? {}
          : { model: modelSummary(config, scenario)! }),
        mode: config.mode,
        topology: co.topology,
        coResidency: {
          metrics: co.result.metrics,
          loadBandwidthBytesPerSec: co.loadBandwidthBytesPerSec,
        },
      }),
      evidence: { kind: "co_residency", result: co.result },
    };
  }
  if (config.mode === "fault") {
    reportProgress({ progress: 38, phase: "Compiling old-epoch plan" });
    const fault = runNodeFaultCampaign(config, scenario, costModel);
    reportProgress({ progress: 78, phase: "Summarizing fault evidence" });
    return {
      summary: attachCalibration({
        scenario: scenarioSummary,
        ...(modelSummary(config, scenario) === undefined
          ? {}
          : { model: modelSummary(config, scenario)! }),
        mode: config.mode,
        topology: summarizeTopology(fault.topology),
        roofline: buildDashboardRoofline({
          scenario,
          model: config.modelBinding,
          costModel,
          topology: fault.topology,
          mode: config.mode,
        }),
        fault: fault.dashboard,
      }),
      evidence: { kind: "fault", topology: fault.topology },
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
      roofline: buildDashboardRoofline({
        scenario,
        model: config.modelBinding,
        costModel,
        topology,
        mode: config.mode,
      }),
      expertCache: workload.dashboard,
    }),
    evidence: {
      kind: "expert_cache",
      workload: workload.result,
      topology,
    },
  };
}

/**
 * Bytes a model must hold in memory no matter what. Routed experts are
 * excluded because a sparse model can leave them on storage and read them as
 * they are routed to; everything else is touched on every token.
 */
export function residentWeightBytes(config: DashboardRunConfig): number {
  const binding = config.modelBinding;
  if (binding === undefined) {
    return 0;
  }
  const plan = dashboardExpertResidency(config, undefined);
  return plan === undefined
    ? binding.weightBytes
    : binding.weightBytes - plan.streamedExpertBytes;
}

/**
 * How the selected model's routed experts divide between memory and storage on
 * this scenario, or undefined when nothing is offloaded: a dense model, a
 * scenario without SSD streaming, or a model that simply fits.
 *
 * Passing no scenario asks the same question against an unbounded budget,
 * which answers "what could be left behind" rather than "what will be".
 */
export function dashboardExpertResidency(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset> | undefined,
): ExpertResidencyPlan | undefined {
  const binding = config.modelBinding;
  if (binding?.source !== "builtin_model") {
    return undefined;
  }
  const model = buildModelProfile(
    binding.executionProfile.modelId,
    binding.modelFormat?.weightDtypes[0],
    binding.modelFormat?.kvCacheDtype,
  );
  if (scenario === undefined) {
    return planExpertResidency(model, 0);
  }
  if (!scenario.execution.features.ssdStreaming) {
    return undefined;
  }
  const budget = expertMemoryBudgetBytes(config, scenario, model);
  const plan = planExpertResidency(model, budget);
  // Everything fits, so this is an ordinary resident run and must be reported
  // as one rather than as an offload that happens to stream nothing.
  return plan === undefined || plan.streamedExpertBytes === 0
    ? undefined
    : plan;
}

/** Memory left for routed experts after everything that must be resident. */
function expertMemoryBudgetBytes(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
  model: ModelProfile,
): number {
  const capacity = targetDomainCapacity(config, scenario);
  const dense = model.layers.reduce(
    (sum, layer) => sum + layer.attentionBytes + layer.ffnBytes,
    0,
  );
  const shared = model.moe === undefined
    ? 0
    : model.architecture.numLayers * model.moe.sharedExpertBytesPerLayer;
  const alwaysResident = dense + shared + (model.embeddingBytes ?? 0);
  return capacity.availableBytes - alwaysResident;
}

/** Weight bytes that must sit in memory on this scenario. */
function residentWeightBytesFor(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
): number {
  const binding = config.modelBinding;
  if (binding === undefined) {
    return 0;
  }
  const offload = dashboardExpertResidency(config, scenario);
  return offload === undefined
    ? binding.weightBytes
    : binding.weightBytes - offload.streamedExpertBytes;
}

/**
 * The execution profile to time this run with. When routed experts are left on
 * storage, the share of each token's routed reads that misses residency is
 * declared so the cost model charges it against the storage link instead of
 * local memory. Without this the run would report memory-bandwidth speed for
 * bytes that never came from memory.
 */
function executionProfileFor(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
): DashboardModelBinding["executionProfile"] | undefined {
  const profile = config.modelBinding?.executionProfile;
  if (profile === undefined) {
    return undefined;
  }
  const offload = dashboardExpertResidency(config, scenario);
  if (offload === undefined || offload.streamedBytesPerToken <= 0) {
    return profile;
  }
  return {
    ...profile,
    streamedFfnWeightBytesPerToken: Math.min(
      Math.round(offload.streamedBytesPerToken),
      profile.ffnWeightBytesPerToken,
    ),
  };
}

/** Target-domain capacity and what non-weight reservations already claim. */
function targetDomainCapacity(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
): { readonly availableBytes: number } {
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
  // Deliberately not allocationBytesForDashboard: sizing the weight allocation
  // needs this capacity, so asking the full ledger here would be circular.
  // Only non-weight purposes are read, and none of them depend on weights.
  const allocationBytes = nonWeightAllocationBytes(config, scenario);
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
  return { availableBytes: capacityBytes - reservedNonWeightBytes };
}

function validateModelCapacity(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
): void {
  const binding = config.modelBinding;
  if (binding === undefined || config.mode === "expert-cache") {
    return;
  }
  const { availableBytes } = targetDomainCapacity(config, scenario);
  // A sparse model can leave its routed experts on storage and read them as
  // routing reaches them, so what must fit is the tier that every token
  // touches, not the whole checkpoint. A dense model has nothing to leave
  // behind and still has to fit outright.
  const offload = dashboardExpertResidency(config, scenario);
  const requiredBytes = offload === undefined
    ? binding.weightBytes
    : binding.weightBytes - offload.streamedExpertBytes;
  if (requiredBytes > availableBytes) {
    const streamable = dashboardExpertResidency(config, undefined);
    const hint = offload !== undefined
      || streamable === undefined
      || streamable.streamedExpertBytes === 0
        ? ""
        : scenario.execution.features.ssdStreaming
          ? ""
          : `. ${formatGiB(streamable.streamedExpertBytes)} GiB of that is routed experts, which SSD streaming would leave on storage`;
    throw new Error(
      `model ${binding.displayName} requires ${formatGiB(requiredBytes)} GiB of weights but topology ${scenario.id} has ${formatGiB(availableBytes)} GiB available in target memory domains${hint}`,
    );
  }
  if (offload !== undefined) {
    // Streaming only works if the experts left behind actually have somewhere
    // to live, so a scenario without storage fails rather than pretending.
    const storageBytes = scenario.memoryDomains
      .filter((domain) => domain.kind === "storage")
      .reduce((sum, domain) => sum + domain.resourceLimitBytes, 0);
    if (offload.streamedExpertBytes > storageBytes) {
      throw new Error(
        `model ${binding.displayName} streams ${formatGiB(offload.streamedExpertBytes)} GiB of routed experts but topology ${scenario.id} declares only ${formatGiB(storageBytes)} GiB of storage`,
      );
    }
  }
  if (binding.pipelineExecution !== undefined) {
    const bytesByDomain = new Map<string, number>();
    for (const component of binding.pipelineExecution.components) {
      const placement = scenario.placements.find((candidate) => {
        if (
          !candidate.requiredCapabilities.includes("attention")
          && !candidate.requiredCapabilities.includes("ffn")
        ) {
          return false;
        }
        const device = scenario.devices.find(
          (candidateDevice) => candidateDevice.id === candidate.deviceId,
        );
        const preference = component.devicePreference;
        return device !== undefined && (
          preference === undefined
          || preference === "auto"
          || (preference === "cpu" && device.kind === "cpu")
          || (preference === "npu" && device.kind === "npu")
          || (
            preference === "coreml"
            && device.executionProvider.toLowerCase().includes("coreml")
          )
          || (
            !["cpu", "npu", "coreml"].includes(preference)
            && device.kind === "gpu"
            && device.executionProvider.toLowerCase().includes(preference)
          )
        );
      });
      if (placement === undefined) {
        throw new Error(
          `pipeline component ${component.id} cannot satisfy device preference ${component.devicePreference ?? "auto"} on topology ${scenario.id}`,
        );
      }
      const domainId = placement.allocations.find(
        (allocation) => allocation.purpose === "weights",
      )?.domainId;
      if (domainId === undefined) {
        throw new Error(
          `pipeline component ${component.id} placement has no weight memory`,
        );
      }
      bytesByDomain.set(
        domainId,
        (bytesByDomain.get(domainId) ?? 0) + component.weightBytes,
      );
    }
    for (const [domainId, componentBytes] of bytesByDomain) {
      const domain = scenario.memoryDomains.find(
        (candidate) => candidate.id === domainId,
      )!;
      if (componentBytes > domain.resourceLimitBytes) {
        throw new Error(
          `pipeline components placed on ${domainId} require ${formatGiB(componentBytes)} GiB but the resource manager allows ${formatGiB(domain.resourceLimitBytes)} GiB`,
        );
      }
    }
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
    && !modelSupportsSpeculativeFamily(binding, selectedFamily)
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
  if (config.scenarioName === "multi-node") {
    const nodeCount = config.multiNodeCount ?? 2;
    if (nodeCount !== 2 && nodeCount !== 3 && nodeCount !== 4) {
      throw new Error(
        `dashboard multi-node count must be 2, 3, or 4; got ${String(nodeCount)}`,
      );
    }
    return buildMultiNodeLanScenario(nodeCount);
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
    executionProfileFor(config, scenario),
    config.modelBinding?.pipelineExecution,
  );
}

function buildServingConfig(
  config: DashboardRunConfig,
): ServingSchedulerConfig {
  const requestCount = clampInteger(config.serving.requestCount, 1, 32);
  const promptTokens = dashboardPromptTokens(config);
  const outputTokens = clampInteger(
    config.serving.outputTokens,
    1,
    32_768,
  );
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
  costModel: TopologyCostModel,
  comparison?: ReturnType<typeof compareTopologyServingWorkloads>,
): Omit<DashboardResult, "durationMs"> {
  return {
    scenario: summarizeScenario(scenario, config),
    ...(modelSummary(config, scenario) === undefined
      ? {}
      : { model: modelSummary(config, scenario)! }),
    mode: "serving",
    topology: summarizeServingTopology(serving),
    roofline: buildDashboardRoofline({
      scenario,
      model: config.modelBinding,
      costModel,
      serving,
      mode: "serving",
    }),
    ...pipelineExecutionSummary(
      serving.batches.map((batch) => batch.topology),
    ),
    serving: {
      decodeMode: config.serving.decodeMode,
      support: config.serving.decodeMode === "target_only"
        ? "target_only"
        : speculativeFamilyContract(config.serving.decodeMode).support,
      metrics: serving.serving.metrics,
      kvBudgetTokens: dashboardKvBudgetTokens(config),
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
  scenario: ReturnType<typeof buildScenarioPreset>,
): DashboardResult["model"] | undefined {
  const binding = config.modelBinding;
  if (binding === undefined) {
    return undefined;
  }
  const offload = dashboardExpertResidency(config, scenario);
  return {
    name: binding.displayName,
    source: binding.source,
    fingerprint: binding.targetModelFingerprint,
    totalParameters: binding.totalParameters,
    weightBytes: binding.weightBytes,
    ...(binding.modelFormat === undefined
      ? {}
      : { modelFormat: binding.modelFormat }),
    ...(offload === undefined
      ? {}
      : {
          expertOffload: {
            residentWeightBytes:
              binding.weightBytes - offload.streamedExpertBytes,
            streamedExpertBytes: offload.streamedExpertBytes,
            residentExperts: offload.residentExpertsPerLayer,
            totalExperts: offload.totalExpertsPerLayer,
            residentHitFraction: offload.residentHitFraction,
            streamedBytesPerToken: offload.streamedBytesPerToken,
          },
        }),
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
    ssdStreaming: scenario.execution.features.ssdStreaming,
  };
}

/**
 * Every allocation override except weights. Sizing the weight allocation needs
 * the capacity left by these, so they are derived without reference to it.
 */
function nonWeightAllocationBytes(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
): Readonly<Record<string, number>> {
  const allocations = scenario.placements.flatMap(
    (placement) => placement.allocations,
  );
  const result: Record<string, number> = {};

  // KV is a real reservation driven by the workload, not a preset constant.
  // Without this the ledger reports the preset's placeholder extent and the
  // memory breakdown cannot be reconciled with the run.
  distributeAllocationBytes(
    result,
    allocations.filter((allocation) => allocation.purpose === "kv"),
    dashboardKvReservationBytes(config),
  );

  const cacheAllocations = allocations.filter(
    (allocation) => allocation.purpose === "cache",
  );
  distributeAllocationBytes(result, cacheAllocations, 0);
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
  result[EXPERT_CACHE_COLD_BYTES] = coldExperts === 0
    ? 0
    : expert.cache.experts.reduce(
        (sum, candidate) => sum + candidate.bytes,
        0,
      );
  return result;
}

/** Sentinel key carrying the expert-cache cold extent between the two passes. */
const EXPERT_CACHE_COLD_BYTES = "\u0000expert-cache-cold";

function allocationBytesForDashboard(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
): Readonly<Record<string, number>> {
  const allocations = scenario.placements.flatMap(
    (placement) => placement.allocations,
  );
  const { [EXPERT_CACHE_COLD_BYTES]: coldBytes, ...result } = {
    ...nonWeightAllocationBytes(config, scenario),
  };

  // Weights charged to memory are the resident tier only. Routed experts left
  // on storage are charged to the backing allocation instead, so a sparse
  // model that spills is not counted twice.
  distributeAllocationBytes(
    result,
    allocations.filter((allocation) => allocation.purpose === "weights"),
    config.mode === "expert-cache"
      ? 0
      : config.modelBinding === undefined
        ? undefined
        : Math.round(residentWeightBytesFor(config, scenario)),
  );

  const offload = dashboardExpertResidency(config, scenario);
  distributeAllocationBytes(
    result,
    allocations.filter((allocation) => allocation.purpose === "backing"),
    Math.round(
      (offload?.streamedExpertBytes ?? 0) + (coldBytes ?? 0),
    ),
  );
  return result;
}

/**
 * KV bytes the run reserves: the scheduler's whole token budget at the model's
 * per-token KV cost. Modes without a continuous-batching scheduler reserve no
 * KV, so their placeholder allocation collapses to zero.
 */
export function dashboardKvReservationBytes(
  config: DashboardRunConfig,
): number | undefined {
  const bytesPerToken = config.modelBinding?.executionProfile
    .kvCacheBytesPerToken;
  if (bytesPerToken === undefined) {
    return undefined;
  }
  if (config.mode !== "serving") {
    return 0;
  }
  return dashboardKvBudgetTokens(config) * bytesPerToken;
}

/**
 * Decoder tokens each request's media expands into. A vision tower emits real
 * prompt positions, so they cost prefill work and KV exactly like text does.
 * Adapters that cross-attend instead of injecting tokens report zero.
 */
export function dashboardMediaTokens(config: DashboardRunConfig): number {
  const input = dashboardMediaInput(config);
  if (input === undefined) {
    return 0;
  }
  return input.decoderTokensPerItem
    * clampInteger(config.mediaItemsPerRequest, 0, 64);
}

/**
 * The media input this run attaches, or undefined for a text-only run or a
 * model that does not accept the selected modality.
 */
export function dashboardMediaInput(
  config: DashboardRunConfig,
): MediaInputProfile | undefined {
  if (config.modality === "text") {
    return undefined;
  }
  return config.modelBinding?.mediaInputs?.find(
    (input) => input.modality === config.modality,
  );
}

/** Prompt positions a request occupies, media included. */
export function dashboardPromptTokens(config: DashboardRunConfig): number {
  return clampInteger(config.serving.promptTokens, 16, 1_048_576)
    + dashboardMediaTokens(config);
}

/** Token budget the serving scheduler is given for the whole run. */
export function dashboardKvBudgetTokens(config: DashboardRunConfig): number {
  const requestCount = clampInteger(config.serving.requestCount, 1, 32);
  const outputTokens = clampInteger(config.serving.outputTokens, 1, 32_768);
  return requestCount * (dashboardPromptTokens(config) + outputTokens - 1);
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
  const ordered = [...allocations].sort((left, right) => (compareIds(left.physicalAllocationId, right.physicalAllocationId)
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
    32_768,
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

/**
 * Injects a node fault into a small concurrent campaign so the workbench can
 * show what a failed node actually does to work in flight. The plan is the
 * plain target-only topology plan; the interesting behaviour is the fault
 * semantics, not the workload shape.
 */
function runNodeFaultCampaign(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
  costModel: TopologyCostModel,
): {
  readonly dashboard: DashboardFaultResult;
  readonly topology: TopologyWorkloadResult;
} {
  const outputTokens = clampInteger(config.serving.outputTokens, 1, 64);
  const profile = targetOnlyTopologyProfile(outputTokens);
  const plan = compileTopologyWorkloadPlan(scenario, profile, costModel);
  const rankDevices = new Map(scenario.groups.flatMap((group) => (
    group.orderedRanks.map((rank) => [rank.rankId, rank.deviceId] as const)
  )));
  const deviceNodes = new Map(
    scenario.devices.map((device) => [device.id, device.nodeId] as const),
  );
  const nodeOf = (rankId: string) => (
    deviceNodes.get(rankDevices.get(rankId) ?? "") ?? ""
  );
  const planNodes = [
    ...new Set(planRankIds(plan).map(nodeOf).filter((node) => node !== "")),
  ].sort(compareIds);
  if (planNodes.length === 0) {
    throw new Error(
      `topology ${scenario.id} has no plan ranks to fail`,
    );
  }
  const failedNodeId = planNodes.includes(config.fault.failedNodeId)
    ? config.fault.failedNodeId
    : planNodes[0]!;
  const faultAtNs = clampInteger(config.fault.faultAtUs, 1, 100_000) * 1_000;
  const quiesceTimeoutNs =
    clampInteger(config.fault.quiesceTimeoutUs, 1, 1_000_000) * 1_000;
  const executionCount = clampInteger(config.fault.executionCount, 1, 32);
  const campaign = runSeededConcurrentNodeFailureCampaign(
    scenario,
    plan,
    { executionCount, seed: clampInteger(config.seed, 0, 0xffff_ffff), arrivalWindowNs: 0 },
    {
      kind: "node_failure",
      atNs: faultAtNs,
      nodeId: failedNodeId,
      reason: `${failedNodeId} heartbeat expired`,
      quiesceTimeoutNs,
    },
  );
  const retainedOperations = campaign.execution.trace.operations.length;
  const plannedOperations = plan.steps.length * executionCount;
  // What quiescence would have been if work that can never complete had been
  // allowed to run to its planned finish.
  const drainedAtNs = campaign.execution.trace.operations.reduce(
    (maximum, { event }) => Math.max(maximum, event.finishNs),
    faultAtNs,
  );
  const rankStates = campaign.execution.trace.terminals
    .flatMap((terminal) => terminal.rankStates)
    .map((state) => ({
      rankId: state.rankId,
      deviceId: rankDevices.get(state.rankId) ?? "",
      nodeId: nodeOf(state.rankId),
      status: state.status,
      terminalAtNs: state.terminalAtNs,
      onFailedNode: nodeOf(state.rankId) === failedNodeId,
    }))
    .filter((state, index, all) => (
      all.findIndex((other) => other.rankId === state.rankId) === index
    ))
    .sort((left, right) => compareIds(left.rankId, right.rankId));
  return {
    dashboard: {
      failedNodeId,
      faultAtNs,
      quiesceTimeoutNs,
      abortDeadlineNs: faultAtNs + quiesceTimeoutNs,
      quiescedAtNs: campaign.execution.completedAtNs,
      drainedAtNs,
      executionCount,
      plannedOperations,
      retainedOperations,
      droppedOperations: Math.max(0, plannedOperations - retainedOperations),
      replayAppliedEvents: campaign.replay.appliedEvents,
      rankStates,
    },
    topology: simulateTopologyWorkload(scenario, profile, costModel),
  };
}

function planRankIds(plan: { readonly steps: readonly { readonly participants: readonly string[] }[] }): string[] {
  return [...new Set(plan.steps.flatMap((step) => step.participants))];
}

/**
 * Builds one tenant per selected preset and serves their interleaved request
 * streams from the scenario's largest compute-visible memory domain.
 */
/** Heuristic prefill rate; calibration replaces it for the serving workloads. */
const PREFILL_NS_PER_TOKEN = 300_000;

/**
 * The domain a co-resident model's weights live in. A discrete GPU holds them
 * in its own VRAM even though it can also reach host memory, so device-local
 * memory wins over the larger host domain rather than the widest one winning.
 */
export function coResidencyMemoryDomain(
  scenario: ReturnType<typeof buildScenarioPreset>,
): { readonly deviceId: string; readonly domain: ReturnType<typeof buildScenarioPreset>["memoryDomains"][number] } {
  const device = scenario.devices.find((candidate) => (
    candidate.capabilities.includes("ffn")
  )) ?? scenario.devices[0]!;
  const reachable = scenario.memoryDomains.filter((domain) => (
    device.memoryDomainIds.includes(domain.id) && domain.kind !== "storage"
  ));
  const byKind = (kind: string) => reachable
    .filter((domain) => domain.kind === kind)
    .reduce<typeof reachable[number] | undefined>(
      (widest, domain) => (
        widest === undefined
        || domain.resourceLimitBytes > widest.resourceLimitBytes
          ? domain
          : widest
      ),
      undefined,
    );
  const domain = byKind("device")
    ?? byKind("unified")
    ?? byKind("host")
    ?? reachable[0]
    ?? scenario.memoryDomains[0]!;
  return { deviceId: device.id, domain };
}

function runCoResidency(
  config: DashboardRunConfig,
  scenario: ReturnType<typeof buildScenarioPreset>,
  costModel: TopologyCostModel,
): {
  readonly result: MultiModelResult;
  readonly loadBandwidthBytesPerSec: number;
  readonly topology: DashboardResult["topology"];
} {
  const placement = coResidencyMemoryDomain(scenario);
  const target = placement.domain;
  // Weights arrive from wherever the device is not: a host or storage domain.
  const source = scenario.memoryDomains
    .filter((domain) => domain.id !== target.id)
    .reduce<{ bandwidthBytesPerSec: number } | undefined>(
      (best, domain) => (
        best === undefined || domain.bandwidthBytesPerSec > best.bandwidthBytesPerSec
          ? domain
          : best
      ),
      undefined,
    );
  const link = scenario.links.find((candidate) => (
    candidate.targetDomainId === target.id
  ));
  const loadBandwidthBytesPerSec = Math.max(
    1,
    Math.min(
      link?.bandwidthBytesPerSec ?? Number.POSITIVE_INFINITY,
      source?.bandwidthBytesPerSec ?? Number.POSITIVE_INFINITY,
      target.bandwidthBytesPerSec,
    ),
  );
  const gapNs = clampInteger(config.coResidency.requestGapMs, 1, 600_000) * 1e6;
  const promptTokens = clampInteger(
    config.coResidency.promptTokens,
    16,
    1_048_576,
  );
  const outputTokens = clampInteger(config.coResidency.outputTokens, 1, 32_768);
  const tenants = config.coResidency.models.map((entry, index) => {
    const binding = createBuiltinModelBinding(
      entry.preset as DashboardModelPreset,
      entry.weightDtype,
      "fp16",
    );
    const kvBytesPerToken = Math.round(
      binding.executionProfile.kvCacheBytesPerToken ?? 0,
    );
    const contextTokens = clampInteger(entry.contextTokens, 64, 1_048_576);
    const requestCount = clampInteger(entry.requestCount, 1, 16);
    // Streams are offset from each other so the models genuinely interleave
    // rather than each running to completion in turn.
    const offsetNs = Math.round(gapNs * index / config.coResidency.models.length);
    return {
      id: entry.preset,
      displayName: binding.displayName,
      weightBytes: Math.max(1, Math.round(binding.weightBytes)),
      kvBytesPerToken,
      maxKvTokens: Math.max(contextTokens, promptTokens + outputTokens - 1),
      pinned: entry.pinned,
      requests: Array.from({ length: requestCount }, (_, request) => ({
        id: `${entry.preset}-r${request}`,
        arrivalNs: Math.round(offsetNs + request * gapNs),
        promptTokens,
        outputTokens,
      })),
    };
  });
  const memoryBandwidth = target.bandwidthBytesPerSec;
  const result = simulateMultiModelWorkload(
    {
      tenants,
      deviceMemoryBytes: target.resourceLimitBytes,
      loadBandwidthBytesPerSec,
      maxBatchTokens: clampInteger(config.serving.maxBatchTokens, 8, 512),
      prefillChunkTokens: clampInteger(config.serving.prefillChunkTokens, 8, 512),
    },
    (batch, tenant) => Math.max(1, Math.round(
      // Decode reads the model's weights once per step; prefill is compute
      // bound and charged per token from the topology cost model.
      batch.decodeTokens * (tenant.weightBytes / memoryBandwidth * 1e9)
      + batch.prefillTokens * PREFILL_NS_PER_TOKEN,
    )),
  );
  const committedTokens = result.metrics.tenants.reduce(
    (sum, tenant) => sum + tenant.outputTokens,
    0,
  );
  const busy = (busyNs: number): TopologyResourceUtilization[] => [{
    resourceId: placement.deviceId,
    busyNs,
    capacityLanes: 1,
    utilization: result.metrics.totalDurationNs === 0
      ? 0
      : busyNs / result.metrics.totalDurationNs,
  }];
  return {
    result,
    loadBandwidthBytesPerSec,
    topology: {
      confidence: "heuristic",
      assumptions: [
        "A model occupies its weights plus a preallocated KV arena while resident.",
        "Eviction releases the KV arena, so partial generation is prefilled again.",
        "Decode cost is the model's weight bytes over device bandwidth; prefill is charged per token.",
      ],
      planSteps: result.trace.length,
      operationCounts: {
        compute: result.trace.filter(
          (event) => event.kind === "batch_start",
        ).length,
        transfer: result.metrics.totalLoads,
        collective: 0,
        allReduce: 0,
        allToAll: 0,
      },
      metrics: {
        totalDurationNs: result.metrics.totalDurationNs,
        foregroundDurationNs: result.metrics.totalDurationNs,
        backgroundDrainNs: 0,
        committedTokens,
        tokensPerSecond: result.metrics.totalDurationNs === 0
          ? 0
          : committedTokens / (result.metrics.totalDurationNs / 1e9),
        computeServiceNs: result.metrics.computeServiceNs,
        transferServiceNs: result.metrics.transferServiceNs,
        collectiveServiceNs: 0,
        computeUtilization: busy(result.metrics.computeServiceNs),
        linkUtilization: busy(result.metrics.transferServiceNs),
      },
      topResources: busy(result.metrics.computeServiceNs),
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
        ||compareIds(left.resourceId, right.resourceId)
      ))
      .slice(0, 8),
  };
}

function pipelineExecutionSummary(
  results: readonly TopologyWorkloadResult[],
): Pick<DashboardResult, "pipelineExecution"> {
  const components = new Map<string, {
    readonly id: string;
    readonly phase: string;
    readonly deviceId: string;
  }>();
  let transferOperations = 0;
  const transferCounts = new Map<TopologyWorkloadResult, number>();
  for (const result of results) {
    const cachedTransferCount = transferCounts.get(result);
    if (cachedTransferCount !== undefined) {
      transferOperations += cachedTransferCount;
      continue;
    }
    let resultTransferOperations = 0;
    for (const step of result.plan.steps) {
      if (step.operation.kind === "transfer") {
        resultTransferOperations++;
      } else if (
        step.operation.kind === "compute"
        && step.operation.componentId !== undefined
      ) {
        const component = {
          id: step.operation.componentId,
          phase: step.operation.pipelinePhase ?? "unspecified",
          deviceId: step.operation.deviceId,
        };
        components.set(
          `${component.id}:${component.phase}:${component.deviceId}`,
          component,
        );
      }
    }
    transferCounts.set(result, resultTransferOperations);
    transferOperations += resultTransferOperations;
  }
  return components.size === 0
    ? {}
    : {
        pipelineExecution: {
          components: [...components.values()],
          transferOperations,
        },
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
