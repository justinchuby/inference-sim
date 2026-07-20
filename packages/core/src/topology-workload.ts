import {
  PLAN_CONTRACT_REVISION,
  type FrozenPlan,
  type FrozenPlanExecutionResult,
  type CollectiveAlgorithm,
  type PlanStep,
} from "./plan-types.js";
import {
  executeFrozenPlan,
  replayPlanTrace,
} from "./frozen-plan.js";
import {
  findTransferRoute,
} from "./scenario.js";
import type {
  ComputeCapability,
  ConfidenceClass,
  PartitionPlacement,
  SimDeviceKind,
  SimDeviceSpec,
  SimLinkSpec,
  SimulationScenario,
  TransferRoute,
} from "./scenario-types.js";
import type {
  ExpertCacheWorkloadResult,
  ExpertCacheTier,
} from "./expert-cache.js";
import type {
  SpeculativeWorkloadResult,
} from "./speculative-workload.js";
import type {
  SpeculativeProposerExecution,
} from "./speculative-family.js";

export const TOPOLOGY_COST_MODEL_REVISION = 9;
export const TRANSFER_CALIBRATION_ALGORITHM = "point_to_point";
export const COLLECTIVE_CALIBRATION_ALGORITHM = "all_reduce_ring";
export const EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM = "all_to_all_v";
export const ALL_TO_ALL_TRAFFIC_SIGNATURE_PREFIX =
  "all_to_all_v_matrix_v1:";

export interface TopologyWorkUnit {
  readonly id: string;
  readonly targetTokenWidth: number;
  readonly committedTokens: number;
  readonly draftTokens: number;
  readonly proposerExecution?: SpeculativeProposerExecution;
  readonly proposerCostScale?: number;
  readonly activeExperts: number;
  readonly expertRouted?: boolean;
  readonly routedExperts?: readonly TopologyRoutedExpert[];
  readonly warmLoadBytes: number;
  readonly coldLoadBytes: number;
  readonly requiredPrefetchIds?: readonly string[];
}

export interface TopologyWorkloadProfile {
  readonly id: string;
  readonly batchSize: number;
  readonly units: readonly TopologyWorkUnit[];
  readonly modelWork?: TopologyModelWork;
  readonly expertPlacement?: TopologyExpertPlacement;
  readonly expertTokenPlacement?: "round_robin";
  readonly backgroundPrefetches?: readonly TopologyBackgroundPrefetch[];
}

export interface TopologyModelWork {
  readonly modelId: string;
  readonly modelName: string;
  readonly attentionWeightBytesPerToken: number;
  readonly ffnWeightBytesPerToken: number;
  readonly forwardFlopsPerToken: number;
}

export interface TopologyExpertPlacement {
  readonly strategy: "contiguous" | "round_robin";
  readonly expertIds: readonly string[];
}

export interface TopologyRoutedExpert {
  readonly expertId: string;
  readonly sourceTier: ExpertCacheTier;
  readonly loadBytes: number;
}

export interface TopologyBackgroundPrefetch {
  readonly id: string;
  readonly expertId?: string;
  readonly afterUnitIndex: number;
  readonly bytes: number;
}

export interface TopologyExpertLoad {
  readonly id: string;
  readonly expertId: string;
  readonly sourceTier: "warm" | "cold";
  readonly bytes: number;
  readonly placement: TopologyExpertPlacement;
}

export interface TopologyExpertLoadPlan {
  readonly load: TopologyExpertLoad;
  readonly plan?: FrozenPlan;
  readonly terminalStepIds: readonly number[];
}

export interface DeviceCapabilityCost {
  readonly invocationOverheadNs: number;
  readonly attentionNsPerToken: number;
  readonly ffnNsPerToken: number;
  readonly draftNsPerToken: number;
  readonly lookupNsPerToken: number;
}

export type TopologyComputeCapability =
  | "attention"
  | "ffn"
  | "draft"
  | "lookup";

export interface CalibratedWorkItemRange {
  readonly minWorkItems: number;
  readonly maxWorkItems: number;
}

export type TransportOperationKind = "transfer" | "collective";

export interface TransportCalibrationPoint {
  readonly bytes: number;
  readonly durationNs: number;
}

export interface TransportCalibrationCurve {
  readonly scenarioId: string;
  readonly operation: TransportOperationKind;
  readonly linkIds: readonly string[];
  readonly participantCount: number;
  readonly algorithm: string;
  readonly trafficSignature?: string;
  readonly points: readonly TransportCalibrationPoint[];
}

export interface TopologyCostModel {
  readonly revision: typeof TOPOLOGY_COST_MODEL_REVISION;
  readonly confidence: ConfidenceClass;
  readonly source: string;
  readonly deviceCosts: Readonly<Record<SimDeviceKind, DeviceCapabilityCost>>;
  readonly activationBytesPerToken: number;
  readonly collectiveBytesPerToken: number;
  readonly coldLoadByteMultiplier: number;
  readonly applicability?: {
    readonly scenarioIds: readonly string[];
    readonly deviceKindLabels: Readonly<Record<SimDeviceKind, string>>;
  };
  readonly validWorkItemRanges?: Readonly<
    Record<
      SimDeviceKind,
      Readonly<Record<TopologyComputeCapability, CalibratedWorkItemRange>>
    >
  >;
  readonly transportCurves?: readonly TransportCalibrationCurve[];
}

export interface TopologyResourceUtilization {
  readonly resourceId: string;
  readonly busyNs: number;
  readonly capacityLanes: number;
  readonly utilization: number;
}

export interface TopologyWorkloadMetrics {
  readonly totalDurationNs: number;
  readonly foregroundDurationNs: number;
  readonly backgroundDrainNs: number;
  readonly committedTokens: number;
  readonly tokensPerSecond: number;
  readonly computeServiceNs: number;
  readonly transferServiceNs: number;
  readonly collectiveServiceNs: number;
  readonly computeUtilization: readonly TopologyResourceUtilization[];
  readonly linkUtilization: readonly TopologyResourceUtilization[];
}

export interface TopologyWorkloadResult {
  readonly scenarioId: string;
  readonly profileId: string;
  readonly confidence: ConfidenceClass;
  readonly assumptions: readonly string[];
  readonly foregroundTerminalStepId: number;
  readonly backgroundPrefetchTerminals: readonly {
    readonly prefetchId: string;
    readonly targetDomainId: string;
    readonly stepIds: readonly number[];
    readonly stepId: number;
  }[];
  readonly plan: FrozenPlan;
  readonly execution: FrozenPlanExecutionResult;
  readonly metrics: TopologyWorkloadMetrics;
}

export interface TopologyComparisonEntry {
  readonly rank: number;
  readonly scenarioId: string;
  readonly durationNs: number;
  readonly tokensPerSecond: number;
  readonly relativeToFastest: number;
  readonly confidence: ConfidenceClass;
}

interface CollectivePhaseTransfer {
  readonly linkIds: readonly string[];
  readonly bytes: number;
}

type CollectivePhases = readonly (readonly CollectivePhaseTransfer[])[];

export class TopologyWorkloadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TopologyWorkloadError";
  }
}

export const DEFAULT_TOPOLOGY_COST_MODEL: TopologyCostModel = {
  revision: TOPOLOGY_COST_MODEL_REVISION,
  confidence: "heuristic",
  source:
    "illustrative normalized decode costs; calibrate against backend traces before hardware claims",
  deviceCosts: {
    cpu: {
      invocationOverheadNs: 400_000,
      attentionNsPerToken: 220_000,
      ffnNsPerToken: 300_000,
      draftNsPerToken: 110_000,
      lookupNsPerToken: 8_000,
    },
    gpu: {
      invocationOverheadNs: 120_000,
      attentionNsPerToken: 28_000,
      ffnNsPerToken: 38_000,
      draftNsPerToken: 17_000,
      lookupNsPerToken: 8_000,
    },
    npu: {
      invocationOverheadNs: 100_000,
      attentionNsPerToken: 22_000,
      ffnNsPerToken: 52_000,
      draftNsPerToken: 24_000,
      lookupNsPerToken: 8_000,
    },
  },
  activationBytesPerToken: 1024 ** 2,
  collectiveBytesPerToken: 512 * 1024,
  coldLoadByteMultiplier: 2,
};

export function topologyProfileFromSpeculative(
  result: SpeculativeWorkloadResult,
  modelWork?: TopologyModelWork,
): TopologyWorkloadProfile {
  return {
    id: `speculative:${result.family}`,
    batchSize: 1,
    ...(modelWork === undefined ? {} : { modelWork }),
    units: result.iterations.map((iteration) => ({
      id: `iteration-${iteration.iteration}`,
      targetTokenWidth: checkedAdd(
        iteration.proposedDraftTokens,
        1,
        "verification token width",
      ),
      committedTokens: iteration.committedTokens,
      draftTokens: iteration.proposedAdditionalTokens,
      proposerExecution: result.familyContract.execution,
      proposerCostScale: result.familyContract.proposerCostScale,
      activeExperts: 1,
      warmLoadBytes: 0,
      coldLoadBytes: 0,
    })),
  };
}

export function topologyProfileFromExpertCache(
  result: ExpertCacheWorkloadResult,
  placementStrategy: TopologyExpertPlacement["strategy"] = "contiguous",
): TopologyWorkloadProfile {
  const initialization = result.trace.find(
    (event) => event.kind === "initialize",
  );
  if (initialization?.kind !== "initialize") {
    throw new TopologyWorkloadError(
      "expert-cache workload trace lacks initialization",
    );
  }
  const expertBytesById = new Map(
    initialization.config.experts.map((expert) => {
      assertPositiveSafeInteger(
        expert.bytes,
        `expert ${expert.id} bytes`,
      );
      return [expert.id, expert.bytes] as const;
    }),
  );
  if (expertBytesById.size !== initialization.config.experts.length) {
    throw new TopologyWorkloadError(
      "expert-cache initialization has duplicate expert ids",
    );
  }
  const expertPlacement: TopologyExpertPlacement = {
    strategy: placementStrategy,
    expertIds: initialization.config.experts.map((expert) => expert.id),
  };
  validateExpertPlacement(expertPlacement, "expert-cache workload");
  const bytesForExpert = (expertId: string): number => {
    const bytes = expertBytesById.get(expertId);
    if (bytes === undefined) {
      throw new TopologyWorkloadError(
        `expert-cache route references unknown expert ${expertId}`,
      );
    }
    return bytes;
  };
  let completedRoutes = 0;
  const backgroundPrefetches: TopologyBackgroundPrefetch[] = [];
  const warmPrefetchByExpert = new Map<string, string>();
  const warmPrefetchLoads = new Set<string>();
  const requiredPrefetchIdsByRequest = new Map<string, readonly string[]>();
  for (const event of result.trace) {
    if (
      event.kind === "load_start"
      && event.load.kind === "prefetch"
      && event.load.targetTier === "warm"
    ) {
      warmPrefetchLoads.add(event.load.loadId);
      backgroundPrefetches.push({
        id: event.load.loadId,
        expertId: event.load.expertId,
        afterUnitIndex: completedRoutes - 1,
        bytes: event.load.bytes,
      });
    } else if (
      event.kind === "load_complete"
      && event.targetTier === "warm"
      && warmPrefetchLoads.has(event.loadId)
    ) {
      warmPrefetchByExpert.set(event.expertId, event.loadId);
    } else if (event.kind === "evict" && event.tier === "warm") {
      warmPrefetchByExpert.delete(event.expertId);
    } else if (event.kind === "access") {
      requiredPrefetchIdsByRequest.set(
        event.requestId,
        unique(event.expertIds.flatMap((expertId, index) => {
          const producer = warmPrefetchByExpert.get(expertId);
          return event.sourceTiers[index] === "warm"
            && producer !== undefined
            ? [producer]
            : [];
        })),
      );
      completedRoutes++;
    }
  }
  return {
    id: "expert-cache",
    batchSize: 1,
    expertPlacement,
    expertTokenPlacement: "round_robin",
    units: result.routes.map((route, index) => ({
      id: `route-${index}`,
      targetTokenWidth: 1,
      committedTokens: 1,
      draftTokens: 0,
      activeExperts: route.expertIds.length,
      expertRouted: true,
      routedExperts: route.expertIds.map((expertId, routeIndex) => ({
        expertId,
        sourceTier: route.sourceTiers[routeIndex],
        loadBytes: route.sourceTiers[routeIndex] === "hot"
          ? 0
          : bytesForExpert(expertId),
      })),
      warmLoadBytes: route.expertIds.reduce((sum, expertId, routeIndex) => (
        route.sourceTiers[routeIndex] === "warm"
          ? checkedAdd(
              sum,
              bytesForExpert(expertId),
              `route ${route.requestId} warm bytes`,
            )
          : sum
      ), 0),
      coldLoadBytes: route.expertIds.reduce((sum, expertId, routeIndex) => (
        route.sourceTiers[routeIndex] === "cold"
          ? checkedAdd(
              sum,
              bytesForExpert(expertId),
              `route ${route.requestId} cold bytes`,
            )
          : sum
      ), 0),
      requiredPrefetchIds:
        requiredPrefetchIdsByRequest.get(route.requestId) ?? [],
    })),
    backgroundPrefetches,
  };
}

export function targetOnlyTopologyProfile(
  tokenCount: number,
): TopologyWorkloadProfile {
  assertPositiveSafeInteger(tokenCount, "target-only token count");
  return {
    id: "target-only",
    batchSize: 1,
    units: Array.from({ length: tokenCount }, (_, index) => ({
      id: `token-${index}`,
      targetTokenWidth: 1,
      committedTokens: 1,
      draftTokens: 0,
      activeExperts: 1,
      warmLoadBytes: 0,
      coldLoadBytes: 0,
    })),
  };
}

export function compileTopologyWorkloadPlan(
  scenario: SimulationScenario,
  profile: TopologyWorkloadProfile,
  costModel: TopologyCostModel = DEFAULT_TOPOLOGY_COST_MODEL,
): FrozenPlan {
  return compileTopologyWorkload(scenario, profile, costModel).plan;
}

export function compileTopologyExpertLoadPlan(
  scenario: SimulationScenario,
  load: TopologyExpertLoad,
  costModel: TopologyCostModel = DEFAULT_TOPOLOGY_COST_MODEL,
): TopologyExpertLoadPlan {
  if (load.id.length === 0) {
    throw new TopologyWorkloadError("expert load id must be non-empty");
  }
  if (load.expertId.length === 0) {
    throw new TopologyWorkloadError(
      `expert load ${load.id} expert id must be non-empty`,
    );
  }
  if (load.sourceTier !== "warm" && load.sourceTier !== "cold") {
    throw new TopologyWorkloadError(
      `expert load ${load.id} has invalid source tier ${String(load.sourceTier)}`,
    );
  }
  assertPositiveSafeInteger(load.bytes, `expert load ${load.id} bytes`);
  validateExpertPlacement(load.placement, `expert load ${load.id}`);
  if (!load.placement.expertIds.includes(load.expertId)) {
    throw new TopologyWorkloadError(
      `expert load ${load.id} expert ${load.expertId} is absent from the placement universe`,
    );
  }
  if (costModel.revision !== TOPOLOGY_COST_MODEL_REVISION) {
    throw new TopologyWorkloadError(
      `unsupported topology cost revision ${costModel.revision}`,
    );
  }
  const compiler = new WorkloadPlanCompiler(
    scenario,
    {
      id: `expert-load:${load.id}`,
      batchSize: 1,
      units: [{
        id: "unused-load-envelope",
        targetTokenWidth: 1,
        committedTokens: 0,
        draftTokens: 0,
        activeExperts: 1,
        warmLoadBytes: 0,
        coldLoadBytes: 0,
      }],
      expertPlacement: {
        strategy: load.placement.strategy,
        expertIds: [...load.placement.expertIds],
      },
    },
    costModel,
  );
  return compiler.compileExpertLoad(load);
}

export function resolveTopologyExpertOwnerPlacement(
  scenario: SimulationScenario,
  placement: TopologyExpertPlacement,
  expertId: string,
): PartitionPlacement {
  validateExpertPlacement(placement, `scenario ${scenario.id}`);
  const expertIndex = placement.expertIds.indexOf(expertId);
  if (expertIndex < 0) {
    throw new TopologyWorkloadError(
      `routed expert ${expertId} is absent from the placement universe`,
    );
  }
  const ffnPlacements = scenario.placements.filter((candidate) => (
    candidate.requiredCapabilities.includes("ffn")
  ));
  const expertDegree = scenario.execution.parallelism.expert;
  if (expertDegree <= 1) {
    throw new TopologyWorkloadError(
      `scenario ${scenario.id} has no expert-parallel owner mapping`,
    );
  }
  if (ffnPlacements.length !== expertDegree) {
    throw new TopologyWorkloadError(
      `scenario ${scenario.id} has ${ffnPlacements.length} FFN placements for EP=${expertDegree}`,
    );
  }
  const groups = scenario.groups.filter((candidate) => (
    candidate.orderedRanks.length === ffnPlacements.length
    && candidate.orderedRanks.every((rank) => (
      ffnPlacements.some((candidatePlacement) => (
        candidatePlacement.deviceId === rank.deviceId
      ))
    ))
  ));
  if (groups.length !== 1) {
    throw new TopologyWorkloadError(
      `scenario ${scenario.id} requires exactly one communicator for expert placements; found ${groups.length}`,
    );
  }
  const orderedPlacements = groups[0].orderedRanks.map((rank) => (
    ffnPlacements.find((candidate) => candidate.deviceId === rank.deviceId)
  )).filter(
    (candidate): candidate is PartitionPlacement => candidate !== undefined,
  );
  if (orderedPlacements.length !== ffnPlacements.length) {
    throw new TopologyWorkloadError(
      `scenario ${scenario.id} cannot order all expert placements`,
    );
  }
  const ownerIndex = placement.strategy === "round_robin"
    ? expertIndex % orderedPlacements.length
    : Math.min(
        orderedPlacements.length - 1,
        Math.floor(
          expertIndex * orderedPlacements.length / placement.expertIds.length,
        ),
      );
  return orderedPlacements[ownerIndex];
}

function compileTopologyWorkload(
  scenario: SimulationScenario,
  profile: TopologyWorkloadProfile,
  costModel: TopologyCostModel,
): {
  readonly plan: FrozenPlan;
  readonly foregroundTerminalStepId: number;
  readonly backgroundPrefetchTerminals: TopologyWorkloadResult[
    "backgroundPrefetchTerminals"
  ];
} {
  validateInputs(scenario, profile, costModel);
  const compiler = new WorkloadPlanCompiler(scenario, profile, costModel);
  const plan = compiler.compile();
  return {
    plan,
    foregroundTerminalStepId: compiler.foregroundTerminalStepId(),
    backgroundPrefetchTerminals: compiler.backgroundPrefetchTerminals(),
  };
}

export function simulateTopologyWorkload(
  scenario: SimulationScenario,
  profile: TopologyWorkloadProfile,
  costModel: TopologyCostModel = DEFAULT_TOPOLOGY_COST_MODEL,
): TopologyWorkloadResult {
  const compiled = compileTopologyWorkload(scenario, profile, costModel);
  const {
    plan,
    foregroundTerminalStepId,
    backgroundPrefetchTerminals,
  } = compiled;
  const execution = executeFrozenPlan(scenario, plan);
  replayPlanTrace(scenario, plan, execution.trace);
  const totalDurationNs = execution.completedAtNs;
  const foregroundEvent = execution.trace.operations.find(
    (event) => event.stepId === foregroundTerminalStepId,
  );
  if (foregroundEvent === undefined) {
    throw new TopologyWorkloadError(
      `foreground terminal step ${foregroundTerminalStepId} did not execute`,
    );
  }
  const foregroundDurationNs = foregroundEvent.finishNs;
  const backgroundDrainNs = totalDurationNs - foregroundDurationNs;
  const committedTokens = profile.units.reduce(
    (sum, unit) => checkedAdd(sum, unit.committedTokens, "committed tokens"),
    0,
  );
  const computeServiceNs = serviceTime(execution, "compute");
  const transferServiceNs = serviceTime(execution, "transfer");
  const collectiveServiceNs = serviceTime(execution, "collective");
  const topologyConfidence =
    topologyPerformanceConfidence(scenario, costModel);
  const confidence = profile.modelWork === undefined
    ? topologyConfidence
    : "heuristic";
  const collectiveAlgorithms = new Set(plan.steps.flatMap((step) => (
    step.operation.kind === "collective"
      ? [step.operation.algorithm]
      : []
  )));
  return {
    scenarioId: scenario.id,
    profileId: profile.id,
    confidence,
    assumptions: [
      costModel.source,
      confidence === costModel.confidence
        ? `overall timing confidence is ${confidence}`
        : profile.modelWork !== undefined
          ? "overall timing confidence is heuristic because active model work is architecture-derived and the 70% memory-bandwidth efficiency is not calibrated"
          : `overall timing confidence is ${confidence} because scenario performance evidence is weaker than the cost model`,
      "decode-only plan; prefill and request batching are outside this profile",
      "compute costs include one device-kind invocation overhead plus linear token work",
      profile.modelWork === undefined
        ? "no executable model profile is bound; compute timing is synthetic normalized work"
        : `model ${profile.modelWork.modelName} contributes an active-weight bandwidth floor for every target attention and FFN invocation`,
      scenario.execution.features.ssdStreaming
        ? "SSD streaming is enabled; cold expert loads may use declared storage domains within resource-manager limits"
        : "SSD streaming is disabled; any cold expert load or storage prefetch fails closed",
      "family-specific proposer multipliers are heuristic and provenance-labeled",
      costModel.transportCurves === undefined
        ? "transport timing uses declared directed-link bandwidth and latency"
        : "transport timing uses exact-path calibration curves without extrapolation",
      "physical routes minimize declared directed-link latency plus message-size service time, never revisit a memory domain, require pinned staging to be a true intermediate, and break equal-cost ties by the ordered link IDs",
      ...(costModel.transportCurves === undefined
        && collectiveAlgorithms.size > 0
        ? [
            ...(collectiveAlgorithms.has(COLLECTIVE_CALIBRATION_ALGORITHM)
              ? [
                  "uncalibrated all-reduce derives a logical ring from immutable communicator order and uses 2(N-1) neighbor phases",
                ]
              : []),
            ...(collectiveAlgorithms.has(
              EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM,
            )
              ? [
                  profile.expertTokenPlacement === "round_robin"
                    ? "uncalibrated AllToAllV uses N-1 pairwise-exchange phases with exact round-robin token-source to expert-owner bytes; owner-local traffic does not reserve links"
                    : "uncalibrated AllToAllV uses N-1 balanced pairwise-exchange phases; aggregate bytes do not encode route skew",
                ]
              : []),
            "each uncalibrated collective phase charges its critical path and shared-link service; the plan conservatively reserves the union of phase links",
          ]
        : []),
      scenario.execution.parallelism.expert > 1
        ? `routed experts use the profile's explicit ${profile.expertPlacement?.strategy ?? "unavailable"} owner mapping; demand loads, prefetches, and FFN work execute only on the owning EP rank`
        : "tensor-sharded FFN expert-load bytes are evenly divided across participating FFN placements",
      "warm and cold expert loads originate in each FFN placement's local host domain",
      "foreground completion is the final workload-unit terminal; independent background transfers may drain afterward",
    ],
    foregroundTerminalStepId,
    backgroundPrefetchTerminals,
    plan,
    execution,
    metrics: {
      totalDurationNs,
      foregroundDurationNs,
      backgroundDrainNs,
      committedTokens,
      tokensPerSecond: totalDurationNs === 0
        ? 0
        : committedTokens * 1_000_000_000 / totalDurationNs,
      computeServiceNs,
      transferServiceNs,
      collectiveServiceNs,
      computeUtilization: resourceUtilization(
        scenario,
        execution,
        "compute:",
        totalDurationNs,
      ),
      linkUtilization: resourceUtilization(
        scenario,
        execution,
        "link:",
        totalDurationNs,
      ),
    },
  };
}

function topologyPerformanceConfidence(
  scenario: SimulationScenario,
  costModel: TopologyCostModel,
): ConfidenceClass {
  const evidence = [
    costModel.confidence,
    ...scenario.devices.map((device) => device.provenance.confidence),
    ...scenario.memoryDomains.map((domain) => domain.provenance.confidence),
    ...scenario.links.map((link) => link.provenance.confidence),
  ];
  if (evidence.includes("heuristic")) {
    return "heuristic";
  }
  // Timing remains an estimate even when the structural inputs are exact or bounded.
  return "calibrated";
}

export function compareTopologyWorkloads(
  scenarios: readonly SimulationScenario[],
  profile:
    | TopologyWorkloadProfile
    | ((scenario: SimulationScenario) => TopologyWorkloadProfile),
  costModel: TopologyCostModel = DEFAULT_TOPOLOGY_COST_MODEL,
): readonly TopologyComparisonEntry[] {
  if (scenarios.length === 0) {
    throw new TopologyWorkloadError("at least one scenario is required");
  }
  const results = scenarios.map((scenario) => (
    simulateTopologyWorkload(
      scenario,
      typeof profile === "function" ? profile(scenario) : profile,
      costModel,
    )
  )).sort((left, right) => (
    left.metrics.totalDurationNs - right.metrics.totalDurationNs
    || left.scenarioId.localeCompare(right.scenarioId)
  ));
  const fastest = results[0].metrics.totalDurationNs;
  return results.map((result, index) => ({
    rank: index + 1,
    scenarioId: result.scenarioId,
    durationNs: result.metrics.totalDurationNs,
    tokensPerSecond: result.metrics.tokensPerSecond,
    relativeToFastest: fastest === 0
      ? 1
      : result.metrics.totalDurationNs / fastest,
    confidence: result.confidence,
  }));
}

class WorkloadPlanCompiler {
  private readonly placements: readonly PartitionPlacement[];
  private readonly draftPlacements: readonly PartitionPlacement[];
  private readonly rankByDevice = new Map<string, string>();
  private readonly steps: PlanStep[] = [];
  private nextStepId = 0;
  private readonly collectiveSequenceByGroup = new Map<string, number>();
  private readonly expertOwnerByKey = new Map<string, PartitionPlacement>();
  private readonly transferRouteByKey = new Map<string, TransferRoute | null>();
  private readonly lastWriterByAllocation = new Map<string, number>();
  private readonly readersSinceWriteByAllocation = new Map<
    string,
    Set<number>
  >();
  private previousTerminalStepId?: number;
  private readonly backgroundPrefetchTerminalByDomain = new Map<
    string,
    number
  >();
  private readonly backgroundPrefetchTerminalByIdAndDomain = new Map<
    string,
    number
  >();
  private readonly backgroundPrefetchTerminalSteps: Array<{
    readonly prefetchId: string;
    readonly targetDomainId: string;
    readonly stepIds: readonly number[];
    readonly stepId: number;
  }> = [];

  constructor(
    private readonly scenario: SimulationScenario,
    private readonly profile: TopologyWorkloadProfile,
    private readonly costModel: TopologyCostModel,
  ) {
    this.placements = scenario.placements.filter((placement) => (
      placement.requiredCapabilities.includes("attention")
      || placement.requiredCapabilities.includes("ffn")
    ));
    this.draftPlacements = scenario.placements.filter((placement) => (
      placement.requiredCapabilities.includes("draft")
    ));
    for (const group of scenario.groups) {
      for (const rank of group.orderedRanks) {
        const prior = this.rankByDevice.get(rank.deviceId);
        if (prior !== undefined && prior !== rank.rankId) {
          throw new TopologyWorkloadError(
            `device ${rank.deviceId} has multiple rank identities`,
          );
        }
        this.rankByDevice.set(rank.deviceId, rank.rankId);
      }
    }
  }

  compile(): FrozenPlan {
    if (this.placements.length === 0) {
      throw new TopologyWorkloadError(
        `scenario ${this.scenario.id} has no target placement`,
      );
    }
    this.compileBackgroundPrefetches(-1, undefined);
    for (const [unitIndex, unit] of this.profile.units.entries()) {
      this.compileUnit(unit);
      this.compileBackgroundPrefetches(
        unitIndex,
        this.previousTerminalStepId,
      );
    }
    return {
      contractRevision: PLAN_CONTRACT_REVISION,
      id: `topology-workload:${this.profile.id}`,
      executionId: `${this.scenario.id}:${this.profile.id}`,
      topologyEpoch: this.scenario.execution.topologyEpoch,
      steps: this.steps,
    };
  }

  compileExpertLoad(load: TopologyExpertLoad): TopologyExpertLoadPlan {
    const ffnPlacements = this.placements.filter((placement) => (
      placement.requiredCapabilities.includes("ffn")
    ));
    if (ffnPlacements.length === 0) {
      throw new TopologyWorkloadError(
        `scenario ${this.scenario.id} has no FFN placement for ${load.id}`,
      );
    }
    const loadPlacements =
      this.scenario.execution.parallelism.expert > 1
        ? [this.expertOwnerPlacement(load.expertId, ffnPlacements)]
        : ffnPlacements;
    const placementsByTarget = new Map<string, PartitionPlacement[]>();
    for (const placement of loadPlacements) {
      const targetDomainId = workspaceDomain(placement);
      const placements = placementsByTarget.get(targetDomainId) ?? [];
      placements.push(placement);
      placementsByTarget.set(targetDomainId, placements);
    }
    const terminalStepIds: number[] = [];
    const lastColdTerminalByNode = new Map<string, number>();
    for (const [targetDomainId, targetPlacements] of [
      ...placementsByTarget.entries(),
    ].sort(([left], [right]) => left.localeCompare(right))) {
      const target = this.scenario.memoryDomains.find(
        (domain) => domain.id === targetDomainId,
      );
      const placement = targetPlacements[0];
      if (target === undefined || placement === undefined) {
        throw new TopologyWorkloadError(
          `expert load ${load.id} has unknown target ${targetDomainId}`,
        );
      }
      if (load.sourceTier === "cold") {
        this.requireSsdStreaming(`expert load ${load.id}`);
      }
      const sourceDomainId = load.sourceTier === "warm"
        ? this.cacheSourceDomain(targetDomainId)
        : this.scenario.memoryDomains.find((domain) => (
            domain.nodeId === target.nodeId && domain.kind === "storage"
          ))?.id;
      if (sourceDomainId === undefined) {
        throw new TopologyWorkloadError(
          `expert load ${load.id} has no ${load.sourceTier} source on ${target.nodeId}`,
        );
      }
      if (sourceDomainId === targetDomainId) {
        continue;
      }
      const shardBytes = Math.ceil(
        checkedMultiply(
          load.bytes,
          targetPlacements.length,
          `expert load ${load.id} target bytes`,
        ) / loadPlacements.length,
      );
      const sourceAllocationId = load.sourceTier === "warm"
        ? `expert-warm-cache:${target.nodeId}`
        : `expert-backing:${target.nodeId}`;
      const priorColdTerminal = load.sourceTier === "cold"
        ? lastColdTerminalByNode.get(target.nodeId)
        : undefined;
      const stepIds = this.addTransferPath(
        sourceDomainId,
        targetDomainId,
        shardBytes,
        priorColdTerminal === undefined ? [] : [priorColdTerminal],
        [this.rank(placement.deviceId)],
        {
          sourceAllocationId,
          targetAllocationId: expertHotCacheId(placement),
        },
      );
      if (stepIds.length === 0) {
        throw new TopologyWorkloadError(
          `expert load ${load.id} produced no path to ${targetDomainId}`,
        );
      }
      const terminalStepId = stepIds[stepIds.length - 1];
      terminalStepIds.push(terminalStepId);
      if (load.sourceTier === "cold") {
        lastColdTerminalByNode.set(target.nodeId, terminalStepId);
      }
    }
    if (this.steps.length === 0) {
      return {
        load: cloneExpertLoad(load),
        terminalStepIds: [],
      };
    }
    return {
      load: cloneExpertLoad(load),
      plan: {
        contractRevision: PLAN_CONTRACT_REVISION,
        id: `topology-expert-load:${load.id}`,
        executionId: `${this.scenario.id}:expert-load:${load.id}`,
        topologyEpoch: this.scenario.execution.topologyEpoch,
        steps: [...this.steps],
      },
      terminalStepIds,
    };
  }

  foregroundTerminalStepId(): number {
    if (this.previousTerminalStepId === undefined) {
      throw new TopologyWorkloadError(
        `profile ${this.profile.id} has no foreground terminal step`,
      );
    }
    return this.previousTerminalStepId;
  }

  backgroundPrefetchTerminals(): TopologyWorkloadResult[
    "backgroundPrefetchTerminals"
  ] {
    return this.backgroundPrefetchTerminalSteps.map((terminal) => ({
      ...terminal,
      stepIds: [...terminal.stepIds],
    }));
  }

  private compileBackgroundPrefetches(
    afterUnitIndex: number,
    triggerStepId: number | undefined,
  ): void {
    const prefetches = (this.profile.backgroundPrefetches ?? []).filter(
      (prefetch) => prefetch.afterUnitIndex === afterUnitIndex,
    );
    if (prefetches.length === 0) {
      return;
    }
    const ffnPlacements = this.placements.filter((placement) => (
      placement.requiredCapabilities.includes("ffn")
    ));
    for (const prefetch of prefetches) {
      const targetPlacements = (
        prefetch.expertId !== undefined
        && this.scenario.execution.parallelism.expert > 1
      )
        ? [this.expertOwnerPlacement(prefetch.expertId, ffnPlacements)]
        : ffnPlacements;
      const targets = unique(targetPlacements.map((placement) => (
        this.cacheSourceDomain(workspaceDomain(placement))
      )));
      for (const targetDomainId of targets) {
        this.requireSsdStreaming(`background prefetch ${prefetch.id}`);
        const localPlacementCount = targetPlacements.filter((placement) => (
          this.cacheSourceDomain(workspaceDomain(placement)) === targetDomainId
        )).length;
        const shardBytes = Math.ceil(
          checkedMultiply(
            prefetch.bytes,
            localPlacementCount,
            "background prefetch node bytes",
          )
          / Math.max(1, targetPlacements.length),
        );
        const target = this.scenario.memoryDomains.find(
          (domain) => domain.id === targetDomainId,
        );
        const storage = this.scenario.memoryDomains.find((domain) => (
          domain.nodeId === target?.nodeId && domain.kind === "storage"
        ));
        if (!target || !storage) {
          throw new TopologyWorkloadError(
            `background prefetch ${prefetch.id} lacks storage/warm domains for ${targetDomainId}`,
          );
        }
        const placement = targetPlacements.find((candidate) => (
          this.cacheSourceDomain(workspaceDomain(candidate)) === targetDomainId
        ));
        if (!placement) {
          throw new TopologyWorkloadError(
            `background prefetch ${prefetch.id} lacks an FFN placement for ${targetDomainId}`,
          );
        }
        const previousBackground = this.backgroundPrefetchTerminalByDomain.get(
          targetDomainId,
        );
        const dependencies = uniqueNumbers([
          ...(triggerStepId === undefined ? [] : [triggerStepId]),
          ...(previousBackground === undefined ? [] : [previousBackground]),
        ]);
        const stepIds = this.addTransferPath(
          storage.id,
          targetDomainId,
          shardBytes,
          dependencies,
          [this.rank(placement.deviceId)],
          {
            sourceAllocationId: `expert-backing:${target.nodeId}`,
            targetAllocationId: `expert-warm-cache:${target.nodeId}`,
          },
        );
        if (stepIds.length === 0) {
          throw new TopologyWorkloadError(
            `background prefetch ${prefetch.id} produced no transfer`,
          );
        }
        this.backgroundPrefetchTerminalByDomain.set(
          targetDomainId,
          stepIds[stepIds.length - 1],
        );
        this.backgroundPrefetchTerminalByIdAndDomain.set(
          prefetchDomainKey(prefetch.id, targetDomainId),
          stepIds[stepIds.length - 1],
        );
        this.backgroundPrefetchTerminalSteps.push({
          prefetchId: prefetch.id,
          targetDomainId,
          stepIds: [...stepIds],
          stepId: stepIds[stepIds.length - 1],
        });
      }
    }
  }

  private compileUnit(unit: TopologyWorkUnit): void {
    let entryDependencies = this.previousTerminalStepId === undefined
      ? []
      : [this.previousTerminalStepId];

    if (unit.draftTokens > 0) {
      const execution = unit.proposerExecution ?? "separate_model";
      const capability = execution === "cpu_lookup" ? "lookup" : "draft";
      const draftPlacement = execution === "cpu_lookup"
        ? this.hostLookupPlacement()
        : execution === "separate_model"
          ? this.draftPlacements[0] ?? this.placements.find((placement) => (
              this.device(placement.deviceId).capabilities.includes("draft")
            ))
          : this.placements.find((placement) => (
              this.device(placement.deviceId).capabilities.includes("draft")
              && (
                placement.requiredCapabilities.includes("attention")
                || execution === "target_early_exit"
              )
            )) ?? this.placements.find((placement) => (
              this.device(placement.deviceId).capabilities.includes("draft")
            ));
      if (!draftPlacement) {
        throw new TopologyWorkloadError(
          `scenario ${this.scenario.id} has no device capable of ${execution} execution`,
        );
      }
      const baseDuration = this.computeDuration(
        draftPlacement.deviceId,
        capability,
        unit.draftTokens,
        1,
        false,
      );
      const draftStep = this.addCompute(
        draftPlacement,
        capability,
        Math.max(
          1,
          Math.ceil(baseDuration * (unit.proposerCostScale ?? 1)),
        ),
        entryDependencies,
      );
      entryDependencies = [draftStep];
    }

    if (this.scenario.execution.parallelism.pipeline > 1) {
      this.previousTerminalStepId = this.compilePipeline(
        unit,
        entryDependencies,
      );
      return;
    }
    if (
      (
        this.scenario.execution.parallelism.tensor > 1
        || (
          unit.expertRouted === true
          && this.scenario.execution.parallelism.expert > 1
        )
      )
      && this.placements.length > 1
    ) {
      this.previousTerminalStepId = this.compileTensorParallel(
        unit,
        entryDependencies,
      );
      return;
    }
    const cacheLoad = this.prepareCacheLoad(
      this.placements[0],
      unit,
      entryDependencies,
      1,
    );
    this.previousTerminalStepId = this.compilePlacement(
      this.placements[0],
      unit,
      cacheLoad.dependencies,
      cacheLoad.localMemoryPenaltyNs,
    );
  }

  private compilePipeline(
    unit: TopologyWorkUnit,
    dependencies: readonly number[],
  ): number {
    let currentDependencies = [...dependencies];
    let terminal = currentDependencies[0];
    for (let index = 0; index < this.placements.length; index++) {
      const placement = this.placements[index];
      const cacheLoad = this.prepareCacheLoad(
        placement,
        unit,
        currentDependencies,
        1,
      );
      terminal = this.compilePlacement(
        placement,
        unit,
        cacheLoad.dependencies,
        cacheLoad.localMemoryPenaltyNs,
      );
      const next = this.placements[index + 1];
      if (next) {
        const bytes = checkedMultiply(
          this.costModel.activationBytesPerToken,
          checkedMultiply(
            unit.targetTokenWidth,
            this.profile.batchSize,
            "pipeline token batch",
          ),
          "pipeline activation bytes",
        );
        const transferSteps = this.addTransferPath(
          workspaceDomain(placement),
          workspaceDomain(next),
          bytes,
          [terminal],
          [this.rank(placement.deviceId), this.rank(next.deviceId)],
        );
        currentDependencies = transferSteps.length === 0
          ? [terminal]
          : [transferSteps[transferSteps.length - 1]];
      }
    }
    if (terminal === undefined) {
      throw new TopologyWorkloadError("pipeline produced no terminal step");
    }
    return terminal;
  }

  private compileTensorParallel(
    unit: TopologyWorkUnit,
    dependencies: readonly number[],
  ): number {
    if (
      unit.expertRouted === true
      && this.scenario.execution.parallelism.expert > 1
    ) {
      return this.compileTensorExpertParallel(unit, dependencies);
    }
    const ffnPlacementCount = this.placements.filter((placement) => (
      placement.requiredCapabilities.includes("ffn")
    )).length;
    const terminals = this.placements.map((placement) => {
      const cacheLoad = this.prepareCacheLoad(
        placement,
        unit,
        dependencies,
        Math.max(1, ffnPlacementCount),
      );
      return this.compilePlacement(
        placement,
        unit,
        cacheLoad.dependencies,
        cacheLoad.localMemoryPenaltyNs,
      );
    });
    const bytes = checkedMultiply(
      this.costModel.collectiveBytesPerToken,
      checkedMultiply(
        unit.targetTokenWidth,
        this.profile.batchSize,
        "collective token batch",
      ),
      "collective bytes",
    );
    return this.addCollective(
      this.groupForPlacements(this.placements, "tensor"),
      COLLECTIVE_CALIBRATION_ALGORITHM,
      bytes,
      terminals,
      this.placements.map((placement) => workspaceId(placement)),
    );
  }

  private compileTensorExpertParallel(
    unit: TopologyWorkUnit,
    dependencies: readonly number[],
  ): number {
    const attentionPlacements = this.placements.filter((placement) => (
      placement.requiredCapabilities.includes("attention")
    ));
    const ffnPlacements = this.placements.filter((placement) => (
      placement.requiredCapabilities.includes("ffn")
    ));
    const expertDegree = this.scenario.execution.parallelism.expert;
    if (
      attentionPlacements.length !== this.scenario.execution.parallelism.tensor
      || ffnPlacements.length !== expertDegree
    ) {
      throw new TopologyWorkloadError(
        `scenario ${this.scenario.id} cannot map ${attentionPlacements.length} attention and ${ffnPlacements.length} FFN placements onto TP=${this.scenario.execution.parallelism.tensor}, EP=${expertDegree}`,
      );
    }
    const group = this.groupForPlacements(this.placements, "tensor/EP");
    const attentionTerminals = attentionPlacements.map((placement) => (
      this.addCompute(
        placement,
        "attention",
        this.computeDuration(
          placement.deviceId,
          "attention",
          unit.targetTokenWidth,
          1,
          false,
        ),
        dependencies,
      )
    ));
    const tensorBytes = checkedMultiply(
      this.costModel.collectiveBytesPerToken,
      checkedMultiply(
        unit.targetTokenWidth,
        this.profile.batchSize,
        "tensor collective token batch",
      ),
      "tensor collective bytes",
    );
    const attentionReduced = this.addCollective(
      group,
      COLLECTIVE_CALIBRATION_ALGORITHM,
      tensorBytes,
      attentionTerminals,
      attentionPlacements.map((placement) => workspaceId(placement)),
    );
    const routedTokens = checkedMultiply(
      checkedMultiply(
        unit.targetTokenWidth,
        this.profile.batchSize,
        "expert dispatch token batch",
      ),
      Math.max(1, unit.activeExperts),
      "expert routed token copies",
    );
    const expertBytes = checkedMultiply(
      this.costModel.activationBytesPerToken,
      routedTokens,
      "expert dispatch bytes",
    );
    const dispatchTraffic = this.routedExpertTraffic(
      group,
      unit,
      ffnPlacements,
    );
    const dispatched = this.addCollective(
      group,
      EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM,
      expertBytes,
      [attentionReduced],
      attentionPlacements.map((placement) => workspaceId(placement)),
      dispatchTraffic,
    );
    const workItemsByPlacement = new Map<string, number>();
    for (const routed of unit.routedExperts ?? []) {
      const owner = this.expertOwnerPlacement(
        routed.expertId,
        ffnPlacements,
      );
      workItemsByPlacement.set(
        owner.partitionId,
        checkedAdd(
          workItemsByPlacement.get(owner.partitionId) ?? 0,
          1,
          `expert work on ${owner.partitionId}`,
        ),
      );
    }
    const activeFfnPlacements: PartitionPlacement[] = [];
    const expertTerminals = ffnPlacements.flatMap((placement) => {
      const workItems = workItemsByPlacement.get(placement.partitionId) ?? 0;
      if (workItems === 0) {
        return [];
      }
      activeFfnPlacements.push(placement);
      const cacheLoad = this.prepareCacheLoad(
        placement,
        unit,
        [dispatched],
        1,
      );
      const durationNs = checkedAdd(
        this.computeDurationForWorkItems(
          placement.deviceId,
          "ffn",
          workItems,
          1,
        ),
        cacheLoad.localMemoryPenaltyNs,
        "expert FFN duration",
      );
      return [this.addCompute(
        placement,
        "ffn",
        durationNs,
        cacheLoad.dependencies,
        [expertHotCacheId(placement)],
      )];
    });
    if (expertTerminals.length === 0) {
      throw new TopologyWorkloadError(
        `routed unit ${unit.id} produced no expert work`,
      );
    }
    return this.addCollective(
      group,
      EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM,
      expertBytes,
      expertTerminals,
      activeFfnPlacements.map((placement) => workspaceId(placement)),
      transposeTraffic(dispatchTraffic),
    );
  }

  private addCollective(
    group: SimulationScenario["groups"][number],
    algorithm: CollectiveAlgorithm,
    bytes: number,
    dependencies: readonly number[],
    reads: readonly string[],
    allToAllTraffic?: readonly (readonly number[])[],
  ): number {
    if (group.orderedRanks.length < 2) {
      throw new TopologyWorkloadError(
        `collective path compilation requires at least two ranks; ${group.id} has ${group.orderedRanks.length}`,
      );
    }
    const phases = this.collectivePhases(
      group,
      algorithm,
      bytes,
      allToAllTraffic,
    );
    const linkIds = unique(phases.flatMap((phase) => (
      phase.flatMap((transfer) => transfer.linkIds)
    )));
    const durationNs = this.transportDuration(
      "collective",
      linkIds,
      group.orderedRanks.length,
      algorithm,
      bytes,
      phases,
      allToAllTraffic === undefined
        ? undefined
        : canonicalAllToAllTrafficSignature(allToAllTraffic),
    );
    const commSequenceId =
      this.collectiveSequenceByGroup.get(group.id) ?? 0;
    this.collectiveSequenceByGroup.set(group.id, commSequenceId + 1);
    return this.addStep({
      participants: group.orderedRanks.map((rank) => rank.rankId),
      dependencies,
      reads: unique(reads),
      writes: [],
      operation: {
        kind: "collective",
        groupId: group.id,
        commSequenceId,
        algorithm,
        linkIds,
        durationNs,
      },
    });
  }

  private prepareCacheLoad(
    placement: PartitionPlacement,
    unit: TopologyWorkUnit,
    dependencies: readonly number[],
    shardCount: number,
  ): {
    readonly dependencies: readonly number[];
    readonly localMemoryPenaltyNs: number;
  } {
    if (!placement.requiredCapabilities.includes("ffn")) {
      return { dependencies, localMemoryPenaltyNs: 0 };
    }
    if (unit.routedExperts !== undefined) {
      return this.prepareRoutedCacheLoads(
        placement,
        unit,
        dependencies,
        shardCount,
      );
    }
    const effectiveBytes = checkedAdd(
      unit.warmLoadBytes,
      checkedMultiply(
        unit.coldLoadBytes,
        this.costModel.coldLoadByteMultiplier,
        "effective cold load bytes",
      ),
      "effective cache load bytes",
    );
    if (effectiveBytes === 0) {
      return { dependencies, localMemoryPenaltyNs: 0 };
    }
    const shardBytes = Math.ceil(effectiveBytes / shardCount);
    const targetDomain = workspaceDomain(placement);
    const sourceDomain = this.cacheSourceDomain(targetDomain);
    const prefetchDependencies = (unit.requiredPrefetchIds ?? []).flatMap(
      (prefetchId) => {
        const prefetch = this.profile.backgroundPrefetches?.find(
          (candidate) => candidate.id === prefetchId,
        );
        if (prefetch === undefined) {
          throw new TopologyWorkloadError(
            `unit ${unit.id} requires unknown background prefetch ${prefetchId}`,
          );
        }
        if (
          prefetch.expertId !== undefined
          && this.scenario.execution.parallelism.expert > 1
        ) {
          const ffnPlacements = this.placements.filter((candidate) => (
            candidate.requiredCapabilities.includes("ffn")
          ));
          const owner = this.expertOwnerPlacement(
            prefetch.expertId,
            ffnPlacements,
          );
          const ownerSourceDomain = this.cacheSourceDomain(
            workspaceDomain(owner),
          );
          if (ownerSourceDomain !== sourceDomain) {
            return [];
          }
        }
        const terminal = this.backgroundPrefetchTerminalByIdAndDomain.get(
          prefetchDomainKey(prefetchId, sourceDomain),
        );
        if (terminal !== undefined) {
          return [terminal];
        }
        throw new TopologyWorkloadError(
          `unit ${unit.id} requires unresolved background prefetch ${prefetchId} for ${sourceDomain}`,
        );
      },
    );
    const priorWarmWriter =
      this.backgroundPrefetchTerminalByDomain.get(sourceDomain);
    const cacheDependencies = uniqueNumbers([
      ...dependencies,
      ...prefetchDependencies,
      ...(priorWarmWriter === undefined ? [] : [priorWarmWriter]),
    ]);
    if (sourceDomain === targetDomain) {
      return {
        dependencies: cacheDependencies,
        localMemoryPenaltyNs: this.domainDuration(targetDomain, shardBytes),
      };
    }
    const transferSteps = this.addTransferPath(
      sourceDomain,
      targetDomain,
      shardBytes,
      cacheDependencies,
      [this.rank(placement.deviceId)],
      {
        sourceAllocationId: `expert-warm-cache:${
          this.device(placement.deviceId).nodeId
        }`,
        targetAllocationId: expertHotCacheId(placement),
      },
    );
    return {
      dependencies: transferSteps.length === 0
        ? cacheDependencies
        : [transferSteps[transferSteps.length - 1]],
      localMemoryPenaltyNs: 0,
    };
  }

  private prepareRoutedCacheLoads(
    placement: PartitionPlacement,
    unit: TopologyWorkUnit,
    dependencies: readonly number[],
    shardCount: number,
  ): {
    readonly dependencies: readonly number[];
    readonly localMemoryPenaltyNs: number;
  } {
    const routedExperts = unit.routedExperts;
    if (routedExperts === undefined) {
      throw new TopologyWorkloadError(
        `routed cache load ${unit.id} lacks expert assignments`,
      );
    }
    const ffnPlacements = this.placements.filter((candidate) => (
      candidate.requiredCapabilities.includes("ffn")
    ));
    const localRoutes = this.scenario.execution.parallelism.expert > 1
      ? routedExperts.filter((routed) => (
          this.expertOwnerPlacement(
            routed.expertId,
            ffnPlacements,
          ).partitionId === placement.partitionId
        ))
      : routedExperts;
    const warmBytes = Math.ceil(localRoutes.reduce((sum, routed) => (
      routed.sourceTier === "warm"
        ? checkedAdd(sum, routed.loadBytes, "local warm expert bytes")
        : sum
    ), 0) / shardCount);
    const coldBytes = Math.ceil(localRoutes.reduce((sum, routed) => (
      routed.sourceTier === "cold"
        ? checkedAdd(sum, routed.loadBytes, "local cold expert bytes")
        : sum
    ), 0) / shardCount);
    if (warmBytes === 0 && coldBytes === 0) {
      return { dependencies, localMemoryPenaltyNs: 0 };
    }
    const targetDomain = workspaceDomain(placement);
    const warmDomain = this.cacheSourceDomain(targetDomain);
    const prefetchDependencies = warmBytes === 0
      ? []
      : (unit.requiredPrefetchIds ?? []).flatMap((prefetchId) => (
          this.prefetchDependency(
            unit,
            prefetchId,
            warmDomain,
            ffnPlacements,
          )
        ));
    const priorWarmWriter = warmBytes === 0
      ? undefined
      : this.backgroundPrefetchTerminalByDomain.get(warmDomain);
    const warmDependencies = uniqueNumbers([
      ...dependencies,
      ...prefetchDependencies,
      ...(priorWarmWriter === undefined ? [] : [priorWarmWriter]),
    ]);
    const terminals: number[] = [];
    let localMemoryPenaltyNs = 0;
    if (warmBytes > 0) {
      if (warmDomain === targetDomain) {
        localMemoryPenaltyNs = this.domainDuration(targetDomain, warmBytes);
      } else {
        const warmSteps = this.addTransferPath(
          warmDomain,
          targetDomain,
          warmBytes,
          warmDependencies,
          [this.rank(placement.deviceId)],
          {
            sourceAllocationId: `expert-warm-cache:${
              this.device(placement.deviceId).nodeId
            }`,
            targetAllocationId: expertHotCacheId(placement),
          },
        );
        if (warmSteps.length === 0) {
          throw new TopologyWorkloadError(
            `routed unit ${unit.id} produced no warm path to ${targetDomain}`,
          );
        }
        terminals.push(warmSteps[warmSteps.length - 1]);
      }
    }
    if (coldBytes > 0) {
      this.requireSsdStreaming(`routed unit ${unit.id}`);
      const nodeId = this.device(placement.deviceId).nodeId;
      const storage = this.scenario.memoryDomains.find((domain) => (
        domain.nodeId === nodeId && domain.kind === "storage"
      ));
      if (storage === undefined) {
        throw new TopologyWorkloadError(
          `routed unit ${unit.id} lacks cold storage on ${nodeId}`,
        );
      }
      const coldSteps = this.addTransferPath(
        storage.id,
        targetDomain,
        coldBytes,
        terminals.length === 0
          ? dependencies
          : [terminals[terminals.length - 1]],
        [this.rank(placement.deviceId)],
        {
          sourceAllocationId: `expert-backing:${nodeId}`,
          targetAllocationId: expertHotCacheId(placement),
        },
      );
      if (coldSteps.length === 0) {
        throw new TopologyWorkloadError(
          `routed unit ${unit.id} produced no cold path to ${targetDomain}`,
        );
      }
      terminals.push(coldSteps[coldSteps.length - 1]);
    }
    return {
      dependencies: uniqueNumbers([
        ...terminals,
        ...(warmBytes > 0 && warmDomain === targetDomain
          ? warmDependencies
          : []),
      ]),
      localMemoryPenaltyNs,
    };
  }

  private prefetchDependency(
    unit: TopologyWorkUnit,
    prefetchId: string,
    sourceDomain: string,
    ffnPlacements: readonly PartitionPlacement[],
  ): readonly number[] {
    const prefetch = this.profile.backgroundPrefetches?.find(
      (candidate) => candidate.id === prefetchId,
    );
    if (prefetch === undefined) {
      throw new TopologyWorkloadError(
        `unit ${unit.id} requires unknown background prefetch ${prefetchId}`,
      );
    }
    if (
      prefetch.expertId !== undefined
      && this.scenario.execution.parallelism.expert > 1
    ) {
      const owner = this.expertOwnerPlacement(
        prefetch.expertId,
        ffnPlacements,
      );
      const ownerSourceDomain = this.cacheSourceDomain(
        workspaceDomain(owner),
      );
      if (ownerSourceDomain !== sourceDomain) {
        return [];
      }
    }
    const terminal = this.backgroundPrefetchTerminalByIdAndDomain.get(
      prefetchDomainKey(prefetchId, sourceDomain),
    );
    if (terminal === undefined) {
      throw new TopologyWorkloadError(
        `unit ${unit.id} requires unresolved background prefetch ${prefetchId} for ${sourceDomain}`,
      );
    }
    return [terminal];
  }

  private compilePlacement(
    placement: PartitionPlacement,
    unit: TopologyWorkUnit,
    dependencies: readonly number[],
    extraDurationNs: number,
  ): number {
    const capabilities = placement.requiredCapabilities.filter(
      (capability): capability is "attention" | "ffn" => (
        capability === "attention" || capability === "ffn"
      ),
    );
    if (capabilities.length === 0) {
      throw new TopologyWorkloadError(
        `placement ${placement.partitionId} has no target compute capability`,
      );
    }
    let currentDependencies = [...dependencies];
    let terminal = currentDependencies[0];
    for (let index = 0; index < capabilities.length; index++) {
      const capability = capabilities[index];
      const durationNs = checkedAdd(
        this.computeDuration(
          placement.deviceId,
          capability,
          unit.targetTokenWidth,
          capability === "ffn" ? unit.activeExperts : 1,
          unit.expertRouted === true,
        ),
        index === 0 ? extraDurationNs : 0,
        "placement compute duration",
      );
      terminal = this.addCompute(
        placement,
        capability,
        durationNs,
        currentDependencies,
        capability === "ffn" && unit.expertRouted === true
          ? [expertHotCacheId(placement)]
          : [],
      );
      currentDependencies = [terminal];
    }
    if (terminal === undefined) {
      throw new TopologyWorkloadError(
        `placement ${placement.partitionId} produced no step`,
      );
    }
    return terminal;
  }

  private addCompute(
    placement: PartitionPlacement,
    capability: "attention" | "ffn" | "draft" | "lookup",
    durationNs: number,
    dependencies: readonly number[],
    additionalReads: readonly string[] = [],
  ): number {
    const state = placement.allocations
      .filter((allocation) => (
        allocation.purpose === "kv"
        || allocation.purpose === "workspace"
        || allocation.purpose === "sidecar"
      ))
      .map((allocation) => allocation.physicalAllocationId);
    const weights = placement.allocations
      .filter((allocation) => allocation.purpose === "weights")
      .map((allocation) => allocation.physicalAllocationId);
    return this.addStep({
      participants: [this.rank(placement.deviceId)],
      dependencies,
      reads: unique([...weights, ...additionalReads]),
      writes: state,
      operation: {
        kind: "compute",
        deviceId: placement.deviceId,
        capability,
        durationNs,
      },
    });
  }

  private addTransferPath(
    sourceDomainId: string,
    targetDomainId: string,
    bytes: number,
    dependencies: readonly number[],
    participants: readonly string[],
    endpointAllocations?: {
      readonly sourceAllocationId: string;
      readonly targetAllocationId: string;
    },
  ): number[] {
    const route = this.transferRoute(sourceDomainId, targetDomainId, bytes);
    if (!route) {
      throw new TopologyWorkloadError(
        `no transfer path from ${sourceDomainId} to ${targetDomainId}`,
      );
    }
    const domains = route.domainIds;
    if (domains.length <= 1) {
      return [];
    }
    const stepIds: number[] = [];
    let currentDependencies = [...dependencies];
    for (let index = 0; index < domains.length - 1; index++) {
      const source = domains[index];
      const target = domains[index + 1];
      const linkId = route.linkIds[index];
      const link = this.scenario.links.find(
        (candidate) => candidate.id === linkId,
      ) ?? fail(`unknown routed link ${String(linkId)}`);
      const stepId = this.addStep({
        participants: unique(participants),
        dependencies: currentDependencies,
        reads: [
          index === 0 && endpointAllocations !== undefined
            ? endpointAllocations.sourceAllocationId
            : this.buffer(source, sourceDomainId, targetDomainId),
        ],
        writes: [
          index === domains.length - 2 && endpointAllocations !== undefined
            ? endpointAllocations.targetAllocationId
            : this.buffer(target, sourceDomainId, targetDomainId),
        ],
        operation: {
          kind: "transfer",
          linkId: link.id,
          durationNs: this.transportDuration(
            "transfer",
            [link.id],
            2,
            TRANSFER_CALIBRATION_ALGORITHM,
            bytes,
          ),
        },
      });
      stepIds.push(stepId);
      currentDependencies = [stepId];
    }
    return stepIds;
  }

  private addStep(
    input: Omit<PlanStep, "id">,
  ): number {
    const id = this.nextStepId++;
    const reads = unique(input.reads);
    const writes = unique(input.writes);
    const writeSet = new Set(writes);
    const hazardDependencies: number[] = [];
    for (const allocationId of reads) {
      const writer = this.lastWriterByAllocation.get(allocationId);
      if (writer !== undefined) {
        hazardDependencies.push(writer);
      }
    }
    for (const allocationId of writes) {
      const writer = this.lastWriterByAllocation.get(allocationId);
      if (writer !== undefined) {
        hazardDependencies.push(writer);
      }
      hazardDependencies.push(
        ...(this.readersSinceWriteByAllocation.get(allocationId) ?? []),
      );
    }
    const dependencies = uniqueNumbers([
      ...input.dependencies,
      ...hazardDependencies,
    ]);
    this.steps.push({
      id,
      ...input,
      dependencies,
      reads,
      writes,
    });
    for (const allocationId of writes) {
      this.lastWriterByAllocation.set(allocationId, id);
      this.readersSinceWriteByAllocation.delete(allocationId);
    }
    for (const allocationId of reads) {
      if (writeSet.has(allocationId)) {
        continue;
      }
      const readers =
        this.readersSinceWriteByAllocation.get(allocationId) ?? new Set();
      readers.add(id);
      this.readersSinceWriteByAllocation.set(allocationId, readers);
    }
    if (this.steps.length > this.scenario.execution.maxEvents) {
      throw new TopologyWorkloadError(
        `compiled plan exceeds scenario maxEvents ${this.scenario.execution.maxEvents}`,
      );
    }
    return id;
  }

  private computeDuration(
    deviceId: string,
    capability: TopologyComputeCapability,
    tokenWidth: number,
    activeExperts: number,
    expertRouted: boolean,
  ): number {
    const workItems = checkedMultiply(
      tokenWidth,
      checkedMultiply(
        this.profile.batchSize,
        Math.max(1, activeExperts),
        "active expert batch",
      ),
      "compute token batch",
    );
    const shardDegree = capability === "ffn" && expertRouted
      ? this.scenario.execution.parallelism.expert
      : capability === "attention" || capability === "ffn"
        ? this.scenario.execution.parallelism.tensor
        : 1;
    return this.computeDurationForWorkItems(
      deviceId,
      capability,
      workItems,
      shardDegree,
    );
  }

  private computeDurationForWorkItems(
    deviceId: string,
    capability: TopologyComputeCapability,
    workItems: number,
    shardDegree: number,
  ): number {
    const deviceKind = this.device(deviceId).kind;
    const costs = this.costModel.deviceCosts[deviceKind];
    const perToken = capability === "attention"
      ? costs.attentionNsPerToken
      : capability === "ffn"
        ? costs.ffnNsPerToken
        : capability === "draft"
          ? costs.draftNsPerToken
          : costs.lookupNsPerToken;
    assertPositiveSafeInteger(workItems, `${capability} work items`);
    assertPositiveSafeInteger(shardDegree, `${capability} shard degree`);
    const validRange = this.costModel.validWorkItemRanges?.[deviceKind][
      capability
    ];
    if (
      validRange !== undefined
      && (
        workItems < validRange.minWorkItems
        || workItems > validRange.maxWorkItems
      )
    ) {
      throw new TopologyWorkloadError(
        `${deviceKind} ${capability} work items ${workItems} are outside calibrated range ${validRange.minWorkItems}..${validRange.maxWorkItems}`,
      );
    }
    const unshardedDuration = checkedMultiply(
      perToken,
      workItems,
      "compute duration",
    );
    const normalizedDuration = checkedAdd(
      costs.invocationOverheadNs,
      Math.ceil(unshardedDuration / shardDegree),
      "compute invocation duration",
    );
    return Math.max(
      normalizedDuration,
      this.modelWeightBandwidthFloor(deviceId, capability, shardDegree),
    );
  }

  private modelWeightBandwidthFloor(
    deviceId: string,
    capability: TopologyComputeCapability,
    shardDegree: number,
  ): number {
    const modelWork = this.profile.modelWork;
    if (
      modelWork === undefined
      || (capability !== "attention" && capability !== "ffn")
    ) {
      return 0;
    }
    const bytes = capability === "attention"
      ? modelWork.attentionWeightBytesPerToken
      : modelWork.ffnWeightBytesPerToken;
    const placement = this.placements.find((candidate) => (
      candidate.deviceId === deviceId
      && candidate.requiredCapabilities.includes(capability)
    )) ?? this.placementForDevice(deviceId);
    const weightAllocation = placement.allocations.find(
      (allocation) => allocation.purpose === "weights",
    );
    if (weightAllocation === undefined) {
      throw new TopologyWorkloadError(
        `model-bound ${capability} placement ${placement.partitionId} has no weight allocation`,
      );
    }
    const domain = this.scenario.memoryDomains.find(
      (candidate) => candidate.id === weightAllocation.domainId,
    );
    if (domain === undefined) {
      throw new TopologyWorkloadError(
        `model-bound placement ${placement.partitionId} references unknown weight domain ${weightAllocation.domainId}`,
      );
    }
    const shardedBytes = Math.ceil(bytes / shardDegree);
    const idealDuration = checkedAdd(
      domain.latencyNs,
      scaledDuration(
        shardedBytes,
        domain.bandwidthBytesPerSec,
        "model weight stream",
      ),
      "model weight stream duration",
    );
    return Math.ceil(idealDuration / 0.7);
  }

  private requireSsdStreaming(context: string): void {
    if (!this.scenario.execution.features.ssdStreaming) {
      throw new TopologyWorkloadError(
        `${context} requires SSD streaming but scenario ${this.scenario.id} disables it`,
      );
    }
  }

  private groupForPlacements(
    placements: readonly PartitionPlacement[],
    label: string,
  ): SimulationScenario["groups"][number] {
    const groups = this.scenario.groups.filter((candidate) => (
      candidate.orderedRanks.length === placements.length
      && candidate.orderedRanks.every((rank) => (
        placements.some((placement) => placement.deviceId === rank.deviceId)
      ))
    ));
    if (groups.length !== 1) {
      throw new TopologyWorkloadError(
        `scenario ${this.scenario.id} requires exactly one communicator for ${label} placements; found ${groups.length}`,
      );
    }
    return groups[0];
  }

  private expertOwnerPlacement(
    expertId: string,
    placements: readonly PartitionPlacement[],
  ): PartitionPlacement {
    const cacheKey = `${expertId}\u0000${
      placements.map((entry) => entry.partitionId).sort().join("\u0000")
    }`;
    const cached = this.expertOwnerByKey.get(cacheKey);
    if (cached !== undefined) {
      return cached;
    }
    const expertPlacement = this.profile.expertPlacement;
    if (expertPlacement === undefined) {
      throw new TopologyWorkloadError(
        `routed expert ${expertId} requires an explicit expert placement`,
      );
    }
    const owner = resolveTopologyExpertOwnerPlacement(
      this.scenario,
      expertPlacement,
      expertId,
    );
    if (
      placements.length !== this.scenario.execution.parallelism.expert
      || !placements.some((candidate) => (
        candidate.partitionId === owner.partitionId
      ))
    ) {
      throw new TopologyWorkloadError(
        `scenario ${this.scenario.id} cannot resolve ${expertId} in the supplied FFN placements`,
      );
    }
    this.expertOwnerByKey.set(cacheKey, owner);
    return owner;
  }

  private placementForDevice(deviceId: string): PartitionPlacement {
    const placement = this.placements.find(
      (candidate) => candidate.deviceId === deviceId,
    );
    if (!placement) {
      throw new TopologyWorkloadError(
        `device ${deviceId} has no target placement`,
      );
    }
    return placement;
  }

  private hostLookupPlacement(): PartitionPlacement | undefined {
    const cpu = this.scenario.devices.find((candidate) => (
      candidate.kind === "cpu" && candidate.capabilities.includes("lookup")
    ));
    if (!cpu) {
      return undefined;
    }
    const allocation = this.scenario.placements
      .flatMap((placement) => placement.allocations)
      .find((candidate) => (
        cpu.memoryDomainIds.includes(candidate.domainId)
        && (
          candidate.purpose === "staging"
          || candidate.purpose === "workspace"
        )
      ));
    if (!allocation) {
      return undefined;
    }
    return {
      partitionId: `host-lookup:${cpu.id}`,
      deviceId: cpu.id,
      requiredCapabilities: ["lookup"],
      allocations: [{
        ...allocation,
        purpose: "workspace",
      }],
    };
  }

  private cacheSourceDomain(targetDomainId: string): string {
    const target = this.scenario.memoryDomains.find(
      (domain) => domain.id === targetDomainId,
    );
    if (!target) {
      throw new TopologyWorkloadError(`unknown target domain ${targetDomainId}`);
    }
    if (target.kind === "host" || target.kind === "unified") {
      return target.id;
    }
    const localHost = this.scenario.memoryDomains.find((domain) => (
      domain.nodeId === target.nodeId && domain.kind === "host"
    ));
    if (!localHost) {
      throw new TopologyWorkloadError(
        `device domain ${target.id} has no local host cache tier`,
      );
    }
    return localHost.id;
  }

  private collectivePhases(
    group: SimulationScenario["groups"][number],
    algorithm: CollectiveAlgorithm,
    bytes: number,
    allToAllTraffic?: readonly (readonly number[])[],
  ): CollectivePhases {
    const domains = group.orderedRanks.map((rank) => (
      workspaceDomain(this.placementForDevice(rank.deviceId))
    ));
    const phaseBytes = this.collectivePhaseBytes(
      domains.length,
      algorithm,
      bytes,
    );
    if (algorithm === COLLECTIVE_CALIBRATION_ALGORITHM) {
      const ring = domains.map((sourceDomainId, index) => ({
        linkIds: this.collectivePath(
          sourceDomainId,
          domains[(index + 1) % domains.length],
          phaseBytes,
        ),
        bytes: phaseBytes,
      }));
      return Array.from(
        {
          length: checkedMultiply(
            2,
            domains.length - 1,
            "all-reduce ring phases",
          ),
        },
        () => ring,
      );
    }
    if (algorithm === EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM) {
      const traffic = allToAllTraffic
        ?? balancedAllToAllTraffic(domains.length, bytes);
      validateAllToAllTraffic(traffic, domains.length, bytes);
      return Array.from(
        { length: domains.length - 1 },
        (_, phaseIndex) => domains.flatMap((sourceDomainId, rankIndex) => {
          const targetIndex =
            (rankIndex + phaseIndex + 1) % domains.length;
          const transferBytes = traffic[rankIndex][targetIndex];
          return transferBytes === 0
            ? []
            : [{
                linkIds: this.collectivePath(
                  sourceDomainId,
                  domains[targetIndex],
                  transferBytes,
                ),
                bytes: transferBytes,
              }];
        }),
      );
    }
    return fail(`unsupported collective algorithm ${algorithm}`);
  }

  private collectivePath(
    sourceDomainId: string,
    targetDomainId: string,
    bytes: number,
  ): string[] {
    const route = this.transferRoute(sourceDomainId, targetDomainId, bytes);
    if (!route || route.domainIds.length < 2) {
      throw new TopologyWorkloadError(
        `no collective path from ${sourceDomainId} to ${targetDomainId}`,
      );
    }
    return [...route.linkIds];
  }

  private transferRoute(
    sourceDomainId: string,
    targetDomainId: string,
    bytes: number,
  ): TransferRoute | undefined {
    const key = JSON.stringify([sourceDomainId, targetDomainId, bytes]);
    const cached = this.transferRouteByKey.get(key);
    if (cached !== undefined) {
      return cached ?? undefined;
    }
    const route = findTransferRoute(this.scenario, {
      id: "compiled-route",
      sourceDomainId,
      targetDomainId,
      bytes,
      requiresPinnedStaging: false,
      stagingAllocationIds: [],
    });
    this.transferRouteByKey.set(key, route ?? null);
    return route;
  }

  private pathDuration(linkIds: readonly string[], bytes: number): number {
    return linkIds.reduce((sum, linkId) => (
      checkedAdd(
        sum,
        linkDuration(
          this.scenario.links.find((link) => link.id === linkId)
            ?? fail(`unknown link ${linkId}`),
          bytes,
        ),
        "path duration",
      )
    ), 0);
  }

  private transportDuration(
    operation: TransportOperationKind,
    linkIds: readonly string[],
    participantCount: number,
    algorithm: string,
    bytes: number,
    collectivePhases?: CollectivePhases,
    trafficSignature?: string,
  ): number {
    if (
      algorithm === EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM
      && linkIds.length === 0
    ) {
      return 0;
    }
    const curves = this.costModel.transportCurves;
    if (curves === undefined) {
      return operation === "transfer"
        ? linkDuration(
            this.scenario.links.find((link) => link.id === linkIds[0])
              ?? fail(`unknown link ${linkIds[0]}`),
            bytes,
          )
        : this.heuristicCollectiveDuration(
            collectivePhases
              ?? [[{ linkIds, bytes }]],
            participantCount,
            algorithm,
          );
    }
    if (
      algorithm === EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM
      && trafficSignature === undefined
    ) {
      throw new TopologyWorkloadError(
        "calibrated all_to_all_v requires a traffic-signature calibration contract",
      );
    }
    const curve = curves.find((candidate) => (
      candidate.scenarioId === this.scenario.id
      && candidate.operation === operation
      && candidate.participantCount === participantCount
      && candidate.algorithm === algorithm
      && candidate.trafficSignature === trafficSignature
      && arraysEqual(candidate.linkIds, linkIds)
    ));
    const identity = [
      this.scenario.id,
      operation,
      algorithm,
      `${participantCount} participants`,
      linkIds.join("->"),
      trafficSignature ?? "no-traffic-signature",
    ].join("/");
    if (!curve) {
      throw new TopologyWorkloadError(
        `no calibrated transport curve for ${identity}`,
      );
    }
    const first = curve.points[0];
    const last = curve.points[curve.points.length - 1];
    if (bytes < first.bytes || bytes > last.bytes) {
      throw new TopologyWorkloadError(
        `${identity} bytes ${bytes} are outside calibrated range ${first.bytes}..${last.bytes}`,
      );
    }
    const upperIndex = curve.points.findIndex((point) => point.bytes >= bytes);
    const upper = curve.points[upperIndex];
    if (upper.bytes === bytes || upperIndex === 0) {
      return upper.durationNs;
    }
    const lower = curve.points[upperIndex - 1];
    const ratio = (bytes - lower.bytes) / (upper.bytes - lower.bytes);
    const durationNs = Math.ceil(
      lower.durationNs + ratio * (upper.durationNs - lower.durationNs),
    );
    if (!Number.isSafeInteger(durationNs) || durationNs <= 0) {
      throw new TopologyWorkloadError(
        `${identity} interpolation produced an unsafe duration`,
      );
    }
    return durationNs;
  }

  private heuristicCollectiveDuration(
    phases: CollectivePhases,
    participantCount: number,
    algorithm: string,
  ): number {
    const expectedPhaseCount = algorithm === COLLECTIVE_CALIBRATION_ALGORITHM
      ? checkedMultiply(
          2,
          participantCount - 1,
          "all-reduce ring phases",
        )
      : algorithm === EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM
        ? participantCount - 1
        : fail(`unsupported collective algorithm ${algorithm}`);
    if (
      participantCount < 2
      || phases.length !== expectedPhaseCount
      || phases.some((paths) => (
        (
          algorithm === COLLECTIVE_CALIBRATION_ALGORITHM
            ? paths.length !== participantCount
            : paths.length > participantCount
        )
        || paths.some((transfer) => (
          transfer.linkIds.length === 0
          || !Number.isSafeInteger(transfer.bytes)
          || transfer.bytes <= 0
        ))
      ))
    ) {
      throw new TopologyWorkloadError(
        `invalid ${participantCount}-participant ${algorithm} phase set`,
      );
    }
    let totalDurationNs = 0;
    for (const paths of phases) {
      const longestPathNs = paths.length === 0
        ? 0
        : Math.max(...paths.map((transfer) => (
            this.pathDuration(transfer.linkIds, transfer.bytes)
          )));
      const serviceByLink = new Map<string, number>();
      for (const transfer of paths) {
        for (const linkId of transfer.linkIds) {
          const link = this.scenario.links.find(
            (candidate) => candidate.id === linkId,
          ) ?? fail(`unknown link ${linkId}`);
          serviceByLink.set(
            linkId,
            checkedAdd(
              serviceByLink.get(linkId) ?? 0,
              linkDuration(link, transfer.bytes),
              `${algorithm} phase service on ${linkId}`,
            ),
          );
        }
      }
      const phaseDurationNs = Math.max(
        longestPathNs,
        ...serviceByLink.values(),
      );
      totalDurationNs = checkedAdd(
        totalDurationNs,
        phaseDurationNs,
        `${algorithm} duration`,
      );
    }
    return totalDurationNs;
  }

  private routedExpertTraffic(
    group: SimulationScenario["groups"][number],
    unit: TopologyWorkUnit,
    ffnPlacements: readonly PartitionPlacement[],
  ): readonly (readonly number[])[] {
    if (this.profile.expertTokenPlacement !== "round_robin") {
      throw new TopologyWorkloadError(
        `routed unit ${unit.id} requires round_robin token-source placement`,
      );
    }
    const rankIndexByDevice = new Map(
      group.orderedRanks.map((rank, index) => [rank.deviceId, index]),
    );
    const traffic = Array.from(
      { length: group.orderedRanks.length },
      () => Array.from({ length: group.orderedRanks.length }, () => 0),
    );
    for (const [assignmentIndex, routed] of (
      unit.routedExperts ?? []
    ).entries()) {
      const tokenIndex = Math.floor(assignmentIndex / unit.activeExperts);
      const sourceIndex = tokenIndex % group.orderedRanks.length;
      const owner = this.expertOwnerPlacement(
        routed.expertId,
        ffnPlacements,
      );
      const targetIndex = rankIndexByDevice.get(owner.deviceId);
      if (targetIndex === undefined) {
        throw new TopologyWorkloadError(
          `expert owner ${owner.deviceId} is absent from group ${group.id}`,
        );
      }
      traffic[sourceIndex][targetIndex] = checkedAdd(
        traffic[sourceIndex][targetIndex],
        this.costModel.activationBytesPerToken,
        `routed traffic ${sourceIndex}->${targetIndex}`,
      );
    }
    return traffic;
  }

  private collectivePhaseBytes(
    participantCount: number,
    algorithm: string,
    bytes: number,
  ): number {
    const phaseDivisor = algorithm === COLLECTIVE_CALIBRATION_ALGORITHM
      ? participantCount
      : algorithm === EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM
        ? checkedMultiply(
            participantCount,
            participantCount - 1,
            "all-to-all pairwise phase divisor",
          )
        : fail(`unsupported collective algorithm ${algorithm}`);
    return Math.max(1, Math.ceil(bytes / phaseDivisor));
  }

  private domainDuration(domainId: string, bytes: number): number {
    const domain = this.scenario.memoryDomains.find(
      (candidate) => candidate.id === domainId,
    );
    if (!domain) {
      throw new TopologyWorkloadError(`unknown memory domain ${domainId}`);
    }
    return checkedAdd(
      domain.latencyNs,
      scaledDuration(
        bytes,
        domain.bandwidthBytesPerSec,
        "memory transfer",
      ),
      "memory duration",
    );
  }

  private buffer(
    domainId: string,
    sourceDomainId: string,
    targetDomainId: string,
  ): string {
    const targetPlacement = this.placements.find(
      (placement) => workspaceDomain(placement) === domainId,
    );
    if (targetPlacement) {
      return workspaceId(targetPlacement);
    }
    const allocations = this.scenario.placements.flatMap(
      (placement) => placement.allocations,
    );
    const preferred = allocations.find((allocation) => (
      allocation.domainId === domainId
      && allocation.purpose === "staging"
    )) ?? allocations.find((allocation) => (
      allocation.domainId === domainId
      && allocation.purpose === "workspace"
    )) ?? allocations.find((allocation) => allocation.domainId === domainId);
    if (!preferred) {
      throw new TopologyWorkloadError(
        `transfer ${sourceDomainId}->${targetDomainId} lacks a buffer in ${domainId}`,
      );
    }
    return preferred.physicalAllocationId;
  }

  private rank(deviceId: string): string {
    const rank = this.rankByDevice.get(deviceId);
    if (rank === undefined) {
      throw new TopologyWorkloadError(`device ${deviceId} has no rank identity`);
    }
    return rank;
  }

  private device(deviceId: string): SimDeviceSpec {
    const device = this.scenario.devices.find(
      (candidate) => candidate.id === deviceId,
    );
    if (!device) {
      throw new TopologyWorkloadError(`unknown device ${deviceId}`);
    }
    return device;
  }
}

function balancedAllToAllTraffic(
  participantCount: number,
  bytes: number,
): readonly (readonly number[])[] {
  const pairCount = checkedMultiply(
    participantCount,
    participantCount - 1,
    "all-to-all pair count",
  );
  const baseBytes = Math.floor(bytes / pairCount);
  let remainder = bytes % pairCount;
  return Array.from({ length: participantCount }, (_, sourceIndex) => (
    Array.from({ length: participantCount }, (_, targetIndex) => {
      if (sourceIndex === targetIndex) {
        return 0;
      }
      const pairBytes = baseBytes + (remainder > 0 ? 1 : 0);
      remainder = Math.max(0, remainder - 1);
      return pairBytes;
    })
  ));
}

export function canonicalAllToAllTrafficSignature(
  traffic: readonly (readonly number[])[],
): string {
  if (
    traffic.length < 2
    || traffic.some((row) => row.length !== traffic.length)
  ) {
    throw new TopologyWorkloadError(
      "all-to-all traffic signature requires a square matrix with at least two ranks",
    );
  }
  const values = traffic.flat();
  if (values.some((bytes) => (
    !Number.isSafeInteger(bytes) || bytes < 0
  ))) {
    throw new TopologyWorkloadError(
      "all-to-all traffic signature requires non-negative safe integers",
    );
  }
  const positive = values.filter((bytes) => bytes > 0);
  if (positive.length === 0) {
    throw new TopologyWorkloadError(
      "all-to-all traffic signature requires at least one positive cell",
    );
  }
  const divisor = positive.reduce(greatestCommonDivisor);
  const reduced = traffic.map((row) => (
    row.map((bytes) => bytes / divisor)
  ));
  return `${ALL_TO_ALL_TRAFFIC_SIGNATURE_PREFIX}${JSON.stringify(reduced)}`;
}

export function assertCanonicalAllToAllTrafficSignature(
  signature: string,
  participantCount: number,
): void {
  if (!signature.startsWith(ALL_TO_ALL_TRAFFIC_SIGNATURE_PREFIX)) {
    throw new TopologyWorkloadError(
      `invalid AllToAllV traffic signature ${JSON.stringify(signature)}`,
    );
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(signature.slice(
      ALL_TO_ALL_TRAFFIC_SIGNATURE_PREFIX.length,
    ));
  } catch {
    throw new TopologyWorkloadError(
      `invalid AllToAllV traffic signature ${JSON.stringify(signature)}`,
    );
  }
  if (
    !Array.isArray(parsed)
    || parsed.length !== participantCount
    || parsed.some((row) => (
      !Array.isArray(row) || row.length !== participantCount
    ))
  ) {
    throw new TopologyWorkloadError(
      `AllToAllV traffic signature must be ${participantCount}x${participantCount}`,
    );
  }
  const canonical = canonicalAllToAllTrafficSignature(
    parsed as readonly (readonly number[])[],
  );
  if (canonical !== signature) {
    throw new TopologyWorkloadError(
      "AllToAllV traffic signature is not canonical",
    );
  }
}

function greatestCommonDivisor(left: number, right: number): number {
  let dividend = left;
  let divisor = right;
  while (divisor !== 0) {
    const remainder = dividend % divisor;
    dividend = divisor;
    divisor = remainder;
  }
  return dividend;
}

function transposeTraffic(
  traffic: readonly (readonly number[])[],
): readonly (readonly number[])[] {
  return traffic.map((row, sourceIndex) => (
    row.map((_, targetIndex) => traffic[targetIndex][sourceIndex])
  ));
}

function validateAllToAllTraffic(
  traffic: readonly (readonly number[])[],
  participantCount: number,
  expectedBytes: number,
): void {
  if (
    traffic.length !== participantCount
    || traffic.some((row) => row.length !== participantCount)
  ) {
    throw new TopologyWorkloadError(
      `all-to-all traffic must be ${participantCount}x${participantCount}`,
    );
  }
  let totalBytes = 0;
  for (const [sourceIndex, row] of traffic.entries()) {
    for (const [targetIndex, bytes] of row.entries()) {
      if (!Number.isSafeInteger(bytes) || bytes < 0) {
        throw new TopologyWorkloadError(
          `all-to-all traffic ${sourceIndex}->${targetIndex} has invalid bytes ${bytes}`,
        );
      }
      totalBytes = checkedAdd(
        totalBytes,
        bytes,
        "all-to-all traffic bytes",
      );
    }
  }
  if (totalBytes !== expectedBytes) {
    throw new TopologyWorkloadError(
      `all-to-all traffic accounts for ${totalBytes}/${expectedBytes} bytes`,
    );
  }
}

function validateInputs(
  scenario: SimulationScenario,
  profile: TopologyWorkloadProfile,
  costModel: TopologyCostModel,
): void {
  if (profile.id.length === 0) {
    throw new TopologyWorkloadError("profile id must be non-empty");
  }
  assertPositiveSafeInteger(profile.batchSize, "profile batch size");
  if (profile.modelWork !== undefined) {
    if (
      profile.modelWork.modelId.length === 0
      || profile.modelWork.modelName.length === 0
    ) {
      throw new TopologyWorkloadError(
        "model work id and name must be non-empty",
      );
    }
    assertPositiveSafeInteger(
      profile.modelWork.attentionWeightBytesPerToken,
      "model attention weight bytes",
    );
    assertPositiveSafeInteger(
      profile.modelWork.ffnWeightBytesPerToken,
      "model FFN weight bytes",
    );
    assertPositiveSafeInteger(
      profile.modelWork.forwardFlopsPerToken,
      "model forward FLOPs per token",
    );
  }
  if (profile.units.length === 0) {
    throw new TopologyWorkloadError("profile must contain at least one work unit");
  }
  if (profile.expertPlacement !== undefined) {
    validateExpertPlacement(
      profile.expertPlacement,
      `profile ${profile.id}`,
    );
  }
  if (
    profile.expertTokenPlacement !== undefined
    && profile.expertTokenPlacement !== "round_robin"
  ) {
    throw new TopologyWorkloadError(
      `profile ${profile.id} has unsupported expert token placement ${String(profile.expertTokenPlacement)}`,
    );
  }
  const prefetchIds = new Set<string>();
  for (const prefetch of profile.backgroundPrefetches ?? []) {
    if (prefetch.id.length === 0 || prefetchIds.has(prefetch.id)) {
      throw new TopologyWorkloadError(
        `background prefetch id ${JSON.stringify(prefetch.id)} must be non-empty and unique`,
      );
    }
    prefetchIds.add(prefetch.id);
    if (profile.expertPlacement !== undefined) {
      if (
        prefetch.expertId === undefined
        || !profile.expertPlacement.expertIds.includes(prefetch.expertId)
      ) {
        throw new TopologyWorkloadError(
          `background prefetch ${prefetch.id} requires a known expert id`,
        );
      }
    } else if (prefetch.expertId !== undefined) {
      throw new TopologyWorkloadError(
        `background prefetch ${prefetch.id} cannot name an expert without a placement`,
      );
    }
    if (
      !Number.isSafeInteger(prefetch.afterUnitIndex)
      || prefetch.afterUnitIndex < -1
      || prefetch.afterUnitIndex >= profile.units.length
    ) {
      throw new TopologyWorkloadError(
        `background prefetch ${prefetch.id} has invalid trigger unit ${prefetch.afterUnitIndex}`,
      );
    }
    assertPositiveSafeInteger(
      prefetch.bytes,
      `background prefetch ${prefetch.id} bytes`,
    );
  }
  for (const unit of profile.units) {
    if (
      unit.expertRouted !== undefined
      && typeof unit.expertRouted !== "boolean"
    ) {
      throw new TopologyWorkloadError(
        `unit ${unit.id} expertRouted must be boolean`,
      );
    }
    const required = unit.requiredPrefetchIds ?? [];
    if (new Set(required).size !== required.length) {
      throw new TopologyWorkloadError(
        `unit ${unit.id} has duplicate required background prefetch ids`,
      );
    }
    for (const prefetchId of required) {
      if (!prefetchIds.has(prefetchId)) {
        throw new TopologyWorkloadError(
          `unit ${unit.id} requires unknown background prefetch ${prefetchId}`,
        );
      }
    }
  }
  if (costModel.revision !== TOPOLOGY_COST_MODEL_REVISION) {
    throw new TopologyWorkloadError(
      `unsupported topology cost revision ${costModel.revision}`,
    );
  }
  assertPositiveSafeInteger(
    costModel.activationBytesPerToken,
    "activation bytes per token",
  );
  assertPositiveSafeInteger(
    costModel.collectiveBytesPerToken,
    "collective bytes per token",
  );
  assertPositiveSafeInteger(
    costModel.coldLoadByteMultiplier,
    "cold load multiplier",
  );
  if (costModel.source.length === 0) {
    throw new TopologyWorkloadError("cost model source must be non-empty");
  }
  if (costModel.applicability !== undefined) {
    if (!costModel.applicability.scenarioIds.includes(scenario.id)) {
      throw new TopologyWorkloadError(
        `cost model is not applicable to scenario ${scenario.id}`,
      );
    }
    if (
      new Set(costModel.applicability.scenarioIds).size
      !== costModel.applicability.scenarioIds.length
    ) {
      throw new TopologyWorkloadError(
        "cost model applicability scenario ids must be unique",
      );
    }
    for (const kind of ["cpu", "gpu", "npu"] as const) {
      if (costModel.applicability.deviceKindLabels[kind]?.trim().length === 0) {
        throw new TopologyWorkloadError(
          `cost model ${kind} applicability label must be non-empty`,
        );
      }
    }
  }
  if (costModel.transportCurves !== undefined) {
    const identities = new Set<string>();
    for (const curve of costModel.transportCurves) {
      const identity = transportCurveIdentity(curve);
      if (identities.has(identity)) {
        throw new TopologyWorkloadError(
          `duplicate transport calibration curve ${identity}`,
        );
      }
      identities.add(identity);
      if (
        curve.operation !== "transfer"
        && curve.operation !== "collective"
      ) {
        throw new TopologyWorkloadError(
          `transport curve ${identity} has an unsupported operation`,
        );
      }
      if (
        costModel.applicability !== undefined
        && !costModel.applicability.scenarioIds.includes(curve.scenarioId)
      ) {
        throw new TopologyWorkloadError(
          `transport curve scenario ${curve.scenarioId} is outside cost model applicability`,
        );
      }
      if (curve.linkIds.length === 0) {
        throw new TopologyWorkloadError(
          `transport curve ${identity} must name at least one link`,
        );
      }
      if (curve.operation === "transfer" && curve.linkIds.length !== 1) {
        throw new TopologyWorkloadError(
          `transfer curve ${identity} must name exactly one link`,
        );
      }
      assertPositiveSafeInteger(
        curve.participantCount,
        `transport curve ${identity} participant count`,
      );
      if (curve.operation === "transfer" && curve.participantCount !== 2) {
        throw new TopologyWorkloadError(
          `transfer curve ${identity} must have two participants`,
        );
      }
      if (curve.operation === "collective" && curve.participantCount < 2) {
        throw new TopologyWorkloadError(
          `collective curve ${identity} must have at least two participants`,
        );
      }
      if (curve.algorithm.trim().length === 0) {
        throw new TopologyWorkloadError(
          `transport curve ${identity} algorithm must be non-empty`,
        );
      }
      if (
        curve.operation === "collective"
        && curve.algorithm === EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM
      ) {
        if (curve.trafficSignature === undefined) {
          throw new TopologyWorkloadError(
            `transport curve ${identity} requires an AllToAllV traffic signature`,
          );
        }
        assertCanonicalAllToAllTrafficSignature(
          curve.trafficSignature,
          curve.participantCount,
        );
      } else if (curve.trafficSignature !== undefined) {
        throw new TopologyWorkloadError(
          `transport curve ${identity} cannot declare a traffic signature`,
        );
      }
      if (curve.points.length < 2) {
        throw new TopologyWorkloadError(
          `transport curve ${identity} requires at least two points`,
        );
      }
      let previousBytes = 0;
      let previousDuration = 0;
      for (const point of curve.points) {
        assertPositiveSafeInteger(
          point.bytes,
          `transport curve ${identity} bytes`,
        );
        assertPositiveSafeInteger(
          point.durationNs,
          `transport curve ${identity} duration`,
        );
        if (point.bytes <= previousBytes) {
          throw new TopologyWorkloadError(
            `transport curve ${identity} byte points must be strictly increasing`,
          );
        }
        if (point.durationNs < previousDuration) {
          throw new TopologyWorkloadError(
            `transport curve ${identity} duration must be non-decreasing`,
          );
        }
        previousBytes = point.bytes;
        previousDuration = point.durationNs;
      }
    }
  }
  for (const kind of ["cpu", "gpu", "npu"] as const) {
    const costs = costModel.deviceCosts[kind];
    assertPositiveSafeInteger(
      costs.invocationOverheadNs,
      `${kind} invocation overhead`,
    );
    assertPositiveSafeInteger(
      costs.attentionNsPerToken,
      `${kind} attention cost`,
    );
    assertPositiveSafeInteger(costs.ffnNsPerToken, `${kind} ffn cost`);
    assertPositiveSafeInteger(costs.draftNsPerToken, `${kind} draft cost`);
    assertPositiveSafeInteger(costs.lookupNsPerToken, `${kind} lookup cost`);
    const ranges = costModel.validWorkItemRanges?.[kind];
    if (ranges !== undefined) {
      for (
        const capability of [
          "attention",
          "ffn",
          "draft",
          "lookup",
        ] as const
      ) {
        const range = ranges[capability];
        assertPositiveSafeInteger(
          range.minWorkItems,
          `${kind} ${capability} minimum work items`,
        );
        assertPositiveSafeInteger(
          range.maxWorkItems,
          `${kind} ${capability} maximum work items`,
        );
        if (range.minWorkItems > range.maxWorkItems) {
          throw new TopologyWorkloadError(
            `${kind} ${capability} calibrated range is inverted`,
          );
        }
      }
    }
  }
  for (const unit of profile.units) {
    if (unit.id.length === 0) {
      throw new TopologyWorkloadError("work unit id must be non-empty");
    }
    assertPositiveSafeInteger(unit.targetTokenWidth, `${unit.id} token width`);
    assertNonNegativeSafeInteger(unit.committedTokens, `${unit.id} committed tokens`);
    assertNonNegativeSafeInteger(unit.draftTokens, `${unit.id} draft tokens`);
    if (
      unit.proposerCostScale !== undefined
      && (
        !Number.isFinite(unit.proposerCostScale)
        || unit.proposerCostScale <= 0
      )
    ) {
      throw new TopologyWorkloadError(
        `${unit.id} proposer cost scale must be positive`,
      );
    }
    assertPositiveSafeInteger(unit.activeExperts, `${unit.id} active experts`);
    if (unit.expertRouted === true) {
      if (
        profile.expertPlacement === undefined
        || profile.expertTokenPlacement === undefined
        || unit.routedExperts === undefined
      ) {
        throw new TopologyWorkloadError(
          `routed unit ${unit.id} requires explicit expert/token placement and routed experts`,
        );
      }
      const expectedRoutes = checkedMultiply(
        checkedMultiply(
          unit.targetTokenWidth,
          profile.batchSize,
          `${unit.id} routed token batch`,
        ),
        unit.activeExperts,
        `${unit.id} routed expert assignments`,
      );
      if (unit.routedExperts.length !== expectedRoutes) {
        throw new TopologyWorkloadError(
          `routed unit ${unit.id} has ${unit.routedExperts.length}/${expectedRoutes} expert assignments`,
        );
      }
      if (unit.activeExperts > profile.expertPlacement.expertIds.length) {
        throw new TopologyWorkloadError(
          `routed unit ${unit.id} active expert count exceeds its placement universe`,
        );
      }
      for (const [routeIndex, routed] of unit.routedExperts.entries()) {
        if (
          routed.expertId.length === 0
          || !profile.expertPlacement.expertIds.includes(routed.expertId)
        ) {
          throw new TopologyWorkloadError(
            `routed unit ${unit.id} has unknown expert ${routed.expertId}`,
          );
        }
        if (
          routed.sourceTier !== "hot"
          && routed.sourceTier !== "warm"
          && routed.sourceTier !== "cold"
        ) {
          throw new TopologyWorkloadError(
            `routed unit ${unit.id} has invalid source tier ${String(routed.sourceTier)}`,
          );
        }
        assertNonNegativeSafeInteger(
          routed.loadBytes,
          `${unit.id} routed expert bytes`,
        );
        if (routed.sourceTier === "hot" && routed.loadBytes !== 0) {
          throw new TopologyWorkloadError(
            `routed unit ${unit.id} hot expert ${routed.expertId} cannot load bytes`,
          );
        }
        if (routed.sourceTier !== "hot" && routed.loadBytes === 0) {
          throw new TopologyWorkloadError(
            `routed unit ${unit.id} ${routed.sourceTier} expert ${routed.expertId} requires positive load bytes`,
          );
        }
        if ((routeIndex + 1) % unit.activeExperts === 0) {
          const tokenRoutes = unit.routedExperts.slice(
            routeIndex + 1 - unit.activeExperts,
            routeIndex + 1,
          );
          if (
            new Set(tokenRoutes.map((entry) => entry.expertId)).size
            !== tokenRoutes.length
          ) {
            throw new TopologyWorkloadError(
              `routed unit ${unit.id} assigns one token to the same expert more than once`,
            );
          }
        }
      }
    } else if (unit.routedExperts !== undefined) {
      throw new TopologyWorkloadError(
        `non-routed unit ${unit.id} cannot declare routed experts`,
      );
    }
    assertNonNegativeSafeInteger(unit.warmLoadBytes, `${unit.id} warm bytes`);
    assertNonNegativeSafeInteger(unit.coldLoadBytes, `${unit.id} cold bytes`);
  }
  if (new Set(profile.units.map((unit) => unit.id)).size !== profile.units.length) {
    throw new TopologyWorkloadError("work unit ids must be unique");
  }
  if (scenario.execution.maxEvents <= 0) {
    throw new TopologyWorkloadError("scenario maxEvents must be positive");
  }
}

function validateExpertPlacement(
  placement: TopologyExpertPlacement,
  label: string,
): void {
  if (
    placement.strategy !== "contiguous"
    && placement.strategy !== "round_robin"
  ) {
    throw new TopologyWorkloadError(
      `${label} has invalid expert placement strategy ${String(placement.strategy)}`,
    );
  }
  if (
    placement.expertIds.length === 0
    || placement.expertIds.some((expertId) => expertId.length === 0)
    || new Set(placement.expertIds).size !== placement.expertIds.length
  ) {
    throw new TopologyWorkloadError(
      `${label} expert placement ids must be non-empty and unique`,
    );
  }
}

function cloneExpertLoad(load: TopologyExpertLoad): TopologyExpertLoad {
  return {
    ...load,
    placement: {
      strategy: load.placement.strategy,
      expertIds: [...load.placement.expertIds],
    },
  };
}

function workspaceId(placement: PartitionPlacement): string {
  const allocation = placement.allocations.find(
    (candidate) => candidate.purpose === "workspace",
  );
  if (!allocation) {
    throw new TopologyWorkloadError(
      `placement ${placement.partitionId} lacks activation workspace`,
    );
  }
  return allocation.physicalAllocationId;
}

function workspaceDomain(placement: PartitionPlacement): string {
  const id = workspaceId(placement);
  const allocation = placement.allocations.find(
    (candidate) => candidate.physicalAllocationId === id,
  );
  if (!allocation) {
    throw new TopologyWorkloadError(`unknown workspace allocation ${id}`);
  }
  return allocation.domainId;
}

function expertHotCacheId(placement: PartitionPlacement): string {
  const workspaceDomainId = workspaceDomain(placement);
  const allocations = placement.allocations.filter((candidate) => (
    candidate.purpose === "cache"
    && candidate.domainId === workspaceDomainId
  ));
  if (allocations.length !== 1) {
    throw new TopologyWorkloadError(
      `FFN placement ${placement.partitionId} requires exactly one hot expert cache in ${workspaceDomainId}; found ${allocations.length}`,
    );
  }
  return allocations[0].physicalAllocationId;
}

function linkDuration(link: SimLinkSpec, bytes: number): number {
  return checkedAdd(
    link.latencyNs,
    scaledDuration(
      bytes,
      link.bandwidthBytesPerSec,
      "link transfer",
    ),
    "link duration",
  );
}

function transportCurveIdentity(curve: TransportCalibrationCurve): string {
  return JSON.stringify({
    scenarioId: curve.scenarioId,
    operation: curve.operation,
    algorithm: curve.algorithm,
    participantCount: curve.participantCount,
    linkIds: curve.linkIds,
    trafficSignature: curve.trafficSignature,
  });
}

function arraysEqual(
  left: readonly string[],
  right: readonly string[],
): boolean {
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function serviceTime(
  execution: FrozenPlanExecutionResult,
  kind: "compute" | "transfer" | "collective",
): number {
  return execution.trace.operations
    .filter((event) => event.kind === kind)
    .reduce((sum, event) => (
      checkedAdd(sum, event.finishNs - event.startNs, `${kind} service time`)
    ), 0);
}

function resourceUtilization(
  scenario: SimulationScenario,
  execution: FrozenPlanExecutionResult,
  prefix: "compute:" | "link:",
  totalDurationNs: number,
): TopologyResourceUtilization[] {
  const resources = prefix === "compute:"
    ? scenario.devices.map((device) => ({
        id: `compute:${device.id}`,
        lanes: device.maxConcurrentCompute,
      }))
    : scenario.links.map((link) => ({
        id: `link:${link.id}`,
        lanes: link.concurrencyLanes,
      }));
  return resources.map((resource) => {
    const busyNs = execution.trace.operations.reduce((sum, event) => (
      event.resources.some((reservation) => (
        reservation.resourceId === resource.id
      ))
        ? sum + event.finishNs - event.startNs
        : sum
    ), 0);
    return {
      resourceId: resource.id,
      busyNs,
      capacityLanes: resource.lanes,
      utilization: totalDurationNs === 0
        ? 0
        : busyNs / (totalDurationNs * resource.lanes),
    };
  }).filter((resource) => resource.busyNs > 0)
    .sort((left, right) => left.resourceId.localeCompare(right.resourceId));
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function uniqueNumbers(values: readonly number[]): number[] {
  return [...new Set(values)];
}

function prefetchDomainKey(prefetchId: string, domainId: string): string {
  return `${prefetchId}\0${domainId}`;
}

function scaledDuration(
  bytes: number,
  bandwidthBytesPerSec: number,
  label: string,
): number {
  assertNonNegativeSafeInteger(bytes, `${label} bytes`);
  assertPositiveSafeInteger(
    bandwidthBytesPerSec,
    `${label} bandwidth`,
  );
  const duration = Math.ceil(bytes / bandwidthBytesPerSec * 1_000_000_000);
  if (!Number.isSafeInteger(duration)) {
    throw new TopologyWorkloadError(
      `${label} duration exceeds Number.MAX_SAFE_INTEGER`,
    );
  }
  return duration;
}

function checkedAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    throw new TopologyWorkloadError(`${label} exceeds Number.MAX_SAFE_INTEGER`);
  }
  return result;
}

function checkedMultiply(left: number, right: number, label: string): number {
  const result = left * right;
  if (!Number.isSafeInteger(result)) {
    throw new TopologyWorkloadError(`${label} exceeds Number.MAX_SAFE_INTEGER`);
  }
  return result;
}

function assertNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new TopologyWorkloadError(
      `${label} must be a non-negative safe integer; got ${value}`,
    );
  }
}

function assertPositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new TopologyWorkloadError(
      `${label} must be a positive safe integer; got ${value}`,
    );
  }
}

function fail(message: string): never {
  throw new TopologyWorkloadError(message);
}
