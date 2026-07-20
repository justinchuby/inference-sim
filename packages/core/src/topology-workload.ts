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
  findTransferPath,
} from "./scenario.js";
import type {
  ComputeCapability,
  ConfidenceClass,
  PartitionPlacement,
  SimDeviceKind,
  SimDeviceSpec,
  SimLinkSpec,
  SimulationScenario,
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

export const TOPOLOGY_COST_MODEL_REVISION = 5;
export const TRANSFER_CALIBRATION_ALGORITHM = "point_to_point";
export const COLLECTIVE_CALIBRATION_ALGORITHM = "all_reduce_ring";
export const EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM = "all_to_all_v";

export interface TopologyWorkUnit {
  readonly id: string;
  readonly targetTokenWidth: number;
  readonly committedTokens: number;
  readonly draftTokens: number;
  readonly proposerExecution?: SpeculativeProposerExecution;
  readonly proposerCostScale?: number;
  readonly activeExperts: number;
  readonly expertRouted?: boolean;
  readonly warmLoadBytes: number;
  readonly coldLoadBytes: number;
  readonly requiredPrefetchIds?: readonly string[];
}

export interface TopologyWorkloadProfile {
  readonly id: string;
  readonly batchSize: number;
  readonly units: readonly TopologyWorkUnit[];
  readonly backgroundPrefetches?: readonly TopologyBackgroundPrefetch[];
}

export interface TopologyBackgroundPrefetch {
  readonly id: string;
  readonly afterUnitIndex: number;
  readonly bytes: number;
}

export interface TopologyExpertLoad {
  readonly id: string;
  readonly sourceTier: "warm" | "cold";
  readonly bytes: number;
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
): TopologyWorkloadProfile {
  return {
    id: `speculative:${result.family}`,
    batchSize: 1,
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
  expertBytes: number,
): TopologyWorkloadProfile {
  assertPositiveSafeInteger(expertBytes, "expert bytes");
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
    units: result.routes.map((route, index) => ({
      id: `route-${index}`,
      targetTokenWidth: 1,
      committedTokens: 1,
      draftTokens: 0,
      activeExperts: route.expertIds.length,
      expertRouted: true,
      warmLoadBytes: countTier(route.sourceTiers, "warm") * expertBytes,
      coldLoadBytes: countTier(route.sourceTiers, "cold") * expertBytes,
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
  if (load.sourceTier !== "warm" && load.sourceTier !== "cold") {
    throw new TopologyWorkloadError(
      `expert load ${load.id} has invalid source tier ${String(load.sourceTier)}`,
    );
  }
  assertPositiveSafeInteger(load.bytes, `expert load ${load.id} bytes`);
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
    },
    costModel,
  );
  return compiler.compileExpertLoad(load);
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
  const confidence = topologyPerformanceConfidence(scenario, costModel);
  return {
    scenarioId: scenario.id,
    profileId: profile.id,
    confidence,
    assumptions: [
      costModel.source,
      confidence === costModel.confidence
        ? `overall timing confidence is ${confidence}`
        : `overall timing confidence is ${confidence} because scenario performance evidence is weaker than the cost model`,
      "decode-only plan; prefill and request batching are outside this profile",
      "compute costs include one device-kind invocation overhead plus linear token work",
      "family-specific proposer multipliers are heuristic and provenance-labeled",
      costModel.transportCurves === undefined
        ? "transport timing uses declared directed-link bandwidth and latency"
        : "transport timing uses exact-path calibration curves without extrapolation",
      "expert-load bytes are evenly sharded across FFN placements",
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
    ...(costModel.transportCurves === undefined
      ? scenario.links.map((link) => link.provenance.confidence)
      : []),
  ];
  if (evidence.includes("heuristic")) {
    return "heuristic";
  }
  // Timing remains an estimate even when the structural inputs are exact or bounded.
  return "calibrated";
}

export function compareTopologyWorkloads(
  scenarios: readonly SimulationScenario[],
  profile: TopologyWorkloadProfile,
  costModel: TopologyCostModel = DEFAULT_TOPOLOGY_COST_MODEL,
): readonly TopologyComparisonEntry[] {
  if (scenarios.length === 0) {
    throw new TopologyWorkloadError("at least one scenario is required");
  }
  const results = scenarios.map((scenario) => (
    simulateTopologyWorkload(scenario, profile, costModel)
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
    const placementsByTarget = new Map<string, PartitionPlacement[]>();
    for (const placement of ffnPlacements) {
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
        ) / ffnPlacements.length,
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
        load: { ...load },
        terminalStepIds: [],
      };
    }
    return {
      load: { ...load },
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
    const targets = unique(ffnPlacements.map((placement) => (
      this.cacheSourceDomain(workspaceDomain(placement))
    )));
    for (const prefetch of prefetches) {
      for (const targetDomainId of targets) {
        const localPlacementCount = ffnPlacements.filter((placement) => (
          this.cacheSourceDomain(workspaceDomain(placement)) === targetDomainId
        )).length;
        const shardBytes = Math.ceil(
          checkedMultiply(
            prefetch.bytes,
            localPlacementCount,
            "background prefetch node bytes",
          )
          / Math.max(1, ffnPlacements.length),
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
        const placement = ffnPlacements.find((candidate) => (
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
    const dispatched = this.addCollective(
      group,
      EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM,
      expertBytes,
      [attentionReduced],
      attentionPlacements.map((placement) => workspaceId(placement)),
    );
    const expertTerminals = ffnPlacements.map((placement) => {
      const cacheLoad = this.prepareCacheLoad(
        placement,
        unit,
        [dispatched],
        expertDegree,
      );
      const durationNs = checkedAdd(
        this.computeDuration(
          placement.deviceId,
          "ffn",
          unit.targetTokenWidth,
          unit.activeExperts,
          true,
        ),
        cacheLoad.localMemoryPenaltyNs,
        "expert FFN duration",
      );
      return this.addCompute(
        placement,
        "ffn",
        durationNs,
        cacheLoad.dependencies,
        [expertHotCacheId(placement)],
      );
    });
    return this.addCollective(
      group,
      EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM,
      expertBytes,
      expertTerminals,
      ffnPlacements.map((placement) => workspaceId(placement)),
    );
  }

  private addCollective(
    group: SimulationScenario["groups"][number],
    algorithm: CollectiveAlgorithm,
    bytes: number,
    dependencies: readonly number[],
    reads: readonly string[],
  ): number {
    if (group.orderedRanks.length !== 2) {
      throw new TopologyWorkloadError(
        `collective path compilation currently requires exactly two ranks; ${group.id} has ${group.orderedRanks.length}`,
      );
    }
    const first = this.placementForDevice(group.orderedRanks[0].deviceId);
    const last = this.placementForDevice(
      group.orderedRanks[group.orderedRanks.length - 1].deviceId,
    );
    const linkIds = this.collectiveLinks(
      workspaceDomain(first),
      workspaceDomain(last),
    );
    const durationNs = this.transportDuration(
      "collective",
      linkIds,
      group.orderedRanks.length,
      algorithm,
      bytes,
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
    const prefetchDependencies = (unit.requiredPrefetchIds ?? []).map(
      (prefetchId) => {
        const terminal = this.backgroundPrefetchTerminalByIdAndDomain.get(
          prefetchDomainKey(prefetchId, sourceDomain),
        );
        if (terminal === undefined) {
          throw new TopologyWorkloadError(
            `unit ${unit.id} requires unresolved background prefetch ${prefetchId} for ${sourceDomain}`,
          );
        }
        return terminal;
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
    const domains = findTransferPath(this.scenario, {
      id: "compiled-transfer",
      sourceDomainId,
      targetDomainId,
      bytes,
      requiresPinnedStaging: false,
      stagingAllocationIds: [],
    });
    if (!domains) {
      throw new TopologyWorkloadError(
        `no transfer path from ${sourceDomainId} to ${targetDomainId}`,
      );
    }
    if (domains.length <= 1) {
      return [];
    }
    const stepIds: number[] = [];
    let currentDependencies = [...dependencies];
    for (let index = 0; index < domains.length - 1; index++) {
      const source = domains[index];
      const target = domains[index + 1];
      const link = this.link(source, target);
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
    this.steps.push({ id, ...input });
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
    const deviceKind = this.device(deviceId).kind;
    const costs = this.costModel.deviceCosts[deviceKind];
    const perToken = capability === "attention"
      ? costs.attentionNsPerToken
      : capability === "ffn"
        ? costs.ffnNsPerToken
        : capability === "draft"
          ? costs.draftNsPerToken
          : costs.lookupNsPerToken;
    const workItems = checkedMultiply(
      tokenWidth,
      checkedMultiply(
        this.profile.batchSize,
        Math.max(1, activeExperts),
        "active expert batch",
      ),
      "compute token batch",
    );
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
    const shardDegree = capability === "ffn" && expertRouted
      ? this.scenario.execution.parallelism.expert
      : capability === "attention" || capability === "ffn"
        ? this.scenario.execution.parallelism.tensor
        : 1;
    return checkedAdd(
      costs.invocationOverheadNs,
      Math.ceil(unshardedDuration / shardDegree),
      "compute invocation duration",
    );
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

  private collectiveLinks(sourceDomainId: string, targetDomainId: string): string[] {
    const path = (source: string, target: string): string[] => {
      const domains = findTransferPath(this.scenario, {
        id: "compiled-collective",
        sourceDomainId: source,
        targetDomainId: target,
        bytes: 1,
        requiresPinnedStaging: false,
        stagingAllocationIds: [],
      });
      if (!domains || domains.length < 2) {
        throw new TopologyWorkloadError(
          `no collective path from ${source} to ${target}`,
        );
      }
      return domains.slice(0, -1).map((domain, index) => (
        this.link(domain, domains[index + 1]).id
      ));
    };
    return unique([
      ...path(sourceDomainId, targetDomainId),
      ...path(targetDomainId, sourceDomainId),
    ]);
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
  ): number {
    const curves = this.costModel.transportCurves;
    if (curves === undefined) {
      return operation === "transfer"
        ? linkDuration(
            this.scenario.links.find((link) => link.id === linkIds[0])
              ?? fail(`unknown link ${linkIds[0]}`),
            bytes,
          )
        : this.pathDuration(linkIds, bytes);
    }
    const curve = curves.find((candidate) => (
      candidate.scenarioId === this.scenario.id
      && candidate.operation === operation
      && candidate.participantCount === participantCount
      && candidate.algorithm === algorithm
      && arraysEqual(candidate.linkIds, linkIds)
    ));
    const identity = [
      this.scenario.id,
      operation,
      algorithm,
      `${participantCount} participants`,
      linkIds.join("->"),
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

  private link(sourceDomainId: string, targetDomainId: string): SimLinkSpec {
    const link = this.scenario.links
      .filter((candidate) => (
        candidate.sourceDomainId === sourceDomainId
        && candidate.targetDomainId === targetDomainId
      ))
      .sort((left, right) => left.id.localeCompare(right.id))[0];
    if (!link) {
      throw new TopologyWorkloadError(
        `missing directed link ${sourceDomainId}->${targetDomainId}`,
      );
    }
    return link;
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

function validateInputs(
  scenario: SimulationScenario,
  profile: TopologyWorkloadProfile,
  costModel: TopologyCostModel,
): void {
  if (profile.id.length === 0) {
    throw new TopologyWorkloadError("profile id must be non-empty");
  }
  assertPositiveSafeInteger(profile.batchSize, "profile batch size");
  if (profile.units.length === 0) {
    throw new TopologyWorkloadError("profile must contain at least one work unit");
  }
  const prefetchIds = new Set<string>();
  for (const prefetch of profile.backgroundPrefetches ?? []) {
    if (prefetch.id.length === 0 || prefetchIds.has(prefetch.id)) {
      throw new TopologyWorkloadError(
        `background prefetch id ${JSON.stringify(prefetch.id)} must be non-empty and unique`,
      );
    }
    prefetchIds.add(prefetch.id);
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

function countTier(
  tiers: readonly ExpertCacheTier[],
  target: ExpertCacheTier,
): number {
  return tiers.filter((tier) => tier === target).length;
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
