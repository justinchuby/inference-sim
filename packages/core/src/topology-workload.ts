import {
  PLAN_CONTRACT_REVISION,
  type FrozenPlan,
  type FrozenPlanExecutionResult,
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

export const TOPOLOGY_COST_MODEL_REVISION = 3;

export interface TopologyWorkUnit {
  readonly id: string;
  readonly targetTokenWidth: number;
  readonly committedTokens: number;
  readonly draftTokens: number;
  readonly proposerExecution?: SpeculativeProposerExecution;
  readonly proposerCostScale?: number;
  readonly activeExperts: number;
  readonly warmLoadBytes: number;
  readonly coldLoadBytes: number;
}

export interface TopologyWorkloadProfile {
  readonly id: string;
  readonly batchSize: number;
  readonly units: readonly TopologyWorkUnit[];
}

export interface DeviceCapabilityCost {
  readonly invocationOverheadNs: number;
  readonly attentionNsPerToken: number;
  readonly ffnNsPerToken: number;
  readonly draftNsPerToken: number;
  readonly lookupNsPerToken: number;
}

export interface TopologyCostModel {
  readonly revision: typeof TOPOLOGY_COST_MODEL_REVISION;
  readonly confidence: ConfidenceClass;
  readonly source: string;
  readonly deviceCosts: Readonly<Record<SimDeviceKind, DeviceCapabilityCost>>;
  readonly activationBytesPerToken: number;
  readonly collectiveBytesPerToken: number;
  readonly coldLoadByteMultiplier: number;
}

export interface TopologyResourceUtilization {
  readonly resourceId: string;
  readonly busyNs: number;
  readonly capacityLanes: number;
  readonly utilization: number;
}

export interface TopologyWorkloadMetrics {
  readonly totalDurationNs: number;
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
      draftTokens: iteration.proposedDraftTokens,
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
  return {
    id: "expert-cache",
    batchSize: 1,
    units: result.routes.map((route, index) => ({
      id: `route-${index}`,
      targetTokenWidth: 1,
      committedTokens: 1,
      draftTokens: 0,
      activeExperts: route.expertIds.length,
      warmLoadBytes: countTier(route.sourceTiers, "warm") * expertBytes,
      coldLoadBytes: countTier(route.sourceTiers, "cold") * expertBytes,
    })),
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
  validateInputs(scenario, profile, costModel);
  const compiler = new WorkloadPlanCompiler(scenario, profile, costModel);
  return compiler.compile();
}

export function simulateTopologyWorkload(
  scenario: SimulationScenario,
  profile: TopologyWorkloadProfile,
  costModel: TopologyCostModel = DEFAULT_TOPOLOGY_COST_MODEL,
): TopologyWorkloadResult {
  const plan = compileTopologyWorkloadPlan(scenario, profile, costModel);
  const execution = executeFrozenPlan(scenario, plan);
  replayPlanTrace(scenario, plan, execution.trace);
  const totalDurationNs = execution.completedAtNs;
  const committedTokens = profile.units.reduce(
    (sum, unit) => checkedAdd(sum, unit.committedTokens, "committed tokens"),
    0,
  );
  const computeServiceNs = serviceTime(execution, "compute");
  const transferServiceNs = serviceTime(execution, "transfer");
  const collectiveServiceNs = serviceTime(execution, "collective");
  return {
    scenarioId: scenario.id,
    profileId: profile.id,
    confidence: costModel.confidence,
    assumptions: [
      costModel.source,
      "decode-only plan; prefill and request batching are outside this profile",
      "compute costs include one device-kind invocation overhead plus linear token work",
      "family-specific proposer multipliers are heuristic and provenance-labeled",
      "transfer duration uses declared directed-link bandwidth and latency",
      "expert-load bytes are evenly sharded across FFN placements",
      "warm and cold expert loads originate in each FFN placement's local host domain",
    ],
    plan,
    execution,
    metrics: {
      totalDurationNs,
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
  private collectiveSequence = 0;
  private previousTerminalStepId?: number;

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
    for (const unit of this.profile.units) {
      this.compileUnit(unit);
    }
    return {
      contractRevision: PLAN_CONTRACT_REVISION,
      id: `topology-workload:${this.profile.id}`,
      executionId: `${this.scenario.id}:${this.profile.id}`,
      topologyEpoch: this.scenario.execution.topologyEpoch,
      steps: this.steps,
    };
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
      this.scenario.execution.parallelism.tensor > 1
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
    const group = this.scenario.groups.find((candidate) => (
      candidate.orderedRanks.length === this.placements.length
      && candidate.orderedRanks.every((rank) => (
        this.placements.some((placement) => placement.deviceId === rank.deviceId)
      ))
    ));
    if (!group) {
      throw new TopologyWorkloadError(
        `scenario ${this.scenario.id} lacks a communicator for tensor placements`,
      );
    }
    const linkIds = this.collectiveLinks(
      workspaceDomain(this.placements[0]),
      workspaceDomain(this.placements[this.placements.length - 1]),
    );
    const bytes = checkedMultiply(
      this.costModel.collectiveBytesPerToken,
      checkedMultiply(
        unit.targetTokenWidth,
        this.profile.batchSize,
        "collective token batch",
      ),
      "collective bytes",
    );
    const durationNs = this.pathDuration(linkIds, bytes);
    const step = this.addStep({
      participants: group.orderedRanks.map((rank) => rank.rankId),
      dependencies: terminals,
      reads: this.placements.map((placement) => workspaceId(placement)),
      writes: [],
      operation: {
        kind: "collective",
        groupId: group.id,
        commSequenceId: this.collectiveSequence++,
        linkIds,
        durationNs,
      },
    });
    return step;
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
    if (sourceDomain === targetDomain) {
      return {
        dependencies,
        localMemoryPenaltyNs: this.domainDuration(targetDomain, shardBytes),
      };
    }
    const transferSteps = this.addTransferPath(
      sourceDomain,
      targetDomain,
      shardBytes,
      dependencies,
      [this.rank(placement.deviceId)],
    );
    return {
      dependencies: transferSteps.length === 0
        ? dependencies
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
        ),
        index === 0 ? extraDurationNs : 0,
        "placement compute duration",
      );
      terminal = this.addCompute(
        placement,
        capability,
        durationNs,
        currentDependencies,
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
      reads: weights,
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
        reads: [this.buffer(source, sourceDomainId, targetDomainId)],
        writes: [this.buffer(target, sourceDomainId, targetDomainId)],
        operation: {
          kind: "transfer",
          linkId: link.id,
          durationNs: linkDuration(link, bytes),
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
    capability: "attention" | "ffn" | "draft" | "lookup",
    tokenWidth: number,
    activeExperts: number,
  ): number {
    const costs = this.costModel.deviceCosts[this.device(deviceId).kind];
    const perToken = capability === "attention"
      ? costs.attentionNsPerToken
      : capability === "ffn"
        ? costs.ffnNsPerToken
        : capability === "draft"
          ? costs.draftNsPerToken
          : costs.lookupNsPerToken;
    const unshardedDuration = checkedMultiply(
      perToken,
      checkedMultiply(
        tokenWidth,
        checkedMultiply(
          this.profile.batchSize,
          Math.max(1, activeExperts),
          "active expert batch",
        ),
        "compute token batch",
      ),
      "compute duration",
    );
    const tensorDegree = capability === "draft" || capability === "lookup"
      ? 1
      : this.scenario.execution.parallelism.tensor;
    const expertDegree = capability === "ffn"
      ? this.scenario.execution.parallelism.expert
      : 1;
    return checkedAdd(
      costs.invocationOverheadNs,
      Math.ceil(
        unshardedDuration
        / checkedMultiply(tensorDegree, expertDegree, "compute shard degree"),
      ),
      "compute invocation duration",
    );
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
    const domains = findTransferPath(this.scenario, {
      id: "compiled-collective",
      sourceDomainId,
      targetDomainId,
      bytes: 1,
      requiresPinnedStaging: false,
      stagingAllocationIds: [],
    });
    if (!domains || domains.length < 2) {
      throw new TopologyWorkloadError(
        `no collective path from ${sourceDomainId} to ${targetDomainId}`,
      );
    }
    return domains.slice(0, -1).map((domain, index) => (
      this.link(domain, domains[index + 1]).id
    ));
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
