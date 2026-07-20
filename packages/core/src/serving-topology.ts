import {
  simulateServingWorkload,
  type ServingBatchWork,
  type ServingSchedulerConfig,
  type ServingSimulationResult,
} from "./serving.js";
import {
  speculativeFamilyContract,
} from "./speculative-family.js";
import {
  ExpertCacheSimulator,
  replayExpertCacheTrace,
  type ExpertCacheConfig,
  type ExpertCacheReplayResult,
  type ExpertCacheSnapshot,
  type ExpertCacheTraceEvent,
  type ExpertPendingLoadSnapshot,
  type ExpertRouteResult,
} from "./expert-cache.js";
import type {
  ConfidenceClass,
  SimulationScenario,
} from "./scenario-types.js";
import {
  DEFAULT_TOPOLOGY_COST_MODEL,
  simulateTopologyWorkload,
  type TopologyCostModel,
  type TopologyResourceUtilization,
  type TopologyWorkloadProfile,
  type TopologyWorkloadResult,
} from "./topology-workload.js";
import {
  StreamingConcurrentPlanRuntime,
  replayConcurrentPlanTrace,
  type ConcurrentPlanExecutionResult,
  type ConcurrentPlanReplayResult,
  type ConcurrentPlanRequest,
} from "./concurrent-plan.js";
import type {
  FrozenPlanExecutionResult,
  PlanTraceEvent,
} from "./plan-types.js";

export const SERVING_EXPERT_CACHE_CONTRACT_REVISION = 2;
const DEFERRED_PREFETCH_COMPLETION_NS = Number.MAX_SAFE_INTEGER;

export interface TopologyServingExpertCacheConfig {
  readonly contractRevision:
    typeof SERVING_EXPERT_CACHE_CONTRACT_REVISION;
  readonly cache: ExpertCacheConfig;
  readonly topK: number;
}

export interface TopologyServingExpertCacheResult {
  readonly contractRevision:
    typeof SERVING_EXPERT_CACHE_CONTRACT_REVISION;
  readonly routes: readonly ExpertRouteResult[];
  readonly snapshot: ExpertCacheSnapshot;
  readonly trace: readonly ExpertCacheTraceEvent[];
  readonly replay: ExpertCacheReplayResult;
}

export interface TopologyServingBatchResult {
  readonly batchId: number;
  readonly startedAtNs: number;
  readonly durationNs: number;
  readonly cacheConstraintNs: number;
  readonly expertRoutes: readonly ExpertRouteResult[];
  readonly foregroundCompletedAtNs: number;
  readonly physicalExecution?: FrozenPlanExecutionResult;
  readonly work: ServingBatchWork;
  readonly topology: TopologyWorkloadResult;
}

export interface TopologyServingPhysicalResult {
  readonly execution: ConcurrentPlanExecutionResult;
  readonly replay: ConcurrentPlanReplayResult;
}

export interface TopologyServingMetrics {
  readonly totalDurationNs: number;
  readonly resourceObservationNs: number;
  readonly backgroundDrainNs: number;
  readonly batchServiceNs: number;
  readonly idleNs: number;
  readonly planSteps: number;
  readonly computeOperations: number;
  readonly transferOperations: number;
  readonly collectiveOperations: number;
  readonly allReduceOperations: number;
  readonly allToAllOperations: number;
  readonly computeServiceNs: number;
  readonly transferServiceNs: number;
  readonly collectiveServiceNs: number;
  readonly resourceUtilization: readonly TopologyResourceUtilization[];
}

export interface TopologyServingResult {
  readonly scenarioId: string;
  readonly confidence: ConfidenceClass;
  readonly assumptions: readonly string[];
  readonly serving: ServingSimulationResult;
  readonly batches: readonly TopologyServingBatchResult[];
  readonly expertCache?: TopologyServingExpertCacheResult;
  readonly physical?: TopologyServingPhysicalResult;
  readonly metrics: TopologyServingMetrics;
}

export interface TopologyServingComparisonRun {
  readonly rank: number;
  readonly relativeToFastest: number;
  readonly result: TopologyServingResult;
}

export interface TopologyServingComparisonResult {
  readonly runs: readonly TopologyServingComparisonRun[];
}

interface PhysicalPrefetchBinding {
  readonly load: ExpertPendingLoadSnapshot;
  readonly expectedStepKeys: ReadonlySet<string>;
  readonly submittedFinishes: Map<string, number>;
  retimed: boolean;
}

export function topologyProfileFromServingBatch(
  batch: ServingBatchWork,
  config?: ServingSchedulerConfig["speculative"],
): TopologyWorkloadProfile {
  const draftTokens = batch.decode.reduce(
    (sum, entry) => sum + entry.proposedAdditionalTokens,
    0,
  );
  const family = config
    ? speculativeFamilyContract(config.family)
    : undefined;
  return {
    id: `serving-batch:${batch.batchId}`,
    batchSize: 1,
    units: [{
      id: `batch-${batch.batchId}`,
      targetTokenWidth: batch.tokenWork,
      committedTokens: batch.expectedOutputTokens,
      draftTokens,
      ...(draftTokens > 0 && family
        ? {
            proposerExecution: family.execution,
            proposerCostScale: family.proposerCostScale,
          }
        : {}),
      activeExperts: 1,
      warmLoadBytes: 0,
      coldLoadBytes: 0,
    }],
  };
}

export function simulateTopologyServingWorkload(
  scenario: SimulationScenario,
  config: ServingSchedulerConfig,
  costModel: TopologyCostModel = DEFAULT_TOPOLOGY_COST_MODEL,
  expertCacheConfig?: TopologyServingExpertCacheConfig,
): TopologyServingResult {
  validateExpertCacheComposition(expertCacheConfig);
  const batches = new Map<number, TopologyServingBatchResult>();
  const expertCache = expertCacheConfig === undefined
    ? undefined
    : new ExpertCacheSimulator(expertCacheConfig.cache);
  const physicalRuntime = expertCacheConfig === undefined
    ? undefined
    : new StreamingConcurrentPlanRuntime(scenario);
  const physicalRequests: ConcurrentPlanRequest[] = [];
  const physicalPrefetchByStep = new Map<string, PhysicalPrefetchBinding>();
  const physicalPrefetchBindings: PhysicalPrefetchBinding[] = [];
  physicalRuntime?.onOperationSubmitted((event) => {
    applyPhysicalPrefetchSubmission(
      expertCache,
      physicalPrefetchByStep,
      event,
    );
  });
  const expertRoutes: ExpertRouteResult[] = [];
  let nextExpertTokenIndex = 0;
  const estimateDuration = (
    work: ServingBatchWork,
    startedAtNs: number,
  ): number => {
    const existing = batches.get(work.batchId);
    if (existing) {
      if (
        existing.startedAtNs !== startedAtNs
        || JSON.stringify(existing.work) !== JSON.stringify(work)
      ) {
        throw new Error(
          `serving batch ${work.batchId} timing/work changed between simulation and replay`,
        );
      }
      return existing.durationNs;
    }
    const batchExpertRoutes: ExpertRouteResult[] = [];
    const batchPrefetchLoads: ExpertPendingLoadSnapshot[] = [];
    physicalRuntime?.advanceTo(startedAtNs);
    const cacheStartedAtNs = expertCache?.snapshot().currentTimeNs
      ?? startedAtNs;
    if (cacheStartedAtNs > startedAtNs) {
      throw new Error(
        `expert cache time ${cacheStartedAtNs}ns exceeds serving batch ${work.batchId} start ${startedAtNs}ns`,
      );
    }
    if (expertCache !== undefined && expertCacheConfig !== undefined) {
      for (let index = 0; index < work.tokenWork; index++) {
        const atNs = Math.max(
          startedAtNs,
          expertCache.snapshot().currentTimeNs,
        );
        const traceStart = expertCache.traceLength();
        const route = expertCache.processToken({
          tokenIndex: nextExpertTokenIndex++,
          topK: expertCacheConfig.topK,
          atNs,
        });
        batchExpertRoutes.push(route);
        expertRoutes.push(route);
        for (const event of expertCache.traceFrom(traceStart)) {
          if (
            event.kind === "load_start"
            && event.load.kind === "prefetch"
            && event.load.targetTier === "warm"
          ) {
            batchPrefetchLoads.push(event.load);
            expertCache.retimePendingPrefetch(
              event.load.loadId,
              DEFERRED_PREFETCH_COMPLETION_NS,
            );
          }
        }
      }
    }
    const cacheConstraintNs = expertCache === undefined
      ? 0
      : expertCache.snapshot().currentTimeNs - startedAtNs;
    const topology = simulateTopologyWorkload(
      scenario,
      batchExpertRoutes.length === 0
        ? topologyProfileFromServingBatch(work, config.speculative)
        : topologyProfileFromExpertServingBatch(
            work,
            config.speculative,
            batchExpertRoutes,
            expertCacheConfig?.cache.experts ?? [],
            batchPrefetchLoads,
          ),
      costModel,
    );
    let foregroundCompletedAtNs = startedAtNs
      + topology.metrics.foregroundDurationNs;
    if (physicalRuntime !== undefined) {
      const stepNotBeforeNs = registerPhysicalPrefetches(
        topology,
        batchPrefetchLoads,
        physicalPrefetchByStep,
        physicalPrefetchBindings,
      );
      physicalRuntime.admit(
        topology.plan,
        startedAtNs,
        stepNotBeforeNs,
      );
      physicalRequests.push({
        plan: topology.plan,
        arrivalNs: startedAtNs,
        admissionOrder: physicalRequests.length,
        ...(Object.keys(stepNotBeforeNs).length === 0
          ? {}
          : { stepNotBeforeNs }),
      });
      foregroundCompletedAtNs = physicalRuntime.runUntilStep(
        topology.plan.executionId,
        topology.foregroundTerminalStepId,
      ).completedAtNs;
    }
    const physicalForegroundNs = foregroundCompletedAtNs - startedAtNs;
    const durationNs = Math.max(
      physicalForegroundNs,
      cacheConstraintNs,
    );
    batches.set(work.batchId, {
      batchId: work.batchId,
      startedAtNs,
      durationNs,
      cacheConstraintNs,
      expertRoutes: batchExpertRoutes,
      foregroundCompletedAtNs,
      work,
      topology,
    });
    return durationNs;
  };
  const serving = simulateServingWorkload(config, estimateDuration);
  const physical = physicalRuntime === undefined
    ? undefined
    : buildPhysicalResult(
        scenario,
        physicalRuntime,
        physicalRequests,
      );
  if (physical !== undefined && expertCache !== undefined) {
    const unresolved = physicalPrefetchBindings.filter(
      (binding) => !binding.retimed,
    );
    if (unresolved.length > 0) {
      throw new Error(
        `physical execution drained with unresolved adaptive prefetches: ${
          unresolved.map((binding) => binding.load.loadId).join(", ")
        }`,
      );
    }
    if (physicalPrefetchBindings.length > 0) {
      const finalPrefetchCompletionNs = Math.max(
        ...physicalPrefetchBindings.flatMap((binding) => (
          [...binding.submittedFinishes.values()]
        )),
      );
      expertCache.advanceTo(Math.max(
        expertCache.snapshot().currentTimeNs,
        finalPrefetchCompletionNs,
      ));
    }
  }
  const physicalByExecution = new Map(
    physical?.execution.executions.map((execution) => [
      execution.executionId,
      execution,
    ]) ?? [],
  );
  const orderedBatches = [...batches.values()].sort(
    (left, right) => left.batchId - right.batchId,
  ).map((batch) => {
    const physicalExecution = physicalByExecution.get(
      batch.topology.plan.executionId,
    );
    return physicalExecution === undefined
      ? batch
      : { ...batch, physicalExecution };
  });
  const planSteps = orderedBatches.reduce(
    (sum, batch) => checkedMetricAdd(
      sum,
      batch.topology.plan.steps.length,
      "serving plan steps",
    ),
    0,
  );
  const operationEvents = physical === undefined
    ? orderedBatches.flatMap(
        (batch) => batch.topology.execution.trace.operations,
      )
    : physical.execution.trace.operations.map(({ event }) => event);
  if (operationEvents.length !== planSteps) {
    throw new Error(
      `serving physical trace has ${operationEvents.length}/${planSteps} operations`,
    );
  }
  let computeOperations = 0;
  let transferOperations = 0;
  let collectiveOperations = 0;
  let allReduceOperations = 0;
  let allToAllOperations = 0;
  let computeServiceNs = 0;
  let transferServiceNs = 0;
  let collectiveServiceNs = 0;
  const resourceBusy = new Map<string, number>();
  for (const event of operationEvents) {
    const serviceNs = event.finishNs - event.startNs;
    if (event.kind === "compute") {
      computeOperations++;
      computeServiceNs = checkedMetricAdd(
        computeServiceNs,
        serviceNs,
        "serving compute service",
      );
    } else if (event.kind === "transfer") {
      transferOperations++;
      transferServiceNs = checkedMetricAdd(
        transferServiceNs,
        serviceNs,
        "serving transfer service",
      );
    } else {
      collectiveOperations++;
      collectiveServiceNs = checkedMetricAdd(
        collectiveServiceNs,
        serviceNs,
        "serving collective service",
      );
      if (event.collectiveAlgorithm === "all_reduce_ring") {
        allReduceOperations++;
      } else if (event.collectiveAlgorithm === "all_to_all_v") {
        allToAllOperations++;
      }
    }
    for (const reservation of event.resources) {
      if (
        !reservation.resourceId.startsWith("compute:")
        && !reservation.resourceId.startsWith("link:")
      ) {
        continue;
      }
      resourceBusy.set(
        reservation.resourceId,
        checkedMetricAdd(
          resourceBusy.get(reservation.resourceId) ?? 0,
          serviceNs,
          `serving resource ${reservation.resourceId} busy time`,
        ),
      );
    }
  }
  const totalDurationNs = serving.metrics.totalDurationNs;
  const resourceObservationNs = Math.max(
    totalDurationNs,
    physical?.execution.completedAtNs ?? totalDurationNs,
  );
  const resourceUtilization = [...resourceBusy.entries()]
    .map(([resourceId, busyNs]) => {
      const capacityLanes = servingResourceCapacity(scenario, resourceId);
      const capacityNs = checkedMetricMultiply(
        resourceObservationNs,
        capacityLanes,
        `serving resource ${resourceId} observation capacity`,
      );
      const utilization = resourceObservationNs === 0
        ? 0
        : busyNs / capacityNs;
      if (utilization > 1) {
        throw new Error(
          `serving resource ${resourceId} utilization ${utilization} exceeds 1`,
        );
      }
      return {
        resourceId,
        busyNs,
        capacityLanes,
        utilization,
      };
    })
    .sort((left, right) => (
      right.utilization - left.utilization
      || left.resourceId.localeCompare(right.resourceId)
    ));
  const confidence = orderedBatches[0]?.topology.confidence
    ?? costModel.confidence;
  const expertCacheResult: TopologyServingExpertCacheResult | undefined =
    expertCache === undefined
      ? undefined
      : buildExpertCacheResult(expertCache, expertRoutes);
  return {
    scenarioId: scenario.id,
    confidence,
    assumptions: [
      costModel.source,
      orderedBatches[0]?.topology.assumptions[1]
        ?? `overall timing confidence is ${confidence}`,
      orderedBatches[0]?.topology.assumptions.find((assumption) => (
        assumption.startsWith("transport timing uses")
      )) ?? "transport timing was not exercised by the first batch",
      "continuous batches are non-preemptive; arrivals during a batch wait for its completion",
      "prefill and target verification share fixed invocation plus linear token cost",
      config.speculative
        ? `${config.speculative.family} proposals use per-request deterministic acceptance streams and transactional restore`
        : "decode uses one target-authoritative token per sequence step",
      expertCache === undefined
        ? "target FFN execution is dense and does not model expert residency"
        : "expert routes preserve cache state across batches; routes within each batch are serialized conservatively",
      expertCache === undefined
        ? "batch duration is determined by topology execution"
        : "composed batch duration is the maximum of global foreground completion and the expert-cache readiness constraint; demand transfer is not added twice",
      expertCache === undefined
        ? "batch plans use isolated relative-time execution"
        : "composed batch plans share one absolute-time resource, lease, and collective timeline through final drain",
      "resource utilization is measured from authoritative operation reservations over the request-start-to-global-quiescence observation window",
      expertCache === undefined
        ? "batch plans execute serially while resources inside each plan retain declared concurrency"
        : "batch foregrounds are non-preemptive; background transfers from older plans may overlap later foreground plans",
    ],
    serving,
    batches: orderedBatches,
    ...(expertCacheResult === undefined
      ? {}
      : { expertCache: expertCacheResult }),
    ...(physical === undefined ? {} : { physical }),
    metrics: {
      totalDurationNs,
      resourceObservationNs,
      backgroundDrainNs: resourceObservationNs - totalDurationNs,
      batchServiceNs: serving.metrics.batchServiceNs,
      idleNs: totalDurationNs - serving.metrics.batchServiceNs,
      planSteps,
      computeOperations,
      transferOperations,
      collectiveOperations,
      allReduceOperations,
      allToAllOperations,
      computeServiceNs,
      transferServiceNs,
      collectiveServiceNs,
      resourceUtilization,
    },
  };
}

function buildPhysicalResult(
  scenario: SimulationScenario,
  runtime: StreamingConcurrentPlanRuntime,
  requests: readonly ConcurrentPlanRequest[],
): TopologyServingPhysicalResult {
  const execution = runtime.drain();
  const replay = replayConcurrentPlanTrace(
    scenario,
    requests,
    execution.trace,
  );
  return { execution, replay };
}

export function compareTopologyServingWorkloads(
  scenarios: readonly SimulationScenario[],
  config: ServingSchedulerConfig,
  costModel: TopologyCostModel = DEFAULT_TOPOLOGY_COST_MODEL,
  expertCacheConfig?: TopologyServingExpertCacheConfig,
): TopologyServingComparisonResult {
  if (scenarios.length === 0) {
    throw new Error("serving comparison requires at least one scenario");
  }
  const scenarioIds = new Set<string>();
  for (const scenario of scenarios) {
    if (scenarioIds.has(scenario.id)) {
      throw new Error(
        `serving comparison scenario id must be unique; got ${scenario.id}`,
      );
    }
    scenarioIds.add(scenario.id);
  }
  const results = scenarios
    .map((scenario) => simulateTopologyServingWorkload(
      scenario,
      config,
      costModel,
      expertCacheConfig,
    ))
    .sort((left, right) => (
      left.metrics.totalDurationNs - right.metrics.totalDurationNs
      || left.scenarioId.localeCompare(right.scenarioId)
    ));
  const fastestDurationNs = results[0].metrics.totalDurationNs;
  return {
    runs: results.map((result, index) => ({
      rank: index + 1,
      relativeToFastest:
        result.metrics.totalDurationNs / fastestDurationNs,
      result,
    })),
  };
}

function topologyProfileFromExpertServingBatch(
  batch: ServingBatchWork,
  speculative: ServingSchedulerConfig["speculative"],
  routes: readonly ExpertRouteResult[],
  experts: ExpertCacheConfig["experts"],
  prefetchLoads: readonly ExpertPendingLoadSnapshot[],
): TopologyWorkloadProfile {
  if (routes.length !== batch.tokenWork || routes.length === 0) {
    throw new Error(
      `serving batch ${batch.batchId} requires one expert route per target token`,
    );
  }
  const base = topologyProfileFromServingBatch(batch, speculative);
  const expertBytes = new Map(experts.map((expert) => [
    expert.id,
    expert.bytes,
  ]));
  const topK = routes[0].expertIds.length;
  if (routes.some((route) => route.expertIds.length !== topK)) {
    throw new Error(
      `serving batch ${batch.batchId} changed expert topK within the batch`,
    );
  }
  let warmLoadBytes = 0;
  let coldLoadBytes = 0;
  for (const route of routes) {
    for (let index = 0; index < route.expertIds.length; index++) {
      const bytes = expertBytes.get(route.expertIds[index]);
      if (bytes === undefined) {
        throw new Error(
          `serving batch ${batch.batchId} routed unknown expert ${route.expertIds[index]}`,
        );
      }
      if (route.sourceTiers[index] === "warm") {
        warmLoadBytes = checkedByteAdd(
          warmLoadBytes,
          bytes,
          `serving batch ${batch.batchId} warm expert loads`,
        );
      } else if (route.sourceTiers[index] === "cold") {
        coldLoadBytes = checkedByteAdd(
          coldLoadBytes,
          bytes,
          `serving batch ${batch.batchId} cold expert loads`,
        );
      }
    }
  }
  return {
    ...base,
    id: `${base.id}:expert-cache`,
    units: base.units.map((unit) => ({
      ...unit,
      activeExperts: topK,
      expertRouted: true,
      warmLoadBytes,
      coldLoadBytes,
    })),
    backgroundPrefetches: prefetchLoads.map((load) => ({
      id: load.loadId,
      afterUnitIndex: 0,
      bytes: load.bytes,
    })),
  };
}

function registerPhysicalPrefetches(
  topology: TopologyWorkloadResult,
  loads: readonly ExpertPendingLoadSnapshot[],
  byStep: Map<string, PhysicalPrefetchBinding>,
  bindings: PhysicalPrefetchBinding[],
): Readonly<Record<number, number>> {
  const stepNotBeforeNs: Record<number, number> = {};
  for (const load of loads) {
    const terminals = topology.backgroundPrefetchTerminals.filter(
      (terminal) => terminal.prefetchId === load.loadId,
    );
    if (terminals.length === 0) {
      throw new Error(
        `adaptive prefetch ${load.loadId} has no physical terminal`,
      );
    }
    const expectedStepKeys = new Set(terminals.map((terminal) => (
      physicalStepKey(topology.plan.executionId, terminal.stepId)
    )));
    const binding: PhysicalPrefetchBinding = {
      load,
      expectedStepKeys,
      submittedFinishes: new Map(),
      retimed: false,
    };
    bindings.push(binding);
    for (const stepKey of expectedStepKeys) {
      if (byStep.has(stepKey)) {
        throw new Error(
          `duplicate physical prefetch terminal ${stepKey}`,
        );
      }
      byStep.set(stepKey, binding);
    }
    for (const terminal of terminals) {
      for (const stepId of terminal.stepIds) {
        const prior = stepNotBeforeNs[stepId];
        if (prior !== undefined && prior !== load.startedAtNs) {
          throw new Error(
            `physical prefetch step ${stepId} has conflicting policy times`,
          );
        }
        stepNotBeforeNs[stepId] = load.startedAtNs;
      }
    }
  }
  return stepNotBeforeNs;
}

function applyPhysicalPrefetchSubmission(
  cache: ExpertCacheSimulator | undefined,
  byStep: ReadonlyMap<string, PhysicalPrefetchBinding>,
  event: PlanTraceEvent,
): void {
  const stepKey = physicalStepKey(event.executionId, event.stepId);
  const binding = byStep.get(stepKey);
  if (binding === undefined) {
    return;
  }
  if (
    cache === undefined
    || event.kind !== "transfer"
    || event.startNs < binding.load.startedAtNs
    || binding.submittedFinishes.has(stepKey)
  ) {
    throw new Error(
      `adaptive prefetch ${binding.load.loadId} physical submission violates cache causality`,
    );
  }
  binding.submittedFinishes.set(stepKey, event.finishNs);
  if (
    binding.submittedFinishes.size === binding.expectedStepKeys.size
  ) {
    const completesAtNs = Math.max(
      ...binding.submittedFinishes.values(),
    );
    cache.retimePendingPrefetch(
      binding.load.loadId,
      Math.max(completesAtNs, cache.snapshot().currentTimeNs),
      completesAtNs,
    );
    binding.retimed = true;
  }
}

function physicalStepKey(executionId: string, stepId: number): string {
  return `${executionId}\u0000${stepId}`;
}

function checkedByteAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result) || result < 0) {
    throw new Error(`${label} exceeds the safe integer range`);
  }
  return result;
}

function checkedMetricAdd(
  left: number,
  right: number,
  label: string,
): number {
  const result = left + right;
  if (!Number.isSafeInteger(result) || result < 0) {
    throw new Error(`${label} exceeds the safe integer range`);
  }
  return result;
}

function checkedMetricMultiply(
  left: number,
  right: number,
  label: string,
): number {
  const result = left * right;
  if (!Number.isSafeInteger(result) || result < 0) {
    throw new Error(`${label} exceeds the safe integer range`);
  }
  return result;
}

function servingResourceCapacity(
  scenario: SimulationScenario,
  resourceId: string,
): number {
  const capacity = resourceId.startsWith("compute:")
    ? scenario.devices.find(
        (device) => resourceId === `compute:${device.id}`,
      )?.maxConcurrentCompute
    : resourceId.startsWith("link:")
      ? scenario.links.find(
          (link) => resourceId === `link:${link.id}`,
        )?.concurrencyLanes
      : undefined;
  if (capacity === undefined || capacity <= 0) {
    throw new Error(`serving trace reserved unknown resource ${resourceId}`);
  }
  return capacity;
}

function validateExpertCacheComposition(
  config: TopologyServingExpertCacheConfig | undefined,
): void {
  if (config === undefined) {
    return;
  }
  if (
    config.contractRevision !== SERVING_EXPERT_CACHE_CONTRACT_REVISION
  ) {
    throw new Error(
      `unsupported serving expert-cache contract revision ${config.contractRevision}`,
    );
  }
  if (!Number.isSafeInteger(config.topK) || config.topK <= 0) {
    throw new Error("serving expert-cache topK must be a positive safe integer");
  }
  if (config.topK > config.cache.experts.length) {
    throw new Error(
      `serving expert-cache topK ${config.topK} exceeds ${config.cache.experts.length} experts`,
    );
  }
}

function buildExpertCacheResult(
  cache: ExpertCacheSimulator,
  routes: readonly ExpertRouteResult[],
): TopologyServingExpertCacheResult {
  const snapshot = cache.snapshot();
  const trace = cache.trace();
  const replay = replayExpertCacheTrace(trace);
  if (JSON.stringify(replay.snapshot) !== JSON.stringify(snapshot)) {
    throw new Error("serving expert-cache replay diverged from live state");
  }
  return {
    contractRevision: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
    routes,
    snapshot,
    trace,
    replay,
  };
}
