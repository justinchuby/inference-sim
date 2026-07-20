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

export const SERVING_EXPERT_CACHE_CONTRACT_REVISION = 1;

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
  readonly work: ServingBatchWork;
  readonly topology: TopologyWorkloadResult;
}

export interface TopologyServingMetrics {
  readonly totalDurationNs: number;
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
        const route = expertCache.processToken({
          tokenIndex: nextExpertTokenIndex++,
          topK: expertCacheConfig.topK,
          atNs,
        });
        batchExpertRoutes.push(route);
        expertRoutes.push(route);
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
          ),
      costModel,
    );
    const durationNs = Math.max(
      topology.metrics.totalDurationNs,
      cacheConstraintNs,
    );
    batches.set(work.batchId, {
      batchId: work.batchId,
      startedAtNs,
      durationNs,
      cacheConstraintNs,
      expertRoutes: batchExpertRoutes,
      work,
      topology,
    });
    return durationNs;
  };
  const serving = simulateServingWorkload(config, estimateDuration);
  const orderedBatches = [...batches.values()].sort(
    (left, right) => left.batchId - right.batchId,
  );
  const resourceBusy = new Map<string, {
    busyNs: number;
    capacityLanes: number;
  }>();
  let planSteps = 0;
  let computeOperations = 0;
  let transferOperations = 0;
  let collectiveOperations = 0;
  let allReduceOperations = 0;
  let allToAllOperations = 0;
  let computeServiceNs = 0;
  let transferServiceNs = 0;
  let collectiveServiceNs = 0;
  for (const batch of orderedBatches) {
    const metrics = batch.topology.metrics;
    planSteps += batch.topology.plan.steps.length;
    computeServiceNs += metrics.computeServiceNs;
    transferServiceNs += metrics.transferServiceNs;
    collectiveServiceNs += metrics.collectiveServiceNs;
    for (const event of batch.topology.execution.trace.operations) {
      if (event.kind === "compute") {
        computeOperations++;
      } else if (event.kind === "transfer") {
        transferOperations++;
      } else {
        collectiveOperations++;
        if (event.collectiveAlgorithm === "all_reduce_ring") {
          allReduceOperations++;
        } else if (event.collectiveAlgorithm === "all_to_all_v") {
          allToAllOperations++;
        }
      }
    }
    for (const resource of [
      ...metrics.computeUtilization,
      ...metrics.linkUtilization,
    ]) {
      const current = resourceBusy.get(resource.resourceId);
      resourceBusy.set(resource.resourceId, {
        busyNs: (current?.busyNs ?? 0) + resource.busyNs,
        capacityLanes: resource.capacityLanes,
      });
    }
  }
  const totalDurationNs = serving.metrics.totalDurationNs;
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
        : "composed batch duration is the maximum of topology execution and the expert-cache readiness constraint; demand transfer is not added twice",
      "batch plans execute serially while resources inside each plan retain declared concurrency",
    ],
    serving,
    batches: orderedBatches,
    ...(expertCacheResult === undefined
      ? {}
      : { expertCache: expertCacheResult }),
    metrics: {
      totalDurationNs,
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
      resourceUtilization: [...resourceBusy.entries()]
        .map(([resourceId, resource]) => ({
          resourceId,
          busyNs: resource.busyNs,
          capacityLanes: resource.capacityLanes,
          utilization: totalDurationNs === 0
            ? 0
            : resource.busyNs / (totalDurationNs * resource.capacityLanes),
        }))
        .sort((left, right) => (
          right.utilization - left.utilization
          || left.resourceId.localeCompare(right.resourceId)
        )),
    },
  };
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
  };
}

function checkedByteAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result) || result < 0) {
    throw new Error(`${label} exceeds the safe integer range`);
  }
  return result;
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
  if (config.cache.adaptivePrefetch !== undefined) {
    throw new Error(
      "serving expert-cache composition does not yet support adaptive background prefetch",
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
