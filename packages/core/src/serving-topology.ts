import {
  simulateServingWorkload,
  type ServingBatchWork,
  type ServingSchedulerConfig,
  type ServingSimulationResult,
} from "./serving.js";
import {
  speculativeFamilyContract,
} from "./speculative-family.js";
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

export interface TopologyServingBatchResult {
  readonly batchId: number;
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
): TopologyServingResult {
  const batches = new Map<number, TopologyServingBatchResult>();
  const estimateDuration = (work: ServingBatchWork): number => {
    const existing = batches.get(work.batchId);
    if (existing) {
      if (JSON.stringify(existing.work) !== JSON.stringify(work)) {
        throw new Error(
          `serving batch ${work.batchId} changed between simulation and replay`,
        );
      }
      return existing.topology.metrics.totalDurationNs;
    }
    const topology = simulateTopologyWorkload(
      scenario,
      topologyProfileFromServingBatch(work, config.speculative),
      costModel,
    );
    batches.set(work.batchId, { batchId: work.batchId, work, topology });
    return topology.metrics.totalDurationNs;
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
      "batch plans execute serially while resources inside each plan retain declared concurrency",
    ],
    serving,
    batches: orderedBatches,
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
