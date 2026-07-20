import {
  simulateServingWorkload,
  type ServingBatchWork,
  type ServingSchedulerConfig,
  type ServingSimulationResult,
} from "./serving.js";
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

export function topologyProfileFromServingBatch(
  batch: ServingBatchWork,
): TopologyWorkloadProfile {
  return {
    id: `serving-batch:${batch.batchId}`,
    batchSize: 1,
    units: [{
      id: `batch-${batch.batchId}`,
      targetTokenWidth: batch.tokenWork,
      committedTokens: batch.expectedOutputTokens,
      draftTokens: 0,
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
      topologyProfileFromServingBatch(work),
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
  return {
    scenarioId: scenario.id,
    confidence: costModel.confidence,
    assumptions: [
      costModel.source,
      "continuous batches are non-preemptive; arrivals during a batch wait for its completion",
      "prefill and decode token work share the topology model's linear per-token coefficient",
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
