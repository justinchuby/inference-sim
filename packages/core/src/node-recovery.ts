import type {
  FrozenPlan,
  FrozenPlanExecutionResult,
  PlanReplayResult,
} from "./plan-types.js";
import type { SimulationScenario } from "./scenario-types.js";
import {
  FrozenPlanExecutionError,
  PlanReplayError,
  executeFrozenPlan,
  replayPlanTrace,
  validateFrozenPlan,
} from "./frozen-plan.js";

export const NODE_RECOVERY_CONTRACT_REVISION = 1;

export interface NodeFailoverRequest {
  readonly failedNodeId: string;
  readonly faultAtNs: number;
  readonly reason: string;
  readonly recoveryScenario: SimulationScenario;
  readonly replannedPlan: FrozenPlan;
}

export interface NodeFailoverHandoff {
  readonly revision: typeof NODE_RECOVERY_CONTRACT_REVISION;
  readonly failedNodeId: string;
  readonly failedExecutionId: string;
  readonly recoveryExecutionId: string;
  readonly oldTopologyEpoch: number;
  readonly newTopologyEpoch: number;
  readonly faultAtNs: number;
  readonly oldExecutionQuiescedAtNs: number;
  readonly recoveryAdmittedAtNs: number;
}

export interface NodeFailoverResult {
  readonly handoff: NodeFailoverHandoff;
  readonly failedExecution: FrozenPlanExecutionResult;
  readonly failedReplay: PlanReplayResult;
  readonly recoveryExecution: FrozenPlanExecutionResult;
  readonly recoveryReplay: PlanReplayResult;
  readonly completedAtNs: number;
}

export interface NodeFailoverReplayResult {
  readonly failedReplay: PlanReplayResult;
  readonly recoveryReplay: PlanReplayResult;
  readonly completedAtNs: number;
}

export function runNodeFailoverCampaign(
  failedScenario: SimulationScenario,
  failedPlan: FrozenPlan,
  request: NodeFailoverRequest,
): NodeFailoverResult {
  validateNodeFailoverRequest(failedScenario, failedPlan, request);
  const failedExecution = executeFrozenPlan(failedScenario, failedPlan, {
    injectFault: {
      kind: "node_failure",
      atNs: request.faultAtNs,
      nodeId: request.failedNodeId,
      reason: request.reason,
    },
  });
  if (
    failedExecution.status !== "failed"
    || failedExecution.trace.terminal.fault?.kind !== "node_failure"
  ) {
    throw new FrozenPlanExecutionError(
      "node fault did not interrupt the old-epoch execution",
    );
  }
  const failedReplay = replayPlanTrace(
    failedScenario,
    failedPlan,
    failedExecution.trace,
  );
  const recoveryExecution = executeFrozenPlan(
    request.recoveryScenario,
    request.replannedPlan,
  );
  if (recoveryExecution.status !== "succeeded") {
    throw new FrozenPlanExecutionError(
      "replanned recovery execution did not succeed",
    );
  }
  const recoveryReplay = replayPlanTrace(
    request.recoveryScenario,
    request.replannedPlan,
    recoveryExecution.trace,
  );
  const handoff: NodeFailoverHandoff = {
    revision: NODE_RECOVERY_CONTRACT_REVISION,
    failedNodeId: request.failedNodeId,
    failedExecutionId: failedPlan.executionId,
    recoveryExecutionId: request.replannedPlan.executionId,
    oldTopologyEpoch: failedPlan.topologyEpoch,
    newTopologyEpoch: request.replannedPlan.topologyEpoch,
    faultAtNs: request.faultAtNs,
    oldExecutionQuiescedAtNs: failedExecution.completedAtNs,
    recoveryAdmittedAtNs: failedExecution.completedAtNs,
  };
  return {
    handoff,
    failedExecution,
    failedReplay,
    recoveryExecution,
    recoveryReplay,
    completedAtNs: checkedAdd(
      handoff.recoveryAdmittedAtNs,
      recoveryExecution.completedAtNs,
      "node failover completion",
    ),
  };
}

export function replayNodeFailoverCampaign(
  failedScenario: SimulationScenario,
  failedPlan: FrozenPlan,
  request: NodeFailoverRequest,
  result: NodeFailoverResult,
): NodeFailoverReplayResult {
  try {
    validateNodeFailoverRequest(failedScenario, failedPlan, request);
    const failedReplay = replayPlanTrace(
      failedScenario,
      failedPlan,
      result.failedExecution.trace,
    );
    const recoveryReplay = replayPlanTrace(
      request.recoveryScenario,
      request.replannedPlan,
      result.recoveryExecution.trace,
    );
    if (
      result.failedExecution.status !== "failed"
      || result.failedExecution.trace.terminal.fault?.kind !== "node_failure"
      || result.failedExecution.trace.terminal.fault.nodeId
        !== request.failedNodeId
      || result.failedExecution.trace.terminal.fault.atNs
        !== request.faultAtNs
      || result.failedExecution.trace.terminal.fault.reason !== request.reason
      || result.recoveryExecution.status !== "succeeded"
    ) {
      replayFail("execution statuses or node fault do not match failover request");
    }
    const expectedHandoff: NodeFailoverHandoff = {
      revision: NODE_RECOVERY_CONTRACT_REVISION,
      failedNodeId: request.failedNodeId,
      failedExecutionId: failedPlan.executionId,
      recoveryExecutionId: request.replannedPlan.executionId,
      oldTopologyEpoch: failedPlan.topologyEpoch,
      newTopologyEpoch: request.replannedPlan.topologyEpoch,
      faultAtNs: request.faultAtNs,
      oldExecutionQuiescedAtNs: failedReplay.completedAtNs,
      recoveryAdmittedAtNs: failedReplay.completedAtNs,
    };
    if (JSON.stringify(result.handoff) !== JSON.stringify(expectedHandoff)) {
      replayFail("failover handoff does not match old-epoch quiescence");
    }
    const completedAtNs = checkedAdd(
      expectedHandoff.recoveryAdmittedAtNs,
      recoveryReplay.completedAtNs,
      "replayed node failover completion",
    );
    if (result.completedAtNs !== completedAtNs) {
      replayFail("failover completion timestamp is inconsistent");
    }
    return { failedReplay, recoveryReplay, completedAtNs };
  } catch (error) {
    if (error instanceof PlanReplayError) {
      throw error;
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new PlanReplayError(`node failover: ${message}`);
  }
}

function validateNodeFailoverRequest(
  failedScenario: SimulationScenario,
  failedPlan: FrozenPlan,
  request: NodeFailoverRequest,
): void {
  const failedValidation = validateFrozenPlan(failedScenario, failedPlan);
  if (!failedValidation.valid) {
    throw new FrozenPlanExecutionError("old-epoch failover plan is invalid");
  }
  if (
    typeof request.failedNodeId !== "string"
    || request.failedNodeId.length === 0
    || !failedScenario.devices.some(
      (device) => device.nodeId === request.failedNodeId,
    )
  ) {
    throw new FrozenPlanExecutionError(
      `unknown failed node ${request.failedNodeId}`,
    );
  }
  if (
    !Number.isSafeInteger(request.faultAtNs)
    || request.faultAtNs < 0
    || typeof request.reason !== "string"
    || request.reason.length === 0
  ) {
    throw new FrozenPlanExecutionError(
      "node failover fault time and reason are invalid",
    );
  }
  if (
    request.recoveryScenario.execution.topologyEpoch
      <= failedScenario.execution.topologyEpoch
    || request.replannedPlan.topologyEpoch
      !== request.recoveryScenario.execution.topologyEpoch
  ) {
    throw new FrozenPlanExecutionError(
      "recovery scenario and plan must use a newer matching topology epoch",
    );
  }
  if (request.replannedPlan.executionId === failedPlan.executionId) {
    throw new FrozenPlanExecutionError(
      "recovery must use a new execution id",
    );
  }
  if (
    request.recoveryScenario.devices.some(
      (device) => device.nodeId === request.failedNodeId,
    )
    || request.recoveryScenario.memoryDomains.some(
      (domain) => domain.nodeId === request.failedNodeId,
    )
  ) {
    throw new FrozenPlanExecutionError(
      `failover scenario still contains failed node ${request.failedNodeId}`,
    );
  }
  const recoveryValidation = validateFrozenPlan(
    request.recoveryScenario,
    request.replannedPlan,
  );
  if (!recoveryValidation.valid) {
    throw new FrozenPlanExecutionError(
      `replanned recovery is invalid: ${
        recoveryValidation.issues.map((issue) => issue.message).join("; ")
      }`,
    );
  }
}

function checkedAdd(left: number, right: number, label: string): number {
  const value = left + right;
  if (!Number.isSafeInteger(value)) {
    throw new FrozenPlanExecutionError(`${label} exceeds safe integer range`);
  }
  return value;
}

function replayFail(message: string): never {
  throw new Error(message);
}
