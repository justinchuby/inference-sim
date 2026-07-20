import type { ComputeCapability } from "./scenario-types.js";

export const PLAN_CONTRACT_REVISION = 2;

export type PlanOperation =
  | {
      readonly kind: "compute";
      readonly deviceId: string;
      readonly capability: ComputeCapability;
      readonly durationNs: number;
    }
  | {
      readonly kind: "transfer";
      readonly linkId: string;
      readonly durationNs: number;
    }
  | {
      readonly kind: "collective";
      readonly groupId: string;
      readonly commSequenceId: number;
      readonly linkIds: readonly string[];
      readonly durationNs: number;
    };

export interface PlanStep {
  readonly id: number;
  readonly participants: readonly string[];
  readonly dependencies: readonly number[];
  readonly reads: readonly string[];
  readonly writes: readonly string[];
  readonly operation: PlanOperation;
}

export interface FrozenPlan {
  readonly contractRevision: typeof PLAN_CONTRACT_REVISION;
  readonly id: string;
  readonly executionId: string;
  readonly topologyEpoch: number;
  readonly steps: readonly PlanStep[];
}

export interface PlanValidationIssue {
  readonly code: string;
  readonly path: string;
  readonly message: string;
}

export interface PlanValidationResult {
  readonly valid: boolean;
  readonly issues: readonly PlanValidationIssue[];
  readonly topologicalStepIds: readonly number[];
}

export interface PlanTraceEvent {
  readonly contractRevision: typeof PLAN_CONTRACT_REVISION;
  readonly executionId: string;
  readonly topologyEpoch: number;
  readonly sourceSequence: number;
  readonly stepId: number;
  readonly kind: PlanOperation["kind"];
  readonly submittedAtNs: number;
  readonly startNs: number;
  readonly finishNs: number;
  readonly resources: readonly PlanResourceReservation[];
  readonly participants: readonly string[];
  readonly reads: readonly string[];
  readonly writes: readonly string[];
  readonly groupId?: string;
  readonly commSequenceId?: number;
}

export interface PlanResourceReservation {
  readonly resourceId: string;
  readonly resourceLane: number;
}

export interface RankCompletion {
  readonly rankId: string;
  readonly completedAtNs: number;
}

export type PlanTerminalStatus = "succeeded" | "failed" | "aborted";

export interface RankTerminalState {
  readonly rankId: string;
  readonly status: PlanTerminalStatus;
  readonly terminalAtNs: number;
}

export interface PlanTerminalEvent {
  readonly contractRevision: typeof PLAN_CONTRACT_REVISION;
  readonly executionId: string;
  readonly topologyEpoch: number;
  readonly sourceSequence: number;
  readonly kind: "execution_terminal";
  readonly status: PlanTerminalStatus;
  readonly timestampNs: number;
  readonly beforeStepId?: number;
  readonly failureAtNs?: number;
  readonly reason?: string;
  readonly rankStates: readonly RankTerminalState[];
}

export interface PlanExecutionTrace {
  readonly operations: readonly PlanTraceEvent[];
  readonly terminal: PlanTerminalEvent;
}

export interface FrozenPlanExecutionResult {
  readonly status: PlanTerminalStatus;
  readonly executionId: string;
  readonly completedAtNs: number;
  readonly trace: PlanExecutionTrace;
  readonly rankCompletions: readonly RankCompletion[];
  readonly rankStates: readonly RankTerminalState[];
}

export interface PlanReplayResult {
  readonly status: PlanTerminalStatus;
  readonly appliedEvents: number;
  readonly completedAtNs: number;
  readonly rankCompletions: readonly RankCompletion[];
  readonly rankStates: readonly RankTerminalState[];
}

export interface PlanExecutionOptions {
  readonly injectTerminalBeforeStep?: {
    readonly stepId: number;
    readonly status: "failed" | "aborted";
    readonly reason: string;
  };
}
