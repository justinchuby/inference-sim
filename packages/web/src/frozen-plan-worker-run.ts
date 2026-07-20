import {
  executeFrozenPlan,
  replayPlanTrace,
  type FrozenPlanExecutionResult,
  type PlanReplayResult,
} from "@inference-sim/core";
import { parseFrozenPlanArtifactFileText } from "./frozen-plan-import.js";
import type {
  FrozenPlanBrowserResult,
  WorkerRunProgressReporter,
} from "./types.js";

export function executeFrozenPlanWorkerRun(
  artifactText: string,
  sourceFileName: string,
  reportProgress: WorkerRunProgressReporter = () => {},
): FrozenPlanBrowserResult {
  reportProgress({ progress: 12, phase: "Validating FrozenPlan artifact" });
  const artifact = parseFrozenPlanArtifactFileText(
    artifactText,
    sourceFileName,
  );
  reportProgress({ progress: 35, phase: "Executing frozen operations" });
  const execution = executeFrozenPlan(artifact.scenario, artifact.plan);
  reportProgress({ progress: 68, phase: "Replaying plan trace" });
  const replay = replayPlanTrace(
    artifact.scenario,
    artifact.plan,
    execution.trace,
  );
  assertExecutionReplayParity(execution, replay);
  reportProgress({ progress: 88, phase: "Summarizing rank terminals" });

  const operationCounts = {
    compute: 0,
    transfer: 0,
    collective: 0,
  };
  for (const operation of execution.trace.operations) {
    operationCounts[operation.kind]++;
  }
  reportProgress({ progress: 94, phase: "Preparing FrozenPlan report" });
  return {
    sourceFileName,
    artifact: {
      revision: artifact.revision,
      artifactFingerprint: artifact.artifactFingerprint,
      scenarioFingerprint: artifact.scenarioFingerprint,
      planFingerprint: artifact.planFingerprint,
    },
    scenario: {
      id: artifact.scenario.id,
      family: artifact.scenario.family,
      devices: artifact.scenario.devices.length,
      links: artifact.scenario.links.length,
      ranks: new Set(artifact.scenario.groups.flatMap((group) => (
        group.orderedRanks.map((rank) => rank.rankId)
      ))).size,
    },
    plan: {
      id: artifact.plan.id,
      executionId: artifact.plan.executionId,
      topologyEpoch: artifact.plan.topologyEpoch,
      steps: artifact.plan.steps.length,
      operationCounts,
    },
    execution: {
      status: execution.status,
      completedAtNs: execution.completedAtNs,
      rankCompletions: execution.rankCompletions,
      rankStates: execution.rankStates,
      operationPreview: execution.trace.operations.slice(0, 100),
      operationCount: execution.trace.operations.length,
    },
    replay: {
      status: replay.status,
      completedAtNs: replay.completedAtNs,
      appliedEvents: replay.appliedEvents,
      exact: true,
    },
  };
}

function assertExecutionReplayParity(
  execution: FrozenPlanExecutionResult,
  replay: PlanReplayResult,
): void {
  if (
    execution.status !== replay.status
    || execution.completedAtNs !== replay.completedAtNs
    || JSON.stringify(execution.rankCompletions)
      !== JSON.stringify(replay.rankCompletions)
    || JSON.stringify(execution.rankStates) !== JSON.stringify(replay.rankStates)
  ) {
    throw new Error("FrozenPlan execution and independent replay diverged");
  }
}
