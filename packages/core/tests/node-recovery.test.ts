import { describe, expect, it } from "vitest";
import {
  PlanReplayError,
  buildScenarioPreset,
  compileTopologyWorkloadPlan,
  replayNodeFailoverCampaign,
  runNodeFailoverCampaign,
  targetOnlyTopologyProfile,
} from "../src/index.js";

function failoverFixture() {
  const failedScenario = buildScenarioPreset("multi-node");
  const failedPlan = compileTopologyWorkloadPlan(
    failedScenario,
    targetOnlyTopologyProfile(2),
  );
  const recoveryBase = buildScenarioPreset("single-gpu-cpu");
  const recoveryScenario = {
    ...recoveryBase,
    execution: {
      ...recoveryBase.execution,
      topologyEpoch: 1,
    },
  };
  const replannedPlan = compileTopologyWorkloadPlan(
    recoveryScenario,
    targetOnlyTopologyProfile(2),
  );
  return {
    failedScenario,
    failedPlan,
    request: {
      failedNodeId: "node1",
      faultAtNs: 1,
      reason: "node1 heartbeat expired",
      recoveryScenario,
      replannedPlan,
    },
  };
}

describe("node failover recovery", () => {
  it("quiesces the old epoch before admitting an explicit failover plan", () => {
    const fixture = failoverFixture();
    const first = runNodeFailoverCampaign(
      fixture.failedScenario,
      fixture.failedPlan,
      fixture.request,
    );
    const second = runNodeFailoverCampaign(
      fixture.failedScenario,
      fixture.failedPlan,
      fixture.request,
    );

    expect(first).toEqual(second);
    expect(first.failedExecution.status).toBe("failed");
    expect(first.failedExecution.trace.terminal.fault).toMatchObject({
      kind: "node_failure",
      nodeId: "node1",
      atNs: 1,
    });
    expect(
      first.failedExecution.rankStates.some(
        (state) => state.status === "failed",
      ),
    ).toBe(true);
    expect(first.recoveryExecution.status).toBe("succeeded");
    expect(first.handoff).toMatchObject({
      oldTopologyEpoch: 0,
      newTopologyEpoch: 1,
      oldExecutionQuiescedAtNs: first.failedExecution.completedAtNs,
      recoveryAdmittedAtNs: first.failedExecution.completedAtNs,
    });
    expect(first.completedAtNs).toBe(
      first.failedExecution.completedAtNs
        + first.recoveryExecution.completedAtNs,
    );
    expect(replayNodeFailoverCampaign(
      fixture.failedScenario,
      fixture.failedPlan,
      fixture.request,
      first,
    ).completedAtNs).toBe(first.completedAtNs);
  });

  it("rejects recovery before quiescence and reuse of the failed node", () => {
    const fixture = failoverFixture();
    const result = runNodeFailoverCampaign(
      fixture.failedScenario,
      fixture.failedPlan,
      fixture.request,
    );
    const early = {
      ...result,
      handoff: {
        ...result.handoff,
        recoveryAdmittedAtNs: result.handoff.recoveryAdmittedAtNs - 1,
      },
    };
    expect(() => replayNodeFailoverCampaign(
      fixture.failedScenario,
      fixture.failedPlan,
      fixture.request,
      early,
    )).toThrowError(PlanReplayError);
    expect(() => replayNodeFailoverCampaign(
      fixture.failedScenario,
      fixture.failedPlan,
      fixture.request,
      early,
    )).toThrowError("handoff does not match old-epoch quiescence");

    const reusedNodeScenario = {
      ...fixture.failedScenario,
      execution: {
        ...fixture.failedScenario.execution,
        topologyEpoch: 1,
      },
    };
    const reusedNodePlan = compileTopologyWorkloadPlan(
      reusedNodeScenario,
      targetOnlyTopologyProfile(2),
    );
    expect(() => runNodeFailoverCampaign(
      fixture.failedScenario,
      fixture.failedPlan,
      {
        ...fixture.request,
        recoveryScenario: reusedNodeScenario,
        replannedPlan: {
          ...reusedNodePlan,
          executionId: `${reusedNodePlan.executionId}:recovery`,
        },
      },
    )).toThrowError("failover scenario still contains failed node node1");
  });

  it("rejects a stale epoch or reused execution identity", () => {
    const fixture = failoverFixture();
    expect(() => runNodeFailoverCampaign(
      fixture.failedScenario,
      fixture.failedPlan,
      {
        ...fixture.request,
        recoveryScenario: {
          ...fixture.request.recoveryScenario,
          execution: {
            ...fixture.request.recoveryScenario.execution,
            topologyEpoch: 0,
          },
        },
      },
    )).toThrowError("must use a newer matching topology epoch");
    expect(() => runNodeFailoverCampaign(
      fixture.failedScenario,
      fixture.failedPlan,
      {
        ...fixture.request,
        replannedPlan: {
          ...fixture.request.replannedPlan,
          executionId: fixture.failedPlan.executionId,
        },
      },
    )).toThrowError("recovery must use a new execution id");
  });
});
