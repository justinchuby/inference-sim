import { describe, expect, it } from "vitest";
import {
  PLAN_CONTRACT_REVISION,
  CollectiveSubmitSequencer,
  FrozenPlanExecutionError,
  FrozenPlanValidationError,
  PlanReplayError,
  assertValidFrozenPlan,
  buildScenarioPreset,
  executeFrozenPlan,
  replayPlanTrace,
  validateFrozenPlan,
  type FrozenPlan,
  type PlanStep,
  type PlanTraceEvent,
} from "../src/index.js";

function validPlan(): FrozenPlan {
  const steps: readonly PlanStep[] = [
    {
      id: 0,
      participants: ["rank-0"],
      dependencies: [],
      reads: ["weights-0"],
      writes: ["kv-0"],
      operation: {
        kind: "compute",
        deviceId: "node0:gpu0",
        capability: "attention",
        durationNs: 10,
      },
    },
    {
      id: 1,
      participants: ["rank-1"],
      dependencies: [],
      reads: ["weights-1"],
      writes: ["kv-1"],
      operation: {
        kind: "compute",
        deviceId: "node0:gpu1",
        capability: "attention",
        durationNs: 20,
      },
    },
    {
      id: 2,
      participants: ["rank-0", "rank-1"],
      dependencies: [0, 1],
      reads: ["kv-0", "kv-1"],
      writes: [],
      operation: {
        kind: "collective",
        groupId: "tp",
        commSequenceId: 0,
        linkIds: ["node0:nvlink:forward", "node0:nvlink:reverse"],
        durationNs: 5,
      },
    },
    {
      id: 3,
      participants: ["rank-0"],
      dependencies: [2],
      reads: ["kv-0"],
      writes: [],
      operation: {
        kind: "compute",
        deviceId: "node0:gpu0",
        capability: "attention",
        durationNs: 7,
      },
    },
  ];
  return {
    contractRevision: PLAN_CONTRACT_REVISION,
    id: "decode-plan",
    executionId: "execution-1",
    topologyEpoch: 0,
    steps,
  };
}

describe("FrozenPlan validation and execution", () => {
  it("executes independent ranks concurrently and replays the exact trace", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const plan = validPlan();

    expect(validateFrozenPlan(scenario, plan)).toEqual({
      valid: true,
      issues: [],
      topologicalStepIds: [0, 1, 2, 3],
    });
    const result = executeFrozenPlan(scenario, plan);

    expect(result.status).toBe("succeeded");
    expect(result.completedAtNs).toBe(32);
    expect(result.trace.operations.map((event) => [
      event.stepId,
      event.startNs,
      event.finishNs,
    ])).toEqual([
      [0, 0, 10],
      [1, 0, 20],
      [2, 20, 25],
      [3, 25, 32],
    ]);
    expect(result.rankCompletions).toEqual([
      { rankId: "rank-0", completedAtNs: 32 },
      { rankId: "rank-1", completedAtNs: 25 },
    ]);

    expect(replayPlanTrace(scenario, plan, result.trace)).toEqual({
      status: "succeeded",
      appliedEvents: 5,
      completedAtNs: 32,
      rankCompletions: result.rankCompletions,
      rankStates: result.rankStates,
    });
  });

  it("rejects unordered write/write allocation leases", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const base = validPlan();
    const plan: FrozenPlan = {
      ...base,
      steps: base.steps.map((step) => (
        step.id === 1 ? { ...step, writes: ["kv-0"] } : step
      )),
    };

    const result = validateFrozenPlan(scenario, plan);
    expect(result.valid).toBe(false);
    expect(
      result.issues.some((issue) => issue.code === "unordered_lease_conflict"),
    ).toBe(true);
    expect(() => assertValidFrozenPlan(scenario, plan)).toThrowError(
      FrozenPlanValidationError,
    );
  });

  it("rejects compute access to another device's private memory domain", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const base = validPlan();
    const plan: FrozenPlan = {
      ...base,
      steps: base.steps.map((step) => (
        step.id === 1 ? { ...step, reads: ["weights-0"] } : step
      )),
    };

    expect(
      validateFrozenPlan(scenario, plan).issues.some(
        (issue) => issue.code === "inaccessible_allocation",
      ),
    ).toBe(true);
  });

  it("rejects dependency cycles", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const base = validPlan();
    const plan: FrozenPlan = {
      ...base,
      steps: base.steps.map((step) => (
        step.id === 0 ? { ...step, dependencies: [3] } : step
      )),
    };

    expect(
      validateFrozenPlan(scenario, plan).issues.some(
        (issue) => issue.code === "dependency_cycle",
      ),
    ).toBe(true);
  });

  it("requires contiguous, dependency-ordered collectives per group", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const base = validPlan();
    const gap: FrozenPlan = {
      ...base,
      steps: [
        ...base.steps,
        {
          id: 4,
          participants: ["rank-0", "rank-1"],
          dependencies: [2],
          reads: [],
          writes: [],
          operation: {
            kind: "collective",
            groupId: "tp",
            commSequenceId: 2,
            linkIds: ["node0:nvlink:forward", "node0:nvlink:reverse"],
            durationNs: 1,
          },
        },
      ],
    };
    expect(
      validateFrozenPlan(scenario, gap).issues.some(
        (issue) => issue.code === "collective_sequence_gap",
      ),
    ).toBe(true);

    const unordered: FrozenPlan = {
      ...base,
      steps: [
        ...base.steps,
        {
          id: 4,
          participants: ["rank-0", "rank-1"],
          dependencies: [0, 1],
          reads: [],
          writes: [],
          operation: {
            kind: "collective",
            groupId: "tp",
            commSequenceId: 1,
            linkIds: ["node0:nvlink:forward", "node0:nvlink:reverse"],
            durationNs: 1,
          },
        },
      ],
    };
    expect(
      validateFrozenPlan(scenario, unordered).issues.some(
        (issue) => issue.code === "collective_order_dependency",
      ),
    ).toBe(true);
  });

  it("rejects the shortest trace prefix that violates a dependency", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const plan = validPlan();
    const result = executeFrozenPlan(scenario, plan);
    const operations: PlanTraceEvent[] = result.trace.operations.map(
      (event) => ({ ...event }),
    );
    operations[2] = {
      ...operations[2],
      submittedAtNs: 19,
      startNs: 19,
      finishNs: 24,
    };
    const trace = { ...result.trace, operations };

    expect(() => replayPlanTrace(scenario, plan, trace)).toThrowError(
      PlanReplayError,
    );
    expect(() => replayPlanTrace(scenario, plan, trace)).toThrowError(
      "event 2: step 2 submits before dependency 1 finishes",
    );
  });

  it("rejects unexplained scheduler delay even when intervals remain valid", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const plan = validPlan();
    const result = executeFrozenPlan(scenario, plan);
    const operations: PlanTraceEvent[] = result.trace.operations.map(
      (event) => ({ ...event }),
    );
    operations[3] = {
      ...operations[3],
      startNs: 26,
      finishNs: 33,
    };
    const trace = { ...result.trace, operations };

    expect(() => replayPlanTrace(scenario, plan, trace)).toThrowError(
      "event 3: step 3 expected deterministic start at 25ns",
    );
  });

  it("serializes independent compute steps on a one-lane device", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const base = validPlan();
    const plan: FrozenPlan = {
      ...base,
      steps: [
        base.steps[0],
        {
          id: 1,
          participants: ["rank-0"],
          dependencies: [],
          reads: ["weights-0"],
          writes: [],
          operation: {
            kind: "compute",
            deviceId: "node0:gpu0",
            capability: "attention",
            durationNs: 6,
          },
        },
      ],
    };

    const result = executeFrozenPlan(scenario, plan);
    expect(
      result.trace.operations.map((event) => [event.startNs, event.finishNs]),
    ).toEqual([
      [0, 10],
      [10, 16],
    ]);
  });

  it("submits all ready roots before a listed dependent becomes ready", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const plan: FrozenPlan = {
      contractRevision: PLAN_CONTRACT_REVISION,
      id: "causal-submission",
      executionId: "execution-causal",
      topologyEpoch: 0,
      steps: [
        {
          id: 0,
          participants: ["rank-0"],
          dependencies: [],
          reads: ["weights-0"],
          writes: ["kv-0"],
          operation: {
            kind: "compute",
            deviceId: "node0:gpu0",
            capability: "attention",
            durationNs: 10,
          },
        },
        {
          id: 1,
          participants: ["rank-0"],
          dependencies: [0],
          reads: ["kv-0"],
          writes: [],
          operation: {
            kind: "compute",
            deviceId: "node0:gpu0",
            capability: "attention",
            durationNs: 1,
          },
        },
        {
          id: 2,
          participants: ["rank-1"],
          dependencies: [],
          reads: ["weights-1"],
          writes: ["kv-1"],
          operation: {
            kind: "compute",
            deviceId: "node0:gpu1",
            capability: "attention",
            durationNs: 100,
          },
        },
      ],
    };

    expect(validateFrozenPlan(scenario, plan).topologicalStepIds).toEqual([
      0,
      1,
      2,
    ]);
    const result = executeFrozenPlan(scenario, plan);
    expect(
      result.trace.operations.map((event) => [
        event.stepId,
        event.submittedAtNs,
        event.startNs,
      ]),
    ).toEqual([
      [0, 0, 0],
      [2, 0, 0],
      [1, 10, 10],
    ]);
    expect(replayPlanTrace(scenario, plan, result.trace).status).toBe(
      "succeeded",
    );
  });

  it("contends collectives and point-to-point transfers on the same link", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const plan: FrozenPlan = {
      contractRevision: PLAN_CONTRACT_REVISION,
      id: "link-contention",
      executionId: "execution-links",
      topologyEpoch: 0,
      steps: [
        {
          id: 0,
          participants: ["rank-0", "rank-1"],
          dependencies: [],
          reads: ["weights-0"],
          writes: ["weights-1"],
          operation: {
            kind: "transfer",
            linkId: "node0:nvlink:forward",
            durationNs: 10,
          },
        },
        {
          id: 1,
          participants: ["rank-0", "rank-1"],
          dependencies: [],
          reads: [],
          writes: [],
          operation: {
            kind: "collective",
            groupId: "tp",
            commSequenceId: 0,
            linkIds: ["node0:nvlink:forward", "node0:nvlink:reverse"],
            durationNs: 5,
          },
        },
      ],
    };

    const result = executeFrozenPlan(scenario, plan);
    expect(
      result.trace.operations.map((event) => [event.startNs, event.finishNs]),
    ).toEqual([
      [0, 10],
      [10, 15],
    ]);
    expect(
      result.trace.operations[1].resources.map(
        (resource) => resource.resourceId,
      ),
    ).toEqual([
      "collective:tp",
      "link:node0:nvlink:forward",
      "link:node0:nvlink:reverse",
    ]);
    expect(replayPlanTrace(scenario, plan, result.trace).completedAtNs).toBe(15);
  });

  it("closes submission on failure and terminalizes only after quiescence", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const plan = validPlan();
    const result = executeFrozenPlan(scenario, plan, {
      injectTerminalBeforeStep: {
        stepId: 1,
        status: "failed",
        reason: "injected device failure",
      },
    });

    expect(result.status).toBe("failed");
    expect(result.trace.operations.map((event) => event.stepId)).toEqual([0]);
    expect(result.trace.terminal).toMatchObject({
      sourceSequence: 1,
      status: "failed",
      beforeStepId: 1,
      failureAtNs: 0,
      timestampNs: 10,
      reason: "injected device failure",
    });
    expect(result.rankStates).toEqual([
      { rankId: "rank-0", status: "aborted", terminalAtNs: 10 },
      { rankId: "rank-1", status: "failed", terminalAtNs: 0 },
    ]);
    expect(result.rankCompletions).toEqual([]);
    expect(replayPlanTrace(scenario, plan, result.trace)).toEqual({
      status: "failed",
      appliedEvents: 2,
      completedAtNs: 10,
      rankCompletions: [],
      rankStates: result.rankStates,
    });
  });

  it("preserves rank-local success when another rank is aborted", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const plan = validPlan();
    const result = executeFrozenPlan(scenario, plan, {
      injectTerminalBeforeStep: {
        stepId: 3,
        status: "aborted",
        reason: "request cancelled",
      },
    });

    expect(result.status).toBe("aborted");
    expect(result.rankStates).toEqual([
      { rankId: "rank-0", status: "aborted", terminalAtNs: 25 },
      { rankId: "rank-1", status: "succeeded", terminalAtNs: 25 },
    ]);
    expect(result.rankCompletions).toEqual([
      { rankId: "rank-1", completedAtNs: 25 },
    ]);
    expect(replayPlanTrace(scenario, plan, result.trace).rankStates).toEqual(
      result.rankStates,
    );
  });

  it("rejects a corrupted terminal rank state", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const plan = validPlan();
    const result = executeFrozenPlan(scenario, plan);
    const trace = {
      ...result.trace,
      terminal: {
        ...result.trace.terminal,
        rankStates: result.trace.terminal.rankStates.map((state, index) => (
          index === 0 ? { ...state, terminalAtNs: state.terminalAtNs + 1 } : state
        )),
      },
    };

    expect(() => replayPlanTrace(scenario, plan, trace)).toThrowError(
      "terminal: terminal rank states do not match rank-local execution state",
    );
  });
});

describe("CollectiveSubmitSequencer", () => {
  it("prevents a later execution from overtaking the current group owner", () => {
    const sequencer = new CollectiveSubmitSequencer();
    sequencer.registerExecution("execution-1", { tp: 2, ep: 1 });
    sequencer.registerExecution("execution-2", { tp: 1 });

    expect(() => sequencer.submit("execution-2", "tp", 0)).toThrowError(
      FrozenPlanExecutionError,
    );
    sequencer.submit("execution-1", "tp", 0);
    expect(() => sequencer.submit("execution-1", "tp", 0)).toThrowError(
      "collective sequence 0 does not match 1",
    );
    sequencer.submit("execution-1", "tp", 1);
    sequencer.submit("execution-2", "tp", 0);
    sequencer.submit("execution-1", "ep", 0);
    sequencer.completeExecution("execution-1");
    sequencer.completeExecution("execution-2");
  });
});
