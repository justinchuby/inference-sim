import { describe, expect, it } from "vitest";
import {
  PLAN_CONTRACT_REVISION,
  SCENARIO_PRESET_NAMES,
  PlanReplayError,
  StreamingConcurrentPlanRuntime,
  buildScenarioPreset,
  compileTopologyWorkloadPlan,
  executeConcurrentFrozenPlans,
  executeConcurrentNodeFailure,
  replayConcurrentPlanTrace,
  runSeededConcurrentPlanCampaign,
  targetOnlyTopologyProfile,
  type ConcurrentPlanRequest,
  type FrozenPlan,
  type PlanStep,
} from "../src/index.js";

function plan(executionId: string): FrozenPlan {
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
        algorithm: "all_reduce_ring",
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
    id: `decode-${executionId}`,
    executionId,
    topologyEpoch: 0,
    steps,
  };
}

function requests(): readonly ConcurrentPlanRequest[] {
  return [
    { plan: plan("execution-a"), arrivalNs: 0, admissionOrder: 0 },
    { plan: plan("execution-b"), arrivalNs: 0, admissionOrder: 1 },
  ];
}

function oneStepPlan(executionId: string, durationNs = 10): FrozenPlan {
  return {
    contractRevision: PLAN_CONTRACT_REVISION,
    id: `one-step-${executionId}`,
    executionId,
    topologyEpoch: 0,
    steps: [{
      id: 0,
      participants: ["rank-0"],
      dependencies: [],
      reads: ["weights-0"],
      writes: ["kv-0"],
      operation: {
        kind: "compute",
        deviceId: "node0:gpu0",
        capability: "attention",
        durationNs,
      },
    }],
  };
}

describe("concurrent FrozenPlan execution", () => {
  it("admits the next foreground while prior background work remains live", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const first: FrozenPlan = {
      ...oneStepPlan("execution-a"),
      steps: [
        oneStepPlan("execution-a").steps[0],
        {
          id: 1,
          participants: ["rank-0"],
          dependencies: [0],
          reads: ["weights-0"],
          writes: ["kv-0"],
          operation: {
            kind: "compute",
            deviceId: "node0:gpu0",
            capability: "attention",
            durationNs: 100,
          },
        },
      ],
    };
    const second = oneStepPlan("execution-b");
    const runtime = new StreamingConcurrentPlanRuntime(scenario);

    runtime.admit(first, 0);
    expect(runtime.runUntilStep("execution-a", 0)).toEqual({
      executionId: "execution-a",
      stepId: 0,
      completedAtNs: 10,
    });
    runtime.admit(second, 10);
    expect(runtime.runUntilStep("execution-b", 0).completedAtNs).toBe(120);
    const result = runtime.drain();
    const requests = [
      { plan: first, arrivalNs: 0, admissionOrder: 0 },
      { plan: second, arrivalNs: 10, admissionOrder: 1 },
    ];

    expect(result.completedAtNs).toBe(120);
    expect(result.maximumConcurrentExecutions).toBe(2);
    expect(result.trace.operations.map(({ event }) => [
      event.executionId,
      event.stepId,
      event.startNs,
      event.finishNs,
    ])).toEqual([
      ["execution-a", 0, 0, 10],
      ["execution-a", 1, 10, 110],
      ["execution-b", 0, 110, 120],
    ]);
    expect(replayConcurrentPlanTrace(scenario, requests, result.trace)
      .completedAtNs).toBe(120);
  });

  it("shares resource lanes and communicator ownership deterministically", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const first = executeConcurrentFrozenPlans(scenario, requests());
    const second = executeConcurrentFrozenPlans(scenario, requests());

    expect(first).toEqual(second);
    expect(first.completedAtNs).toBe(59);
    expect(first.maximumConcurrentExecutions).toBe(2);
    expect(first.trace.operations.map(({ event }) => [
      event.executionId,
      event.stepId,
      event.submittedAtNs,
      event.startNs,
      event.finishNs,
    ])).toEqual([
      ["execution-a", 0, 0, 0, 10],
      ["execution-a", 1, 0, 0, 20],
      ["execution-b", 0, 0, 10, 20],
      ["execution-b", 1, 0, 20, 40],
      ["execution-a", 2, 20, 40, 45],
      ["execution-b", 2, 40, 45, 50],
      ["execution-a", 3, 45, 45, 52],
      ["execution-b", 3, 50, 52, 59],
    ]);
    expect(
      replayConcurrentPlanTrace(scenario, requests(), first.trace),
    ).toEqual({
      appliedEvents: 12,
      completedAtNs: 59,
      maximumConcurrentExecutions: 2,
      executions: first.executions.map((execution) => ({
        status: "succeeded",
        appliedEvents: 5,
        completedAtNs: execution.completedAtNs,
        rankCompletions: execution.rankCompletions,
        rankStates: execution.rankStates,
      })),
    });
  });

  it("uses admission order rather than caller array order for equal arrivals", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const unordered = [
      { plan: plan("execution-a"), arrivalNs: 0, admissionOrder: 1 },
      { plan: plan("execution-b"), arrivalNs: 0, admissionOrder: 0 },
    ];
    const result = executeConcurrentFrozenPlans(scenario, unordered);

    expect(result.trace.admissions.map((entry) => entry.executionId)).toEqual([
      "execution-b",
      "execution-a",
    ]);
    expect(result.trace.operations[0].event.executionId).toBe("execution-b");
    expect(
      replayConcurrentPlanTrace(scenario, [...unordered].reverse(), result.trace)
        .completedAtNs,
    ).toBe(result.completedAtNs);
  });

  it("rejects an admission order that contradicts arrival order", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    expect(() => executeConcurrentFrozenPlans(scenario, [
      { plan: plan("execution-a"), arrivalNs: 10, admissionOrder: 0 },
      { plan: plan("execution-b"), arrivalNs: 0, admissionOrder: 1 },
    ])).toThrowError(
      "admission order must be the contiguous arrival-ordered sequence 0..N-1",
    );
  });

  it("charges admissions and terminals to the scenario event budget", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const limited = {
      ...scenario,
      execution: {
        ...scenario.execution,
        maxEvents: 5,
      },
    };
    expect(() => executeConcurrentFrozenPlans(limited, [
      { plan: oneStepPlan("execution-a"), arrivalNs: 0, admissionOrder: 0 },
      { plan: oneStepPlan("execution-b"), arrivalNs: 0, admissionOrder: 1 },
    ])).toThrowError("require 6 runtime events, limit is 5");
  });

  it("enforces the same event budget for incremental streaming admission", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const limited = {
      ...scenario,
      execution: {
        ...scenario.execution,
        maxEvents: 5,
      },
    };
    const runtime = new StreamingConcurrentPlanRuntime(limited);

    runtime.admit(oneStepPlan("execution-a"), 0);
    expect(() => runtime.admit(oneStepPlan("execution-b"), 0))
      .toThrowError("require 6 runtime events, limit is 5");
  });

  it("releases constrained steps without reserving resources early", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const constrained = {
      plan: oneStepPlan("execution-a"),
      arrivalNs: 0,
      admissionOrder: 0,
      stepNotBeforeNs: { 0: 20 },
    };
    const unconstrained = {
      plan: oneStepPlan("execution-b"),
      arrivalNs: 0,
      admissionOrder: 1,
    };
    const concurrentRequests = [constrained, unconstrained];
    const result = executeConcurrentFrozenPlans(
      scenario,
      concurrentRequests,
    );

    expect(result.trace.operations.map(({ event }) => [
      event.executionId,
      event.submittedAtNs,
      event.startNs,
      event.finishNs,
    ])).toEqual([
      ["execution-b", 0, 0, 10],
      ["execution-a", 20, 20, 30],
    ]);
    expect(
      replayConcurrentPlanTrace(scenario, concurrentRequests, result.trace)
        .completedAtNs,
    ).toBe(30);
    expect(() => replayConcurrentPlanTrace(
      scenario,
      [{
        ...constrained,
        stepNotBeforeNs: { 0: 19 },
      }, unconstrained],
      result.trace,
    )).toThrowError(PlanReplayError);
  });

  it("processes completions before admissions at the same timestamp", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const zeroChain: FrozenPlan = {
      ...oneStepPlan("execution-a"),
      steps: [
        oneStepPlan("execution-a").steps[0],
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
            durationNs: 0,
          },
        },
        {
          id: 2,
          participants: ["rank-0"],
          dependencies: [1],
          reads: ["kv-0"],
          writes: [],
          operation: {
            kind: "compute",
            deviceId: "node0:gpu0",
            capability: "attention",
            durationNs: 0,
          },
        },
      ],
    };
    const boundaryRequests = [
      {
        plan: zeroChain,
        arrivalNs: 0,
        admissionOrder: 0,
      },
      {
        plan: oneStepPlan("execution-b"),
        arrivalNs: 10,
        admissionOrder: 1,
      },
    ];
    const result = executeConcurrentFrozenPlans(scenario, boundaryRequests);

    expect(result.maximumConcurrentExecutions).toBe(1);
    expect(result.trace.operations.map(({ event }) => [
      event.executionId,
      event.stepId,
      event.startNs,
      event.finishNs,
    ])).toEqual([
      ["execution-a", 0, 0, 10],
      ["execution-a", 1, 10, 10],
      ["execution-a", 2, 10, 10],
      ["execution-b", 0, 10, 20],
    ]);
    expect(
      replayConcurrentPlanTrace(scenario, boundaryRequests, result.trace)
        .maximumConcurrentExecutions,
    ).toBe(1);
  });

  it("serializes conflicting physical leases even across distinct resources", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const writer: FrozenPlan = {
      contractRevision: PLAN_CONTRACT_REVISION,
      id: "writer",
      executionId: "writer",
      topologyEpoch: 0,
      steps: [plan("writer").steps[0]],
    };
    const reader: FrozenPlan = {
      contractRevision: PLAN_CONTRACT_REVISION,
      id: "reader",
      executionId: "reader",
      topologyEpoch: 0,
      steps: [{
        id: 0,
        participants: ["rank-0", "rank-1"],
        dependencies: [],
        reads: ["kv-0"],
        writes: ["kv-1"],
        operation: {
          kind: "transfer",
          linkId: "node0:nvlink:forward",
          durationNs: 5,
        },
      }],
    };
    const concurrentRequests = [
      { plan: writer, arrivalNs: 0, admissionOrder: 0 },
      { plan: reader, arrivalNs: 0, admissionOrder: 1 },
    ];
    const result = executeConcurrentFrozenPlans(
      scenario,
      concurrentRequests,
    );

    expect(result.trace.operations[1].event).toMatchObject({
      executionId: "reader",
      submittedAtNs: 0,
      startNs: 10,
      finishNs: 15,
    });
    expect(
      replayConcurrentPlanTrace(scenario, concurrentRequests, result.trace)
        .completedAtNs,
    ).toBe(15);
  });

  it("rejects timing, resource, and collective-owner trace mutations", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const result = executeConcurrentFrozenPlans(scenario, requests());
    const bCollectiveIndex = result.trace.operations.findIndex(
      ({ event }) => event.executionId === "execution-b" && event.stepId === 2,
    );
    const timingMutation = {
      ...result.trace,
      operations: result.trace.operations.map((wrapper, index) => (
        index === bCollectiveIndex
          ? {
              ...wrapper,
              event: {
                ...wrapper.event,
                submittedAtNs: wrapper.event.submittedAtNs - 1,
                startNs: wrapper.event.startNs - 1,
                finishNs: wrapper.event.finishNs - 1,
              },
            }
          : wrapper
      )),
    };
    expect(() => (
      replayConcurrentPlanTrace(scenario, requests(), timingMutation)
    )).toThrowError(PlanReplayError);
    expect(() => (
      replayConcurrentPlanTrace(scenario, requests(), timingMutation)
    )).toThrowError("submitted at 39ns instead of 40ns");

    const resourceMutation = {
      ...result.trace,
      operations: result.trace.operations.map((wrapper, index) => (
        index === 2
          ? {
              ...wrapper,
              event: {
                ...wrapper.event,
                resources: [{
                  ...wrapper.event.resources[0],
                  resourceLane: 1,
                }],
              },
            }
          : wrapper
      )),
    };
    expect(() => (
      replayConcurrentPlanTrace(scenario, requests(), resourceMutation)
    )).toThrowError("resource reservations are not deterministic");

    const ownerMutation = {
      ...result.trace,
      operations: result.trace.operations.map((wrapper) => (
        wrapper.event.executionId === "execution-b"
          && wrapper.event.stepId === 2
          ? {
              ...wrapper,
              event: {
                ...wrapper.event,
                submittedAtNs: 20,
                startNs: 25,
                finishNs: 30,
              },
            }
          : wrapper
      )),
    };
    expect(() => (
      replayConcurrentPlanTrace(scenario, requests(), ownerMutation)
    )).toThrowError();

    const reordered = {
      ...result.trace,
      operations: [
        {
          ...result.trace.operations[1],
          globalSequence: 0,
          event: {
            ...result.trace.operations[1].event,
            sourceSequence: 0,
          },
        },
        {
          ...result.trace.operations[0],
          globalSequence: 1,
          event: {
            ...result.trace.operations[0].event,
            sourceSequence: 1,
          },
        },
        ...result.trace.operations.slice(2),
      ],
    };
    expect(() => (
      replayConcurrentPlanTrace(scenario, requests(), reordered)
    )).toThrowError("violates canonical ready-work arbitration");
  });

  it("runs a repeatable seeded large-scale campaign", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const options = {
      executionCount: 32,
      seed: 0x5eed,
      arrivalWindowNs: 100,
    };
    const first = runSeededConcurrentPlanCampaign(
      scenario,
      plan("campaign"),
      options,
    );
    const second = runSeededConcurrentPlanCampaign(
      scenario,
      plan("campaign"),
      options,
    );

    expect(first).toEqual(second);
    expect(first.execution.executions).toHaveLength(32);
    expect(first.execution.trace.operations).toHaveLength(128);
    expect(first.replay.appliedEvents).toBe(192);
    expect(first.replay.completedAtNs).toBe(first.execution.completedAtNs);
    expect(first.execution.maximumConcurrentExecutions).toBeGreaterThan(1);
    expect(new Set(
      first.requests.map((request) => request.admissionOrder),
    ).size).toBe(32);
  });

  it("executes and replays concurrent campaigns on all topology families", () => {
    for (const preset of SCENARIO_PRESET_NAMES) {
      const scenario = buildScenarioPreset(preset);
      const template = compileTopologyWorkloadPlan(
        scenario,
        targetOnlyTopologyProfile(2),
      );
      const campaign = runSeededConcurrentPlanCampaign(
        scenario,
        template,
        {
          executionCount: 4,
          seed: 0x5eed,
          arrivalWindowNs: 100_000,
        },
      );

      expect(campaign.execution.executions, preset).toHaveLength(4);
      expect(
        campaign.execution.executions.every(
          (execution) => execution.status === "succeeded",
        ),
        preset,
      ).toBe(true);
      expect(campaign.replay.completedAtNs, preset).toBe(
        campaign.execution.completedAtNs,
      );
      expect(campaign.replay.executions, preset).toHaveLength(4);
    }
  });

  it("fans one node fault out to every admitted old-epoch execution", () => {
    const scenario = buildScenarioPreset("multi-node");
    const template = compileTopologyWorkloadPlan(
      scenario,
      targetOnlyTopologyProfile(2),
    );
    const concurrentRequests = Array.from({ length: 4 }, (_, index) => ({
      plan: {
        ...template,
        id: `${template.id}:node-fault:${index}`,
        executionId: `${template.executionId}:node-fault:${index}`,
      },
      arrivalNs: 0,
      admissionOrder: index,
    }));
    const fault = {
      kind: "node_failure" as const,
      atNs: 1,
      nodeId: "node1",
      reason: "node1 heartbeat expired",
    };
    const result = executeConcurrentNodeFailure(
      scenario,
      concurrentRequests,
      fault,
    );

    expect(result.executions).toHaveLength(4);
    expect(
      result.executions.every((execution) => execution.status === "failed"),
    ).toBe(true);
    expect(
      result.trace.operations.every(({ event }) => (
        event.submittedAtNs < fault.atNs
      )),
    ).toBe(true);
    expect(result.trace.terminals.every((terminal) => (
      terminal.fault?.kind === "node_failure"
      && terminal.fault.nodeId === "node1"
    ))).toBe(true);
    const replay = replayConcurrentPlanTrace(
      scenario,
      concurrentRequests,
      result.trace,
      { nodeFailure: fault },
    );
    expect(replay.completedAtNs).toBe(result.completedAtNs);
    expect(replay.executions.every(
      (execution) => execution.status === "failed",
    )).toBe(true);

    const operations = result.trace.operations.slice(0, -1);
    expect(() => replayConcurrentPlanTrace(
      scenario,
      concurrentRequests,
      { ...result.trace, operations },
      { nodeFailure: fault },
    )).toThrowError("was ready before node fault but omitted");
  });

  it("does not admit an execution at the node-fault timestamp", () => {
    const scenario = buildScenarioPreset("multi-node");
    const template = compileTopologyWorkloadPlan(
      scenario,
      targetOnlyTopologyProfile(1),
    );
    expect(() => executeConcurrentNodeFailure(
      scenario,
      [{
        plan: template,
        arrivalNs: 1,
        admissionOrder: 0,
      }],
      {
        kind: "node_failure",
        atNs: 1,
        nodeId: "node1",
        reason: "node1 failed",
      },
    )).toThrowError("is not admitted before node fault");

    const admitted = [{
      plan: template,
      arrivalNs: 0,
      admissionOrder: 0,
    }];
    const baseline = executeConcurrentFrozenPlans(scenario, admitted);
    expect(() => executeConcurrentNodeFailure(
      scenario,
      admitted,
      {
        kind: "node_failure",
        atNs: baseline.completedAtNs + 1,
        nodeId: "node1",
        reason: "late node failure",
      },
    )).toThrowError("node fault occurs after executions completed");
  });
});
