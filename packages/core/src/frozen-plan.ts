import {
  PLAN_CONTRACT_REVISION,
  type FrozenPlan,
  type FrozenPlanExecutionResult,
  type PlanOperation,
  type PlanExecutionOptions,
  type PlanExecutionTrace,
  type PlanResourceReservation,
  type PlanReplayResult,
  type PlanStep,
  type PlanTerminalEvent,
  type PlanTerminalStatus,
  type PlanTraceEvent,
  type PlanValidationIssue,
  type PlanValidationResult,
  type RankCompletion,
  type RankTerminalState,
} from "./plan-types.js";
import type {
  CommunicatorGroupSpec,
  SimDeviceSpec,
  SimLinkSpec,
  SimulationScenario,
} from "./scenario-types.js";
import { validateScenario } from "./scenario.js";
import { DiscreteEventSimulator } from "./event-loop.js";

export class FrozenPlanValidationError extends Error {
  readonly issues: readonly PlanValidationIssue[];

  constructor(issues: readonly PlanValidationIssue[]) {
    super(
      `frozen plan validation failed with ${issues.length} issue(s): ${
        issues.map((issue) => `${issue.path}: ${issue.message}`).join("; ")
      }`,
    );
    this.name = "FrozenPlanValidationError";
    this.issues = issues;
  }
}

export class FrozenPlanExecutionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FrozenPlanExecutionError";
  }
}

export class PlanReplayError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PlanReplayError";
  }
}

export function assertValidFrozenPlan(
  scenario: SimulationScenario,
  plan: FrozenPlan,
): void {
  const result = validateFrozenPlan(scenario, plan);
  if (!result.valid) {
    throw new FrozenPlanValidationError(result.issues);
  }
}

export function validateFrozenPlan(
  scenario: SimulationScenario,
  plan: FrozenPlan,
): PlanValidationResult {
  const issues: PlanValidationIssue[] = [];
  const add = (code: string, path: string, message: string): void => {
    issues.push({ code, path, message });
  };

  const scenarioResult = validateScenario(scenario);
  for (const issue of scenarioResult.issues) {
    add(`scenario.${issue.code}`, `scenario.${issue.path}`, issue.message);
  }
  if (plan.contractRevision !== PLAN_CONTRACT_REVISION) {
    add(
      "contract_revision",
      "contractRevision",
      `expected ${PLAN_CONTRACT_REVISION}, got ${plan.contractRevision}`,
    );
  }
  if (plan.id.length === 0 || plan.executionId.length === 0) {
    add("empty_id", "id", "plan and execution ids must not be empty");
  }
  if (plan.topologyEpoch !== scenario.execution.topologyEpoch) {
    add(
      "topology_epoch",
      "topologyEpoch",
      `plan epoch ${plan.topologyEpoch} does not match scenario epoch ${scenario.execution.topologyEpoch}`,
    );
  }
  if (plan.steps.length === 0) {
    add("empty_plan", "steps", "plan must contain at least one step");
  }

  const devices = new Map(scenario.devices.map((device) => [device.id, device]));
  const links = new Map(scenario.links.map((link) => [link.id, link]));
  const groups = new Map(scenario.groups.map((group) => [group.id, group]));
  const rankDevices = buildRankDeviceMap(scenario.groups);
  const allocations = new Map(
    scenario.placements.flatMap((placement) => (
      placement.allocations.map((allocation) => [
        allocation.physicalAllocationId,
        allocation.domainId,
      ] as const)
    )),
  );
  const stepMap = new Map<number, PlanStep>();

  for (const [index, step] of plan.steps.entries()) {
    const path = `steps[${index}]`;
    if (!Number.isSafeInteger(step.id) || step.id < 0 || stepMap.has(step.id)) {
      add("duplicate_step", `${path}.id`, `invalid or duplicate step id ${step.id}`);
    } else {
      stepMap.set(step.id, step);
    }
    validateUniqueStrings(step.participants, `${path}.participants`, add);
    if (step.participants.length === 0) {
      add("empty_participants", `${path}.participants`, "must not be empty");
    }
    for (const rankId of step.participants) {
      if (!rankDevices.has(rankId)) {
        add(
          "unknown_rank",
          `${path}.participants`,
          `unknown rank ${rankId}`,
        );
      }
    }
    validateUniqueNumbers(step.dependencies, `${path}.dependencies`, add);
    validateUniqueStrings(step.reads, `${path}.reads`, add);
    validateUniqueStrings(step.writes, `${path}.writes`, add);
    for (const allocationId of [...step.reads, ...step.writes]) {
      if (!allocations.has(allocationId)) {
        add(
          "unknown_allocation",
          path,
          `unknown physical allocation ${allocationId}`,
        );
      }
    }
    const writeSet = new Set(step.writes);
    for (const allocationId of step.reads) {
      if (writeSet.has(allocationId)) {
        add(
          "read_write_overlap",
          path,
          `${allocationId} appears in both reads and writes; in-place access must be exclusive`,
        );
      }
    }
    validateOperation(
      step,
      path,
      devices,
      links,
      groups,
      rankDevices,
      allocations,
      add,
    );
  }

  for (const [index, step] of plan.steps.entries()) {
    for (const dependency of step.dependencies) {
      if (!stepMap.has(dependency)) {
        add(
          "unknown_dependency",
          `steps[${index}].dependencies`,
          `unknown step ${dependency}`,
        );
      }
      if (dependency === step.id) {
        add(
          "self_dependency",
          `steps[${index}].dependencies`,
          "step cannot depend on itself",
        );
      }
    }
  }

  const topologicalStepIds = topologicalSort(plan.steps, stepMap);
  if (topologicalStepIds.length !== stepMap.size) {
    add("dependency_cycle", "steps", "step dependencies contain a cycle");
  }

  if (topologicalStepIds.length === stepMap.size) {
    const reachability = buildReachability(plan.steps, stepMap);
    validateLeaseOrdering(plan.steps, reachability, add);
    validateCollectiveOrdering(plan.steps, groups, reachability, add);
  }

  return {
    valid: issues.length === 0,
    issues,
    topologicalStepIds,
  };
}

export function executeFrozenPlan(
  scenario: SimulationScenario,
  plan: FrozenPlan,
  options: PlanExecutionOptions = {},
): FrozenPlanExecutionResult {
  const validation = validateFrozenPlan(scenario, plan);
  if (!validation.valid) {
    throw new FrozenPlanValidationError(validation.issues);
  }

  const stepMap = new Map(plan.steps.map((step) => [step.id, step]));
  const resourceLanes = buildResourceLanes(scenario);
  const trace: PlanTraceEvent[] = [];
  const rankCompletions = new Map<string, number>();
  const leaseTimeline = new AllocationLeaseTimeline();
  const scheduledStepIds = new Set<number>();
  const simulator = new DiscreteEventSimulator<{ readonly stepId: number }>();
  const planOrder = new Map(plan.steps.map((step, index) => [step.id, index]));
  const remainingDependencies = new Map(
    plan.steps.map((step) => [step.id, step.dependencies.length]),
  );
  const dependents = new Map<number, number[]>();
  for (const step of plan.steps) {
    for (const dependency of step.dependencies) {
      const targets = dependents.get(dependency) ?? [];
      targets.push(step.id);
      dependents.set(dependency, targets);
    }
  }
  const ready = plan.steps
    .filter((step) => step.dependencies.length === 0)
    .map((step) => step.id);
  const injection = options.injectTerminalBeforeStep;
  if (
    injection
    && (
      !validation.topologicalStepIds.includes(injection.stepId)
      || (injection.status !== "failed" && injection.status !== "aborted")
      || injection.reason.length === 0
    )
  ) {
    throw new FrozenPlanExecutionError(
      `invalid terminal injection before step ${injection.stepId}`,
    );
  }

  let terminalInjection:
    | {
        status: "failed" | "aborted";
        step: PlanStep;
        failureAtNs: number;
        reason: string;
      }
    | undefined;

  const sortReady = (): void => {
    ready.sort((left, right) => (
      (planOrder.get(left) ?? 0) - (planOrder.get(right) ?? 0)
    ));
  };
  const submitReady = (submittedAtNs: number): void => {
    sortReady();
    while (ready.length > 0 && !terminalInjection) {
      const stepId = ready.shift();
      const step = stepId === undefined ? undefined : stepMap.get(stepId);
      if (!step) {
        throw new FrozenPlanExecutionError(
          `ready step ${String(stepId)} disappeared`,
        );
      }
      if (injection?.stepId === step.id) {
        terminalInjection = {
          status: injection.status,
          step,
          failureAtNs: submittedAtNs,
          reason: injection.reason,
        };
        ready.length = 0;
        return;
      }

      const resources = selectResourceReservations(
        step.operation,
        resourceLanes,
      );
      const resourceReady = Math.max(
        submittedAtNs,
        ...resources.map((reservation) => (
          resourceLanes.get(reservation.resourceId)?.[
            reservation.resourceLane
          ] ?? 0
        )),
      );
      const startNs = resourceReady;
      const finishNs = checkedTimeAdd(
        startNs,
        step.operation.durationNs,
        `step ${step.id}`,
      );
      leaseTimeline.reserve(step, startNs, finishNs);
      for (const reservation of resources) {
        const lanes = resourceLanes.get(reservation.resourceId);
        if (!lanes) {
          throw new FrozenPlanExecutionError(
            `missing resource ${reservation.resourceId}`,
          );
        }
        lanes[reservation.resourceLane] = finishNs;
      }
      scheduledStepIds.add(step.id);
      trace.push(
        traceEvent(
          plan,
          step,
          trace.length,
          submittedAtNs,
          startNs,
          finishNs,
          resources,
        ),
      );
      simulator.scheduleAt(finishNs, { stepId: step.id });
    }
  };

  submitReady(0);
  simulator.run((event, simulation) => {
    const step = stepMap.get(event.payload.stepId);
    if (!step) {
      throw new FrozenPlanExecutionError(
        `completion references unknown step ${event.payload.stepId}`,
      );
    }
    for (const rankId of step.participants) {
      rankCompletions.set(
        rankId,
        Math.max(rankCompletions.get(rankId) ?? 0, simulation.nowNs),
      );
    }
    if (terminalInjection) {
      return;
    }
    for (const dependentId of dependents.get(step.id) ?? []) {
      const remaining = (remainingDependencies.get(dependentId) ?? 0) - 1;
      remainingDependencies.set(dependentId, remaining);
      if (remaining === 0) {
        ready.push(dependentId);
      }
    }
    submitReady(simulation.nowNs);
  }, { maxEvents: scenario.execution.maxEvents });

  const completedAtNs = simulator.nowNs;
  if (terminalInjection) {
    const rankStates = deriveRankStates(
      plan,
      scheduledStepIds,
      rankCompletions,
      terminalInjection.status,
      terminalInjection.step,
      terminalInjection.failureAtNs,
      completedAtNs,
    );
    return {
      status: terminalInjection.status,
      executionId: plan.executionId,
      completedAtNs,
      trace: terminalTrace(
        plan,
        trace,
        terminalInjection.status,
        completedAtNs,
        rankStates,
        terminalInjection.step.id,
        terminalInjection.failureAtNs,
        terminalInjection.reason,
      ),
      rankCompletions: successfulRankCompletions(rankStates),
      rankStates,
    };
  }
  if (scheduledStepIds.size !== plan.steps.length) {
    throw new FrozenPlanExecutionError(
      `execution quiesced after scheduling ${scheduledStepIds.size}/${plan.steps.length} steps`,
    );
  }
  const rankStates = allSucceededRankStates(plan, rankCompletions);
  return {
    status: "succeeded",
    executionId: plan.executionId,
    completedAtNs,
    trace: terminalTrace(
      plan,
      trace,
      "succeeded",
      completedAtNs,
      rankStates,
    ),
    rankCompletions: sortedRankCompletions(rankCompletions),
    rankStates,
  };
}

/**
 * Independently validates a serialized plan trace. It does not call the
 * executor or its resource/lease timeline.
 */
export function replayPlanTrace(
  scenario: SimulationScenario,
  plan: FrozenPlan,
  trace: PlanExecutionTrace,
): PlanReplayResult {
  const validation = validateFrozenPlan(scenario, plan);
  if (!validation.valid) {
    throw new PlanReplayError("cannot replay an invalid FrozenPlan");
  }
  const operations = trace.operations;
  if (
    trace.terminal.status === "succeeded"
    && operations.length !== plan.steps.length
  ) {
    throw new PlanReplayError(
      `successful trace has ${operations.length} events for ${plan.steps.length} steps`,
    );
  }
  if (
    trace.terminal.status !== "succeeded"
    && operations.length >= plan.steps.length
  ) {
    throw new PlanReplayError("non-success terminal must stop before a plan step");
  }

  const stepMap = new Map(plan.steps.map((step) => [step.id, step]));
  const eventByStep = new Map<number, PlanTraceEvent>();
  const resourceIntervals = new Map<string, PlanTraceEvent[]>();
  const expectedResourceLanes = buildResourceLanes(scenario);
  const allocationIntervals = new Map<
    string,
    Array<{ event: PlanTraceEvent; mode: "read" | "write" }>
  >();
  const rankCompletions = new Map<string, number>();
  let lastSubmittedAtNs = 0;

  for (let index = 0; index < operations.length; index++) {
    const event = operations[index];
    try {
      if (event.contractRevision !== PLAN_CONTRACT_REVISION) {
        replayFail(`contract revision ${event.contractRevision} is unsupported`);
      }
      if (
        event.executionId !== plan.executionId
        || event.topologyEpoch !== plan.topologyEpoch
      ) {
        replayFail("execution identity or topology epoch mismatch");
      }
      if (event.sourceSequence !== index) {
        replayFail(`source sequence ${event.sourceSequence} does not match ${index}`);
      }
      const step = stepMap.get(event.stepId);
      if (!step || eventByStep.has(event.stepId)) {
        replayFail(`unknown or duplicate step ${event.stepId}`);
      }
      assertEventMatchesStep(step, event);
      if (
        !Number.isSafeInteger(event.submittedAtNs)
        || !Number.isSafeInteger(event.startNs)
        || !Number.isSafeInteger(event.finishNs)
        || event.submittedAtNs < 0
        || event.startNs < 0
        || event.finishNs < event.startNs
        || event.finishNs - event.startNs !== step.operation.durationNs
      ) {
        replayFail(`invalid timing for step ${step.id}`);
      }
      if (event.submittedAtNs < lastSubmittedAtNs) {
        replayFail(`step ${step.id} submission time moved backwards`);
      }
      lastSubmittedAtNs = event.submittedAtNs;
      for (const dependency of step.dependencies) {
        const dependencyEvent = eventByStep.get(dependency);
        if (
          !dependencyEvent
          || event.submittedAtNs < dependencyEvent.finishNs
        ) {
          replayFail(
            `step ${step.id} submits before dependency ${dependency} finishes`,
          );
        }
      }

      const expectedResources = selectResourceReservations(
        step.operation,
        expectedResourceLanes,
      );
      if (!resourceReservationsEqual(event.resources, expectedResources)) {
        replayFail(`step ${step.id} resource reservations are not deterministic`);
      }
      let dependencyReady = 0;
      for (const dependency of step.dependencies) {
        dependencyReady = Math.max(
          dependencyReady,
          eventByStep.get(dependency)?.finishNs ?? 0,
        );
      }
      if (event.submittedAtNs !== dependencyReady) {
        replayFail(
          `step ${step.id} submitted at ${event.submittedAtNs}ns instead of readiness ${dependencyReady}ns`,
        );
      }
      const expectedStart = Math.max(
        event.submittedAtNs,
        ...expectedResources.map((reservation) => (
          expectedResourceLanes.get(reservation.resourceId)?.[
            reservation.resourceLane
          ] ?? 0
        )),
      );
      if (event.startNs !== expectedStart) {
        replayFail(
          `step ${step.id} expected deterministic start at ${expectedStart}ns`,
        );
      }
      for (const reservation of expectedResources) {
        const laneCapacity = resourceCapacity(
          scenario,
          reservation.resourceId,
        );
        if (
          !Number.isSafeInteger(reservation.resourceLane)
          || reservation.resourceLane < 0
          || reservation.resourceLane >= laneCapacity
        ) {
          replayFail(
            `invalid lane ${reservation.resourceLane} for ${reservation.resourceId}`,
          );
        }
        const lanes = expectedResourceLanes.get(reservation.resourceId);
        if (!lanes) {
          replayFail(`unknown resource ${reservation.resourceId}`);
        }
        lanes[reservation.resourceLane] = event.finishNs;
        const laneKey =
          `${reservation.resourceId}#${reservation.resourceLane}`;
        const priorResourceEvents = resourceIntervals.get(laneKey) ?? [];
        if (priorResourceEvents.some((prior) => intervalsOverlap(prior, event))) {
          replayFail(`resource lane ${laneKey} overlaps`);
        }
        priorResourceEvents.push(event);
        resourceIntervals.set(laneKey, priorResourceEvents);
      }

      replayAllocationIntervals(allocationIntervals, event);
      eventByStep.set(step.id, event);
      for (const rankId of event.participants) {
        rankCompletions.set(
          rankId,
          Math.max(rankCompletions.get(rankId) ?? 0, event.finishNs),
        );
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new PlanReplayError(`event ${index}: ${message}`);
    }
  }

  const expectedNextStepId = trace.terminal.status === "succeeded"
    ? undefined
    : trace.terminal.beforeStepId;
  let terminal: PlanTerminalEvent;
  try {
    terminal = validateAndReplayTerminal(
      plan,
      trace.terminal,
      operations,
      expectedNextStepId,
      stepMap,
      eventByStep,
      rankCompletions,
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new PlanReplayError(`terminal: ${message}`);
  }
  return {
    status: terminal.status,
    appliedEvents: operations.length + 1,
    completedAtNs: terminal.timestampNs,
    rankCompletions: successfulRankCompletions(terminal.rankStates),
    rankStates: terminal.rankStates,
  };
}

export class CollectiveSubmitSequencer {
  private readonly executions = new Map<
    string,
    Map<string, { expected: number; submitted: number }>
  >();
  private readonly groupQueues = new Map<string, string[]>();

  registerExecution(
    executionId: string,
    expectedCollectivesByGroup: Readonly<Record<string, number>>,
  ): void {
    if (executionId.length === 0 || this.executions.has(executionId)) {
      throw new FrozenPlanExecutionError(
        `execution id must be non-empty and unique; got ${executionId}`,
      );
    }
    const entries = Object.entries(expectedCollectivesByGroup);
    for (const [groupId, expected] of entries) {
      if (groupId.length === 0 || !Number.isSafeInteger(expected) || expected < 0) {
        throw new FrozenPlanExecutionError(
          `invalid collective count ${expected} for ${groupId}`,
        );
      }
    }

    const groups = new Map<string, { expected: number; submitted: number }>();
    for (const [groupId, expected] of entries) {
      if (expected === 0) {
        continue;
      }
      groups.set(groupId, { expected, submitted: 0 });
      const queue = this.groupQueues.get(groupId) ?? [];
      queue.push(executionId);
      this.groupQueues.set(groupId, queue);
    }
    this.executions.set(executionId, groups);
  }

  submit(executionId: string, groupId: string, commSequenceId: number): void {
    const groups = this.executions.get(executionId);
    if (!groups) {
      throw new FrozenPlanExecutionError(`unknown execution ${executionId}`);
    }
    const progress = groups.get(groupId);
    if (!progress) {
      throw new FrozenPlanExecutionError(
        `execution ${executionId} has no pending collectives for ${groupId}`,
      );
    }
    const queue = this.groupQueues.get(groupId);
    if (!queue || queue[0] !== executionId) {
      throw new FrozenPlanExecutionError(
        `execution ${executionId} is not the submit owner for ${groupId}`,
      );
    }
    if (commSequenceId !== progress.submitted) {
      throw new FrozenPlanExecutionError(
        `collective sequence ${commSequenceId} does not match ${progress.submitted} for ${executionId}/${groupId}`,
      );
    }
    progress.submitted++;
    if (progress.submitted === progress.expected) {
      queue.shift();
      groups.delete(groupId);
      if (queue.length === 0) {
        this.groupQueues.delete(groupId);
      }
    }
  }

  completeExecution(executionId: string): void {
    const groups = this.executions.get(executionId);
    if (!groups) {
      throw new FrozenPlanExecutionError(`unknown execution ${executionId}`);
    }
    if (groups.size !== 0) {
      throw new FrozenPlanExecutionError(
        `execution ${executionId} still has pending collective submissions`,
      );
    }
    this.executions.delete(executionId);
  }
}

function validateOperation(
  step: PlanStep,
  path: string,
  devices: ReadonlyMap<string, SimDeviceSpec>,
  links: ReadonlyMap<string, SimLinkSpec>,
  groups: ReadonlyMap<string, CommunicatorGroupSpec>,
  rankDevices: ReadonlyMap<string, string>,
  allocations: ReadonlyMap<string, string>,
  add: (code: string, path: string, message: string) => void,
): void {
  validateDuration(step.operation.durationNs, `${path}.operation.durationNs`, add);
  switch (step.operation.kind) {
    case "compute": {
      const device = devices.get(step.operation.deviceId);
      if (!device) {
        add(
          "unknown_device",
          `${path}.operation.deviceId`,
          `unknown device ${step.operation.deviceId}`,
        );
      }
      if (device && !device.capabilities.includes(step.operation.capability)) {
        add(
          "missing_capability",
          `${path}.operation.capability`,
          `${device.id} lacks ${step.operation.capability}`,
        );
      }
      if (
        step.participants.length !== 1
        || rankDevices.get(step.participants[0]) !== step.operation.deviceId
      ) {
        add(
          "compute_participant",
          `${path}.participants`,
          "compute step must have the one rank owned by its device",
        );
      }
      if (device) {
        validateAllocationAccess(
          step,
          device.memoryDomainIds,
          allocations,
          path,
          add,
        );
      }
      return;
    }
    case "transfer": {
      const link = links.get(step.operation.linkId);
      if (!link) {
        add(
          "unknown_link",
          `${path}.operation.linkId`,
          `unknown link ${step.operation.linkId}`,
        );
      } else {
        if (step.reads.length === 0 || step.writes.length === 0) {
          add(
            "transfer_buffers",
            path,
            "transfer step must declare source reads and destination writes",
          );
        }
        for (const allocationId of step.reads) {
          if (allocations.get(allocationId) !== link.sourceDomainId) {
            add(
              "transfer_source",
              `${path}.reads`,
              `${allocationId} is not in ${link.sourceDomainId}`,
            );
          }
        }
        for (const allocationId of step.writes) {
          if (allocations.get(allocationId) !== link.targetDomainId) {
            add(
              "transfer_target",
              `${path}.writes`,
              `${allocationId} is not in ${link.targetDomainId}`,
            );
          }
        }
      }
      return;
    }
    case "collective": {
      const group = groups.get(step.operation.groupId);
      if (!group) {
        add(
          "unknown_group",
          `${path}.operation.groupId`,
          `unknown group ${step.operation.groupId}`,
        );
      } else {
        const expected = group.orderedRanks.map((rank) => rank.rankId);
        if (!arraysEqual(step.participants, expected)) {
          add(
            "collective_participants",
            `${path}.participants`,
            `must exactly match ordered group ranks [${expected.join(",")}]`,
          );
        }
        const participantDomains = new Set(
          group.orderedRanks.flatMap((rank) => (
            devices.get(rank.deviceId)?.memoryDomainIds ?? []
          )),
        );
        validateAllocationAccess(
          step,
          participantDomains,
          allocations,
          path,
          add,
        );
        for (const rank of group.orderedRanks) {
          if (
            devices.get(rank.deviceId)?.capabilities.includes("collective")
            !== true
          ) {
            add(
              "missing_capability",
              `${path}.participants`,
              `${rank.deviceId} lacks collective`,
            );
          }
        }
      }
      if (
        !Number.isSafeInteger(step.operation.commSequenceId)
        || step.operation.commSequenceId < 0
      ) {
        add(
          "collective_sequence",
          `${path}.operation.commSequenceId`,
          "must be a non-negative safe integer",
        );
      }
      validateUniqueStrings(
        step.operation.linkIds,
        `${path}.operation.linkIds`,
        add,
      );
      for (const linkId of step.operation.linkIds) {
        if (!links.has(linkId)) {
          add(
            "unknown_link",
            `${path}.operation.linkIds`,
            `unknown collective link ${linkId}`,
          );
        }
      }
      return;
    }
  }
}

function validateAllocationAccess(
  step: PlanStep,
  accessibleDomains: Iterable<string>,
  allocations: ReadonlyMap<string, string>,
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  const accessible = new Set(accessibleDomains);
  for (const allocationId of [...step.reads, ...step.writes]) {
    const domainId = allocations.get(allocationId);
    if (domainId && !accessible.has(domainId)) {
      add(
        "inaccessible_allocation",
        path,
        `${allocationId} in ${domainId} is not accessible to the operation`,
      );
    }
  }
}

function topologicalSort(
  steps: readonly PlanStep[],
  stepMap: ReadonlyMap<number, PlanStep>,
): number[] {
  const indegree = new Map<number, number>();
  const outgoing = new Map<number, number[]>();
  for (const step of steps) {
    if (!stepMap.has(step.id)) {
      continue;
    }
    const validDependencies = step.dependencies.filter((id) => stepMap.has(id));
    indegree.set(step.id, new Set(validDependencies).size);
    for (const dependency of validDependencies) {
      const targets = outgoing.get(dependency) ?? [];
      targets.push(step.id);
      outgoing.set(dependency, targets);
    }
  }

  const orderIndex = new Map(steps.map((step, index) => [step.id, index]));
  const ready = [...indegree.entries()]
    .filter(([, count]) => count === 0)
    .map(([id]) => id)
    .sort((left, right) => (
      (orderIndex.get(left) ?? 0) - (orderIndex.get(right) ?? 0)
    ));
  const result: number[] = [];
  while (ready.length > 0) {
    const id = ready.shift();
    if (id === undefined) {
      break;
    }
    result.push(id);
    for (const target of outgoing.get(id) ?? []) {
      const remaining = (indegree.get(target) ?? 0) - 1;
      indegree.set(target, remaining);
      if (remaining === 0) {
        ready.push(target);
        ready.sort((left, right) => (
          (orderIndex.get(left) ?? 0) - (orderIndex.get(right) ?? 0)
        ));
      }
    }
  }
  return result;
}

function buildReachability(
  steps: readonly PlanStep[],
  stepMap: ReadonlyMap<number, PlanStep>,
): ReadonlyMap<number, ReadonlySet<number>> {
  const result = new Map<number, Set<number>>();
  const visit = (stepId: number, visiting: Set<number>): Set<number> => {
    const cached = result.get(stepId);
    if (cached) {
      return cached;
    }
    if (visiting.has(stepId)) {
      return new Set();
    }
    visiting.add(stepId);
    const reachable = new Set<number>();
    for (const dependency of stepMap.get(stepId)?.dependencies ?? []) {
      reachable.add(dependency);
      for (const ancestor of visit(dependency, visiting)) {
        reachable.add(ancestor);
      }
    }
    visiting.delete(stepId);
    result.set(stepId, reachable);
    return reachable;
  };
  for (const step of steps) {
    visit(step.id, new Set());
  }
  return result;
}

function validateLeaseOrdering(
  steps: readonly PlanStep[],
  reachability: ReadonlyMap<number, ReadonlySet<number>>,
  add: (code: string, path: string, message: string) => void,
): void {
  for (let leftIndex = 0; leftIndex < steps.length; leftIndex++) {
    for (let rightIndex = leftIndex + 1; rightIndex < steps.length; rightIndex++) {
      const left = steps[leftIndex];
      const right = steps[rightIndex];
      const conflicts = conflictingAllocations(left, right);
      if (conflicts.length === 0) {
        continue;
      }
      const ordered = reachability.get(left.id)?.has(right.id) === true
        || reachability.get(right.id)?.has(left.id) === true;
      if (!ordered) {
        add(
          "unordered_lease_conflict",
          `steps[${leftIndex}],steps[${rightIndex}]`,
          `steps ${left.id} and ${right.id} conflict on ${conflicts.join(",")} without a dependency path`,
        );
      }
    }
  }
}

function validateCollectiveOrdering(
  steps: readonly PlanStep[],
  groups: ReadonlyMap<string, CommunicatorGroupSpec>,
  reachability: ReadonlyMap<number, ReadonlySet<number>>,
  add: (code: string, path: string, message: string) => void,
): void {
  for (const groupId of groups.keys()) {
    const collectives = steps
      .filter((step) => (
        step.operation.kind === "collective"
        && step.operation.groupId === groupId
      ))
      .sort((left, right) => {
        const leftSequence = left.operation.kind === "collective"
          ? left.operation.commSequenceId
          : 0;
        const rightSequence = right.operation.kind === "collective"
          ? right.operation.commSequenceId
          : 0;
        return leftSequence - rightSequence;
      });
    for (let index = 0; index < collectives.length; index++) {
      const step = collectives[index];
      if (
        step.operation.kind !== "collective"
        || step.operation.commSequenceId !== index
      ) {
        add(
          "collective_sequence_gap",
          "steps",
          `${groupId} collective sequence must be contiguous from zero`,
        );
        break;
      }
      if (
        index > 0
        && reachability.get(step.id)?.has(collectives[index - 1].id) !== true
      ) {
        add(
          "collective_order_dependency",
          "steps",
          `${groupId} collective ${index} must depend on collective ${index - 1}`,
        );
      }
    }
  }
}

function conflictingAllocations(left: PlanStep, right: PlanStep): string[] {
  const leftWrites = new Set(left.writes);
  const rightWrites = new Set(right.writes);
  const rightAccesses = new Set([...right.reads, ...right.writes]);
  const conflicts = new Set<string>();
  for (const allocation of leftWrites) {
    if (rightAccesses.has(allocation)) {
      conflicts.add(allocation);
    }
  }
  for (const allocation of rightWrites) {
    if (left.reads.includes(allocation)) {
      conflicts.add(allocation);
    }
  }
  return [...conflicts].sort();
}

class AllocationLeaseTimeline {
  private readonly intervals = new Map<
    string,
    Array<{ stepId: number; startNs: number; finishNs: number; mode: "read" | "write" }>
  >();

  reserve(step: PlanStep, startNs: number, finishNs: number): void {
    for (const [allocationId, mode] of accessModes(step)) {
      const intervals = this.intervals.get(allocationId) ?? [];
      for (const existing of intervals) {
        if (
          (mode === "write" || existing.mode === "write")
          && startNs < existing.finishNs
          && existing.startNs < finishNs
        ) {
          throw new FrozenPlanExecutionError(
            `step ${step.id} ${mode} lease overlaps step ${existing.stepId} ${existing.mode} lease on ${allocationId}`,
          );
        }
      }
      intervals.push({ stepId: step.id, startNs, finishNs, mode });
      this.intervals.set(allocationId, intervals);
    }
  }
}

function buildResourceLanes(
  scenario: SimulationScenario,
): Map<string, number[]> {
  const resources = new Map<string, number[]>();
  for (const device of scenario.devices) {
    resources.set(
      `compute:${device.id}`,
      Array.from({ length: device.maxConcurrentCompute }, () => 0),
    );
  }
  for (const link of scenario.links) {
    resources.set(
      `link:${link.id}`,
      Array.from({ length: link.concurrencyLanes }, () => 0),
    );
  }
  for (const group of scenario.groups) {
    resources.set(`collective:${group.id}`, [0]);
  }
  return resources;
}

function resourceIdsFor(operation: PlanOperation): readonly string[] {
  switch (operation.kind) {
    case "compute":
      return [`compute:${operation.deviceId}`];
    case "transfer":
      return [`link:${operation.linkId}`];
    case "collective":
      return [
        `collective:${operation.groupId}`,
        ...operation.linkIds.map((linkId) => `link:${linkId}`),
      ];
  }
}

function selectResourceReservations(
  operation: PlanOperation,
  resourceLanes: ReadonlyMap<string, readonly number[]>,
): PlanResourceReservation[] {
  return resourceIdsFor(operation).map((resourceId) => {
    const lanes = resourceLanes.get(resourceId);
    if (!lanes) {
      throw new FrozenPlanExecutionError(`missing resource ${resourceId}`);
    }
    return {
      resourceId,
      resourceLane: earliestLane(lanes),
    };
  });
}

function resourceCapacity(
  scenario: SimulationScenario,
  resourceId: string,
): number {
  if (resourceId.startsWith("compute:")) {
    return scenario.devices.find(
      (device) => resourceId === `compute:${device.id}`,
    )?.maxConcurrentCompute ?? 0;
  }
  if (resourceId.startsWith("link:")) {
    return scenario.links.find(
      (link) => resourceId === `link:${link.id}`,
    )?.concurrencyLanes ?? 0;
  }
  if (resourceId.startsWith("collective:")) {
    return scenario.groups.some(
      (group) => resourceId === `collective:${group.id}`,
    ) ? 1 : 0;
  }
  return 0;
}

function earliestLane(lanes: readonly number[]): number {
  let lane = 0;
  for (let index = 1; index < lanes.length; index++) {
    if (lanes[index] < lanes[lane]) {
      lane = index;
    }
  }
  return lane;
}

function traceEvent(
  plan: FrozenPlan,
  step: PlanStep,
  sourceSequence: number,
  submittedAtNs: number,
  startNs: number,
  finishNs: number,
  resources: readonly PlanResourceReservation[],
): PlanTraceEvent {
  return {
    contractRevision: PLAN_CONTRACT_REVISION,
    executionId: plan.executionId,
    topologyEpoch: plan.topologyEpoch,
    sourceSequence,
    stepId: step.id,
    kind: step.operation.kind,
    submittedAtNs,
    startNs,
    finishNs,
    resources: resources.map((resource) => ({ ...resource })),
    participants: [...step.participants],
    reads: [...step.reads],
    writes: [...step.writes],
    ...(step.operation.kind === "collective"
      ? {
          groupId: step.operation.groupId,
          commSequenceId: step.operation.commSequenceId,
        }
      : {}),
  };
}

function terminalTrace(
  plan: FrozenPlan,
  operations: readonly PlanTraceEvent[],
  status: PlanTerminalStatus,
  timestampNs: number,
  rankStates: readonly RankTerminalState[],
  beforeStepId?: number,
  failureAtNs?: number,
  reason?: string,
): PlanExecutionTrace {
  return {
    operations: operations.map((event) => ({
      ...event,
      resources: event.resources.map((resource) => ({ ...resource })),
      participants: [...event.participants],
      reads: [...event.reads],
      writes: [...event.writes],
    })),
    terminal: {
      contractRevision: PLAN_CONTRACT_REVISION,
      executionId: plan.executionId,
      topologyEpoch: plan.topologyEpoch,
      sourceSequence: operations.length,
      kind: "execution_terminal",
      status,
      timestampNs,
      ...(beforeStepId === undefined ? {} : { beforeStepId }),
      ...(failureAtNs === undefined ? {} : { failureAtNs }),
      ...(reason === undefined ? {} : { reason }),
      rankStates: rankStates.map((state) => ({ ...state })),
    },
  };
}

function allSucceededRankStates(
  plan: FrozenPlan,
  rankCompletions: ReadonlyMap<string, number>,
): RankTerminalState[] {
  return planRanks(plan).map((rankId) => ({
    rankId,
    status: "succeeded",
    terminalAtNs: rankCompletions.get(rankId) ?? 0,
  }));
}

function deriveRankStates(
  plan: FrozenPlan,
  scheduledStepIds: ReadonlySet<number>,
  rankCompletions: ReadonlyMap<string, number>,
  terminalStatus: "failed" | "aborted",
  terminalStep: PlanStep,
  failureAtNs: number,
  quiescedAtNs: number,
): RankTerminalState[] {
  return planRanks(plan).map((rankId) => {
    const rankStepIds = plan.steps
      .filter((step) => step.participants.includes(rankId))
      .map((step) => step.id);
    if (
      terminalStatus === "failed"
      && terminalStep.participants.includes(rankId)
    ) {
      return {
        rankId,
        status: "failed" as const,
        terminalAtNs: failureAtNs,
      };
    }
    if (rankStepIds.every((stepId) => scheduledStepIds.has(stepId))) {
      return {
        rankId,
        status: "succeeded" as const,
        terminalAtNs: rankCompletions.get(rankId) ?? 0,
      };
    }
    return {
      rankId,
      status: "aborted" as const,
      terminalAtNs: quiescedAtNs,
    };
  });
}

function validateAndReplayTerminal(
  plan: FrozenPlan,
  terminal: PlanTerminalEvent,
  operations: readonly PlanTraceEvent[],
  expectedNextStepId: number | undefined,
  stepMap: ReadonlyMap<number, PlanStep>,
  eventByStep: ReadonlyMap<number, PlanTraceEvent>,
  rankCompletions: ReadonlyMap<string, number>,
): PlanTerminalEvent {
  if (
    terminal.contractRevision !== PLAN_CONTRACT_REVISION
    || terminal.executionId !== plan.executionId
    || terminal.topologyEpoch !== plan.topologyEpoch
    || terminal.sourceSequence !== operations.length
    || terminal.kind !== "execution_terminal"
  ) {
    replayFail("terminal envelope does not match execution");
  }
  if (
    terminal.status !== "succeeded"
    && terminal.status !== "failed"
    && terminal.status !== "aborted"
  ) {
    replayFail(`unknown terminal status ${String(terminal.status)}`);
  }

  const operationQuiescence = Math.max(
    0,
    ...operations.map((event) => event.finishNs),
  );
  let expectedRankStates: RankTerminalState[];
  let expectedTimestamp: number;
  if (terminal.status === "succeeded") {
    if (
      expectedNextStepId !== undefined
      || terminal.beforeStepId !== undefined
      || terminal.failureAtNs !== undefined
      || terminal.reason !== undefined
    ) {
      replayFail("successful terminal has failure/abort metadata");
    }
    expectedTimestamp = operationQuiescence;
    expectedRankStates = allSucceededRankStates(plan, rankCompletions);
  } else {
    if (
      expectedNextStepId === undefined
      || terminal.beforeStepId !== expectedNextStepId
      || terminal.reason === undefined
      || terminal.reason.length === 0
    ) {
      replayFail("non-success terminal does not identify the next plan step");
    }
    const step = stepMap.get(expectedNextStepId);
    if (!step) {
      replayFail(`missing terminal step ${expectedNextStepId}`);
    }
    let dependencyReady = 0;
    for (const dependency of step.dependencies) {
      const event = eventByStep.get(dependency);
      if (!event) {
        replayFail(`terminal step dependency ${dependency} was not submitted`);
      }
      dependencyReady = Math.max(dependencyReady, event.finishNs);
    }
    const expectedFailureAt = dependencyReady;
    if (terminal.failureAtNs !== expectedFailureAt) {
      replayFail(
        `terminal failure/abort time ${terminal.failureAtNs} does not match ${expectedFailureAt}`,
      );
    }
    expectedTimestamp = Math.max(operationQuiescence, expectedFailureAt);
    expectedRankStates = deriveRankStates(
      plan,
      new Set(operations.map((event) => event.stepId)),
      rankCompletions,
      terminal.status,
      step,
      expectedFailureAt,
      expectedTimestamp,
    );
  }

  if (terminal.timestampNs !== expectedTimestamp) {
    replayFail(
      `terminal timestamp ${terminal.timestampNs} does not match quiescence ${expectedTimestamp}`,
    );
  }
  if (!rankStatesEqual(terminal.rankStates, expectedRankStates)) {
    replayFail("terminal rank states do not match rank-local execution state");
  }
  return { ...terminal, rankStates: expectedRankStates };
}

function planRanks(plan: FrozenPlan): string[] {
  return [...new Set(plan.steps.flatMap((step) => step.participants))].sort();
}

function rankStatesEqual(
  left: readonly RankTerminalState[],
  right: readonly RankTerminalState[],
): boolean {
  return left.length === right.length
    && left.every((state, index) => (
      state.rankId === right[index].rankId
      && state.status === right[index].status
      && state.terminalAtNs === right[index].terminalAtNs
    ));
}

function assertEventMatchesStep(step: PlanStep, event: PlanTraceEvent): void {
  if (
    event.kind !== step.operation.kind
    || !arraysEqual(
      event.resources.map((resource) => resource.resourceId),
      resourceIdsFor(step.operation),
    )
    || !arraysEqual(event.participants, step.participants)
    || !arraysEqual(event.reads, step.reads)
    || !arraysEqual(event.writes, step.writes)
  ) {
    replayFail(`event payload does not match step ${step.id}`);
  }
  if (step.operation.kind === "collective") {
    if (
      event.groupId !== step.operation.groupId
      || event.commSequenceId !== step.operation.commSequenceId
    ) {
      replayFail(`collective metadata does not match step ${step.id}`);
    }
  } else if (event.groupId !== undefined || event.commSequenceId !== undefined) {
    replayFail(`non-collective step ${step.id} carries collective metadata`);
  }
}

function resourceReservationsEqual(
  left: readonly PlanResourceReservation[],
  right: readonly PlanResourceReservation[],
): boolean {
  return left.length === right.length
    && left.every((reservation, index) => (
      reservation.resourceId === right[index].resourceId
      && reservation.resourceLane === right[index].resourceLane
    ));
}

function replayAllocationIntervals(
  intervalsByAllocation: Map<
    string,
    Array<{ event: PlanTraceEvent; mode: "read" | "write" }>
  >,
  event: PlanTraceEvent,
): void {
  for (const allocationId of event.reads) {
    replayOneAllocationInterval(intervalsByAllocation, allocationId, event, "read");
  }
  for (const allocationId of event.writes) {
    replayOneAllocationInterval(intervalsByAllocation, allocationId, event, "write");
  }
}

function replayOneAllocationInterval(
  intervalsByAllocation: Map<
    string,
    Array<{ event: PlanTraceEvent; mode: "read" | "write" }>
  >,
  allocationId: string,
  event: PlanTraceEvent,
  mode: "read" | "write",
): void {
  const intervals = intervalsByAllocation.get(allocationId) ?? [];
  if (
    intervals.some((prior) => (
      (mode === "write" || prior.mode === "write")
      && intervalsOverlap(prior.event, event)
    ))
  ) {
    replayFail(`${mode} lease on ${allocationId} overlaps another access`);
  }
  intervals.push({ event, mode });
  intervalsByAllocation.set(allocationId, intervals);
}

function intervalsOverlap(
  left: Pick<PlanTraceEvent, "startNs" | "finishNs">,
  right: Pick<PlanTraceEvent, "startNs" | "finishNs">,
): boolean {
  return left.startNs < right.finishNs && right.startNs < left.finishNs;
}

function accessModes(step: PlanStep): Array<[string, "read" | "write"]> {
  return [
    ...step.reads.map((allocation): [string, "read"] => [allocation, "read"]),
    ...step.writes.map((allocation): [string, "write"] => [allocation, "write"]),
  ];
}

function buildRankDeviceMap(
  groups: readonly CommunicatorGroupSpec[],
): Map<string, string> {
  const result = new Map<string, string>();
  for (const group of groups) {
    for (const rank of group.orderedRanks) {
      if (!result.has(rank.rankId)) {
        result.set(rank.rankId, rank.deviceId);
      }
    }
  }
  return result;
}

function sortedRankCompletions(
  values: ReadonlyMap<string, number>,
): RankCompletion[] {
  return [...values.entries()]
    .map(([rankId, completedAtNs]) => ({ rankId, completedAtNs }))
    .sort((left, right) => left.rankId.localeCompare(right.rankId));
}

function successfulRankCompletions(
  states: readonly RankTerminalState[],
): RankCompletion[] {
  return states
    .filter((state) => state.status === "succeeded")
    .map((state) => ({
      rankId: state.rankId,
      completedAtNs: state.terminalAtNs,
    }))
    .sort((left, right) => left.rankId.localeCompare(right.rankId));
}

function validateDuration(
  value: number,
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    add(
      "duration",
      path,
      `must be a non-negative safe integer; got ${value}`,
    );
  }
}

function validateUniqueStrings(
  values: readonly string[],
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  if (new Set(values).size !== values.length || values.some((value) => value.length === 0)) {
    add("duplicate_value", path, "values must be non-empty and unique");
  }
}

function validateUniqueNumbers(
  values: readonly number[],
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  if (
    new Set(values).size !== values.length
    || values.some((value) => !Number.isSafeInteger(value) || value < 0)
  ) {
    add(
      "duplicate_value",
      path,
      "values must be unique non-negative safe integers",
    );
  }
}

function arraysEqual<T>(left: readonly T[], right: readonly T[]): boolean {
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function checkedTimeAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    throw new FrozenPlanExecutionError(
      `${label} finish time exceeds Number.MAX_SAFE_INTEGER`,
    );
  }
  return result;
}

function replayFail(message: string): never {
  throw new Error(message);
}
