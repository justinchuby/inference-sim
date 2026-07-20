import {
  PLAN_CONTRACT_REVISION,
  type FrozenPlan,
  type FrozenPlanExecutionResult,
  type PlanFault,
  type PlanExecutionTrace,
  type PlanOperation,
  type PlanReplayResult,
  type PlanResourceReservation,
  type PlanStep,
  type PlanTerminalEvent,
  type PlanTraceEvent,
  type RankCompletion,
  type RankTerminalState,
} from "./plan-types.js";
import type {
  CommunicatorGroupSpec,
  SimulationScenario,
} from "./scenario-types.js";
import { DiscreteEventSimulator } from "./event-loop.js";
import {
  CollectiveSubmitSequencer,
  FrozenPlanExecutionError,
  PlanReplayError,
  validateFrozenPlan,
} from "./frozen-plan.js";

export const CONCURRENT_PLAN_TRACE_REVISION = 1;

export interface ConcurrentPlanRequest {
  readonly plan: FrozenPlan;
  readonly arrivalNs: number;
  readonly admissionOrder: number;
  readonly stepNotBeforeNs?: Readonly<Record<number, number>>;
}

export interface ConcurrentPlanAdmission {
  readonly executionId: string;
  readonly arrivalNs: number;
  readonly admissionOrder: number;
}

export interface ConcurrentPlanOperationEvent {
  readonly globalSequence: number;
  readonly event: PlanTraceEvent;
}

export interface ConcurrentPlanExecutionTrace {
  readonly revision: typeof CONCURRENT_PLAN_TRACE_REVISION;
  readonly admissions: readonly ConcurrentPlanAdmission[];
  readonly operations: readonly ConcurrentPlanOperationEvent[];
  readonly terminals: readonly PlanTerminalEvent[];
}

export interface ConcurrentPlanExecutionResult {
  readonly completedAtNs: number;
  readonly maximumConcurrentExecutions: number;
  readonly executions: readonly FrozenPlanExecutionResult[];
  readonly trace: ConcurrentPlanExecutionTrace;
}

export interface ConcurrentPlanReplayResult {
  readonly appliedEvents: number;
  readonly completedAtNs: number;
  readonly maximumConcurrentExecutions: number;
  readonly executions: readonly PlanReplayResult[];
}

export type ConcurrentNodeFailure = Extract<
  PlanFault,
  { readonly kind: "node_failure" }
>;

export interface ConcurrentPlanReplayOptions {
  readonly nodeFailure?: ConcurrentNodeFailure;
}

export interface ConcurrentPlanCampaignOptions {
  readonly executionCount: number;
  readonly seed: number;
  readonly arrivalWindowNs: number;
}

export interface ConcurrentPlanCampaignResult {
  readonly options: ConcurrentPlanCampaignOptions;
  readonly requests: readonly ConcurrentPlanAdmission[];
  readonly execution: ConcurrentPlanExecutionResult;
  readonly replay: ConcurrentPlanReplayResult;
}

export interface ConcurrentNodeFailureCampaignResult {
  readonly options: ConcurrentPlanCampaignOptions;
  readonly fault: ConcurrentNodeFailure;
  readonly requests: readonly ConcurrentPlanAdmission[];
  readonly execution: ConcurrentPlanExecutionResult;
  readonly replay: ConcurrentPlanReplayResult;
}

interface ExecutionState {
  readonly request: ConcurrentPlanRequest;
  readonly stepMap: ReadonlyMap<number, PlanStep>;
  readonly planOrder: ReadonlyMap<number, number>;
  readonly dependents: ReadonlyMap<number, readonly number[]>;
  readonly remainingDependencies: Map<number, number>;
  readonly releasedSteps: Set<number>;
  readonly ready: number[];
  readonly completedSteps: Set<number>;
  readonly localOperations: PlanTraceEvent[];
  readonly rankCompletions: Map<string, number>;
  admitted: boolean;
  result?: FrozenPlanExecutionResult;
}

type RuntimeEvent =
  | {
      readonly kind: "step_completion";
      readonly executionId: string;
      readonly stepId: number;
    }
  | {
      readonly kind: "step_release";
      readonly executionId: string;
      readonly stepId: number;
    };

export interface StreamingPlanStepCompletion {
  readonly executionId: string;
  readonly stepId: number;
  readonly completedAtNs: number;
}

export class StreamingConcurrentPlanRuntime {
  private readonly requests: ConcurrentPlanRequest[] = [];
  private readonly states = new Map<string, ExecutionState>();
  private readonly resourceLanes: Map<string, number[]>;
  private readonly leases = new ConcurrentLeaseTimeline();
  private readonly sequencer: CollectiveSubmitSequencer;
  private readonly simulator = new DiscreteEventSimulator<RuntimeEvent>();
  private readonly operations: ConcurrentPlanOperationEvent[] = [];
  private readonly admissions: ConcurrentPlanAdmission[] = [];
  private readonly terminals: PlanTerminalEvent[] = [];
  private readonly operationListeners = new Set<
    (event: PlanTraceEvent) => void
  >();
  private lastAdmissionNs = 0;
  private validatedRuntimeEvents = 0;

  constructor(private readonly scenario: SimulationScenario) {
    this.resourceLanes = buildResourceLanes(scenario);
    this.sequencer = sequencerForEpoch(scenario.execution.topologyEpoch);
  }

  get currentTimeNs(): number {
    return this.simulator.nowNs;
  }

  onOperationSubmitted(
    listener: (event: PlanTraceEvent) => void,
  ): () => void {
    this.operationListeners.add(listener);
    return () => {
      this.operationListeners.delete(listener);
    };
  }

  advanceTo(timestampNs: number): void {
    if (
      !Number.isSafeInteger(timestampNs)
      || timestampNs < this.simulator.nowNs
    ) {
      throw new FrozenPlanExecutionError(
        `cannot advance streaming runtime from ${this.simulator.nowNs}ns to ${timestampNs}ns`,
      );
    }
    this.runCompletionsThrough(timestampNs);
  }

  admit(
    plan: FrozenPlan,
    arrivalNs: number,
    stepNotBeforeNs: Readonly<Record<number, number>> = {},
  ): ConcurrentPlanAdmission {
    if (
      !Number.isSafeInteger(arrivalNs)
      || arrivalNs < this.lastAdmissionNs
      || arrivalNs < this.simulator.nowNs
    ) {
      throw new FrozenPlanExecutionError(
        `streaming admission ${arrivalNs}ns precedes runtime time ${this.simulator.nowNs}ns or prior admission ${this.lastAdmissionNs}ns`,
      );
    }
    const request: ConcurrentPlanRequest = {
      plan,
      arrivalNs,
      admissionOrder: this.requests.length,
      ...(Object.keys(stepNotBeforeNs).length === 0
        ? {}
        : { stepNotBeforeNs: { ...stepNotBeforeNs } }),
    };
    const validatedRuntimeEvents = validateStreamingAdmission(
      this.scenario,
      request,
      this.states.has(plan.executionId),
      this.requests.length,
      this.validatedRuntimeEvents,
    );
    this.advanceTo(arrivalNs);
    const state = buildExecutionState(request);
    this.requests.push(request);
    this.states.set(plan.executionId, state);
    this.sequencer.registerExecution(
      plan.executionId,
      plan.topologyEpoch,
      collectiveCounts(plan),
    );
    state.admitted = true;
    for (const [stepIdText, releaseNs] of Object.entries(
      request.stepNotBeforeNs ?? {},
    )) {
      const stepId = Number(stepIdText);
      if (releaseNs > arrivalNs) {
        this.simulator.scheduleAt(releaseNs, {
          kind: "step_release",
          executionId: plan.executionId,
          stepId,
        });
      }
    }
    const admission = admissionFor(request);
    this.admissions.push(admission);
    this.lastAdmissionNs = arrivalNs;
    this.validatedRuntimeEvents = validatedRuntimeEvents;
    this.submitReady(arrivalNs);
    return admission;
  }

  runUntilStep(
    executionId: string,
    stepId: number,
  ): StreamingPlanStepCompletion {
    const state = this.states.get(executionId);
    if (!state?.stepMap.has(stepId)) {
      throw new FrozenPlanExecutionError(
        `unknown streaming step ${executionId}/${stepId}`,
      );
    }
    while (!state.completedSteps.has(stepId)) {
      const processed = this.simulator.runNext(
        (scheduled, activeSimulator) => {
          this.handleRuntimeEvent(
            scheduled.payload,
            activeSimulator.nowNs,
          );
          this.submitReady(activeSimulator.nowNs);
        },
      );
      if (!processed) {
        throw new FrozenPlanExecutionError(
          `streaming execution quiesced before ${executionId}/${stepId}`,
        );
      }
    }
    const event = state.localOperations.find(
      (operation) => operation.stepId === stepId,
    );
    if (event === undefined) {
      throw new FrozenPlanExecutionError(
        `streaming completion for ${executionId}/${stepId} is inconsistent`,
      );
    }
    return { executionId, stepId, completedAtNs: event.finishNs };
  }

  drain(): ConcurrentPlanExecutionResult {
    if (this.requests.length === 0) {
      throw new FrozenPlanExecutionError(
        "streaming execution requires at least one admitted plan",
      );
    }
    this.runCompletionsThrough();
    const unfinished = [...this.states.values()].filter(
      (state) => !state.result,
    );
    if (unfinished.length > 0) {
      throw new FrozenPlanExecutionError(
        `streaming execution quiesced with unfinished plans: ${
          unfinished.map(
            (state) => state.request.plan.executionId,
          ).join(", ")
        }`,
      );
    }
    const executions = this.requests.map((request) => {
      const result = this.states.get(request.plan.executionId)?.result;
      if (!result) {
        throw new FrozenPlanExecutionError(
          `missing result for ${request.plan.executionId}`,
        );
      }
      return result;
    });
    const completedAtNs = executions.reduce(
      (maximum, execution) => Math.max(maximum, execution.completedAtNs),
      0,
    );
    const terminals = [...this.terminals].sort(
      compareTerminals(this.requests),
    );
    return {
      completedAtNs,
      maximumConcurrentExecutions: maximumConcurrency(
        this.admissions,
        terminals,
      ),
      executions,
      trace: {
        revision: CONCURRENT_PLAN_TRACE_REVISION,
        admissions: [...this.admissions],
        operations: [...this.operations],
        terminals,
      },
    };
  }

  private submitReady(submittedAtNs: number): void {
    while (true) {
      const candidates = [...this.states.values()]
        .filter((state) => state.admitted && !state.result)
        .flatMap((state) => state.ready.map((stepId) => ({ state, stepId })))
        .sort(compareReadyCandidates);
      let submitted = false;
      for (const candidate of candidates) {
        const step = candidate.state.stepMap.get(candidate.stepId);
        if (!step) {
          throw new FrozenPlanExecutionError(
            `ready step ${candidate.stepId} disappeared from ${candidate.state.request.plan.executionId}`,
          );
        }
        if (
          step.operation.kind === "collective"
          && !this.sequencer.canSubmit(
            candidate.state.request.plan.executionId,
            step.operation.groupId,
            step.operation.commSequenceId,
          )
        ) {
          continue;
        }
        const readyIndex = candidate.state.ready.indexOf(step.id);
        if (readyIndex < 0) {
          throw new FrozenPlanExecutionError(
            `ready step ${step.id} was selected twice`,
          );
        }
        candidate.state.ready.splice(readyIndex, 1);
        if (step.operation.kind === "collective") {
          this.sequencer.submit(
            candidate.state.request.plan.executionId,
            step.operation.groupId,
            step.operation.commSequenceId,
          );
        }
        const resources = selectResourceReservations(
          step.operation,
          this.resourceLanes,
        );
        const resourceReadyNs = Math.max(
          submittedAtNs,
          ...resources.map((reservation) => (
            this.resourceLanes.get(reservation.resourceId)?.[
              reservation.resourceLane
            ] ?? 0
          )),
        );
        const startNs = this.leases.earliestStart(
          step,
          resourceReadyNs,
          step.operation.durationNs,
          submittedAtNs,
        );
        const finishNs = checkedAdd(
          startNs,
          step.operation.durationNs,
          `step ${step.id} finish`,
        );
        this.leases.reserve(
          candidate.state.request.plan.executionId,
          step,
          startNs,
          finishNs,
        );
        reserveResources(this.resourceLanes, resources, finishNs);
        const event = buildTraceEvent(
          candidate.state.request.plan,
          step,
          candidate.state.localOperations.length,
          submittedAtNs,
          startNs,
          finishNs,
          resources,
        );
        candidate.state.localOperations.push(event);
        this.operations.push({
          globalSequence: this.operations.length,
          event,
        });
        for (const listener of this.operationListeners) {
          listener(structuredClone(event));
        }
        this.simulator.scheduleAt(finishNs, {
          kind: "step_completion",
          executionId: candidate.state.request.plan.executionId,
          stepId: step.id,
        });
        submitted = true;
        break;
      }
      if (!submitted) {
        return;
      }
    }
  }

  private handleRuntimeEvent(
    event: RuntimeEvent,
    atNs: number,
  ): void {
    if (event.kind === "step_completion") {
      this.completeStep(event, atNs);
      return;
    }
    this.releaseStep(event.executionId, event.stepId, atNs);
  }

  private releaseStep(
    executionId: string,
    stepId: number,
    atNs: number,
  ): void {
    const state = this.states.get(executionId);
    const releaseNs = state?.request.stepNotBeforeNs?.[stepId];
    if (
      !state
      || !state.stepMap.has(stepId)
      || releaseNs !== atNs
      || state.releasedSteps.has(stepId)
    ) {
      throw new FrozenPlanExecutionError(
        `invalid step release ${executionId}/${stepId} at ${atNs}ns`,
      );
    }
    state.releasedSteps.add(stepId);
    if (
      state.remainingDependencies.get(stepId) === 0
      && !state.completedSteps.has(stepId)
      && !state.localOperations.some((event) => event.stepId === stepId)
    ) {
      state.ready.push(stepId);
    }
  }

  private completeStep(
    completion: Extract<RuntimeEvent, { readonly kind: "step_completion" }>,
    completedAtNs: number,
  ): void {
    const state = this.states.get(completion.executionId);
    const step = state?.stepMap.get(completion.stepId);
    if (!state || !step || state.completedSteps.has(completion.stepId)) {
      throw new FrozenPlanExecutionError(
        `invalid completion ${completion.executionId}/${completion.stepId}`,
      );
    }
    state.completedSteps.add(step.id);
    for (const rankId of step.participants) {
      state.rankCompletions.set(
        rankId,
        Math.max(state.rankCompletions.get(rankId) ?? 0, completedAtNs),
      );
    }
    for (const dependentId of state.dependents.get(step.id) ?? []) {
      const remaining =
        (state.remainingDependencies.get(dependentId) ?? 0) - 1;
      state.remainingDependencies.set(dependentId, remaining);
      if (remaining === 0 && state.releasedSteps.has(dependentId)) {
        state.ready.push(dependentId);
      }
    }
    if (state.completedSteps.size === state.request.plan.steps.length) {
      this.sequencer.completeExecution(state.request.plan.executionId);
      state.result = successfulExecutionResult(state, completedAtNs);
      this.terminals.push(state.result.trace.terminal);
    }
  }

  private runCompletionsThrough(untilNs?: number): void {
    this.simulator.run((scheduled, activeSimulator) => {
      this.handleRuntimeEvent(scheduled.payload, activeSimulator.nowNs);
      this.submitReady(activeSimulator.nowNs);
    }, {
      ...(untilNs === undefined ? {} : { untilNs }),
      maxEvents: this.scenario.execution.maxEvents,
    });
  }
}

export function executeConcurrentFrozenPlans(
  scenario: SimulationScenario,
  requests: readonly ConcurrentPlanRequest[],
): ConcurrentPlanExecutionResult {
  const ordered = validateConcurrentRequests(scenario, requests);
  const runtime = new StreamingConcurrentPlanRuntime(scenario);
  for (const request of ordered) {
    runtime.admit(
      request.plan,
      request.arrivalNs,
      request.stepNotBeforeNs,
    );
  }
  return runtime.drain();
}

export function executeConcurrentNodeFailure(
  scenario: SimulationScenario,
  requests: readonly ConcurrentPlanRequest[],
  fault: ConcurrentNodeFailure,
): ConcurrentPlanExecutionResult {
  const ordered = validateConcurrentNodeFailure(scenario, requests, fault);
  const baseline = executeConcurrentFrozenPlans(scenario, ordered);
  const completedBeforeFault = baseline.executions.filter(
    (execution) => execution.completedAtNs < fault.atNs,
  );
  if (completedBeforeFault.length > 0) {
    throw new FrozenPlanExecutionError(
      `node fault occurs after executions completed: ${
        completedBeforeFault.map((execution) => execution.executionId).join(", ")
      }`,
    );
  }
  const operations = baseline.trace.operations.filter(
    ({ event }) => event.submittedAtNs < fault.atNs,
  );
  const terminals: PlanTerminalEvent[] = [];
  const executions = ordered.map((request): FrozenPlanExecutionResult => {
    const localOperations = operations
      .map(({ event }) => event)
      .filter((event) => event.executionId === request.plan.executionId);
    const terminal = nodeFailureTerminal(
      scenario,
      request.plan,
      localOperations,
      fault,
    );
    terminals.push(terminal);
    const trace: PlanExecutionTrace = {
      operations: localOperations,
      terminal,
    };
    return {
      status: "failed",
      executionId: request.plan.executionId,
      completedAtNs: terminal.timestampNs,
      trace,
      rankCompletions: terminal.rankStates
        .filter((state) => state.status === "succeeded")
        .map((state) => ({
          rankId: state.rankId,
          completedAtNs: state.terminalAtNs,
        })),
      rankStates: terminal.rankStates,
    };
  });
  terminals.sort(compareTerminals(ordered));
  const completedAtNs = executions.reduce(
    (maximum, execution) => Math.max(maximum, execution.completedAtNs),
    fault.atNs,
  );
  return {
    completedAtNs,
    maximumConcurrentExecutions: maximumConcurrency(
      baseline.trace.admissions,
      terminals,
    ),
    executions,
    trace: {
      revision: CONCURRENT_PLAN_TRACE_REVISION,
      admissions: baseline.trace.admissions,
      operations,
      terminals,
    },
  };
}

export function replayConcurrentPlanTrace(
  scenario: SimulationScenario,
  requests: readonly ConcurrentPlanRequest[],
  trace: ConcurrentPlanExecutionTrace,
  options: ConcurrentPlanReplayOptions = {},
): ConcurrentPlanReplayResult {
  const ordered = validateConcurrentRequests(scenario, requests);
  if (trace.revision !== CONCURRENT_PLAN_TRACE_REVISION) {
    throw new PlanReplayError(
      `unsupported concurrent trace revision ${trace.revision}`,
    );
  }
  const expectedAdmissions = ordered.map(admissionFor);
  if (!admissionsEqual(trace.admissions, expectedAdmissions)) {
    throw new PlanReplayError("concurrent admissions do not match requests");
  }

  const requestsById = new Map(
    ordered.map((request) => [request.plan.executionId, request]),
  );
  const stepsByExecution = new Map(
    ordered.map((request) => [
      request.plan.executionId,
      new Map(request.plan.steps.map((step) => [step.id, step])),
    ]),
  );
  const eventsByExecution = new Map(
    ordered.map((request) => [
      request.plan.executionId,
      new Map<number, PlanTraceEvent>(),
    ]),
  );
  const localCounts = new Map(
    ordered.map((request) => [request.plan.executionId, 0]),
  );
  const resourceLanes = buildResourceLanes(scenario);
  const leases = new ConcurrentLeaseTimeline();
  const sequencer = sequencerForEpoch(scenario.execution.topologyEpoch);
  for (const request of ordered) {
    sequencer.registerExecution(
      request.plan.executionId,
      request.plan.topologyEpoch,
      collectiveCounts(request.plan),
    );
  }
  const priorGroupOwner = priorGroupOwners(ordered);
  const finalCollectiveSubmission = new Map<string, number>();
  const arbitrationPhaseByStep = new Map<string, number>();
  const admissionPhaseByTime = new Map<number, number>();
  let lastSubmittedAtNs = 0;
  let lastArbitration:
    | {
        readonly submittedAtNs: number;
        readonly phase: number;
        readonly admissionOrder: number;
        readonly planOrder: number;
      }
    | undefined;

  for (let index = 0; index < trace.operations.length; index++) {
    const wrapper = trace.operations[index];
    const event = wrapper.event;
    try {
      if (wrapper.globalSequence !== index) {
        replayFail(
          `global sequence ${wrapper.globalSequence} does not match ${index}`,
        );
      }
      const request = requestsById.get(event.executionId);
      const step = stepsByExecution.get(event.executionId)?.get(event.stepId);
      const executionEvents = eventsByExecution.get(event.executionId);
      if (!request || !step || !executionEvents) {
        replayFail(`unknown execution or step ${event.executionId}/${event.stepId}`);
      }
      if (executionEvents.has(step.id)) {
        replayFail(`duplicate step ${event.executionId}/${step.id}`);
      }
      const localSequence = localCounts.get(event.executionId) ?? 0;
      if (event.sourceSequence !== localSequence) {
        replayFail(
          `source sequence ${event.sourceSequence} does not match ${localSequence} for ${event.executionId}`,
        );
      }
      assertEventMatchesStep(request.plan, step, event);
      if (
        !Number.isSafeInteger(event.submittedAtNs)
        || !Number.isSafeInteger(event.startNs)
        || !Number.isSafeInteger(event.finishNs)
        || event.submittedAtNs < request.arrivalNs
        || event.startNs < event.submittedAtNs
        || event.finishNs - event.startNs !== step.operation.durationNs
      ) {
        replayFail(`invalid timing for ${event.executionId}/${step.id}`);
      }
      if (event.submittedAtNs < lastSubmittedAtNs) {
        replayFail("global submission time moved backwards");
      }
      lastSubmittedAtNs = event.submittedAtNs;

      let dependencyReadyNs = request.arrivalNs;
      let arbitrationPhase = 0;
      if (request.arrivalNs === event.submittedAtNs) {
        let admissionPhase = admissionPhaseByTime.get(event.submittedAtNs);
        if (admissionPhase === undefined) {
          admissionPhase = lastArbitration?.submittedAtNs
              === event.submittedAtNs
            ? lastArbitration.phase + 1
            : 0;
          admissionPhaseByTime.set(event.submittedAtNs, admissionPhase);
        }
        arbitrationPhase = admissionPhase;
      }
      for (const dependency of step.dependencies) {
        const dependencyEvent = executionEvents.get(dependency);
        if (!dependencyEvent) {
          replayFail(
            `${event.executionId}/${step.id} submitted before dependency ${dependency}`,
          );
        }
        dependencyReadyNs = Math.max(
          dependencyReadyNs,
          dependencyEvent.finishNs,
        );
        if (
          dependencyEvent.submittedAtNs === event.submittedAtNs
          && dependencyEvent.finishNs === event.submittedAtNs
        ) {
          arbitrationPhase = Math.max(
            arbitrationPhase,
            (arbitrationPhaseByStep.get(
              identityKey(event.executionId, dependency),
            ) ?? 0) + 1,
          );
        }
      }
      const planOrder = request.plan.steps.findIndex(
        (candidate) => candidate.id === step.id,
      );
      const arbitration = {
        submittedAtNs: event.submittedAtNs,
        phase: arbitrationPhase,
        admissionOrder: request.admissionOrder,
        planOrder,
      };
      if (
        lastArbitration
        && arbitration.submittedAtNs === lastArbitration.submittedAtNs
        && compareArbitration(arbitration, lastArbitration) < 0
      ) {
        replayFail(
          `${event.executionId}/${step.id} violates canonical ready-work arbitration`,
        );
      }
      lastArbitration = arbitration;
      let expectedSubmittedAtNs = Math.max(
        dependencyReadyNs,
        request.stepNotBeforeNs?.[step.id] ?? request.arrivalNs,
      );
      if (step.operation.kind === "collective") {
        const groupKey = identityKey(
          event.executionId,
          step.operation.groupId,
        );
        const priorOwner = priorGroupOwner.get(groupKey);
        if (priorOwner) {
          const priorFinal = finalCollectiveSubmission.get(
            identityKey(priorOwner, step.operation.groupId),
          );
          if (priorFinal === undefined) {
            replayFail(
              `${event.executionId} overtook ${priorOwner} on ${step.operation.groupId}`,
            );
          }
          expectedSubmittedAtNs = Math.max(
            expectedSubmittedAtNs,
            priorFinal,
          );
        }
        if (!sequencer.canSubmit(
          event.executionId,
          step.operation.groupId,
          step.operation.commSequenceId,
        )) {
          replayFail(
            `${event.executionId} is not collective owner for ${step.operation.groupId}`,
          );
        }
      }
      if (event.submittedAtNs !== expectedSubmittedAtNs) {
        replayFail(
          `${event.executionId}/${step.id} submitted at ${event.submittedAtNs}ns instead of ${expectedSubmittedAtNs}ns`,
        );
      }

      const expectedResources = selectResourceReservations(
        step.operation,
        resourceLanes,
      );
      if (!reservationsEqual(event.resources, expectedResources)) {
        replayFail(
          `${event.executionId}/${step.id} resource reservations are not deterministic`,
        );
      }
      const resourceReadyNs = Math.max(
        event.submittedAtNs,
        ...expectedResources.map((reservation) => (
          resourceLanes.get(reservation.resourceId)?.[
            reservation.resourceLane
          ] ?? 0
        )),
      );
      const expectedStartNs = leases.earliestStart(
        step,
        resourceReadyNs,
        step.operation.durationNs,
        event.submittedAtNs,
      );
      if (event.startNs !== expectedStartNs) {
        replayFail(
          `${event.executionId}/${step.id} starts at ${event.startNs}ns instead of ${expectedStartNs}ns`,
        );
      }
      const expectedFinishNs = checkedAdd(
        expectedStartNs,
        step.operation.durationNs,
        `${event.executionId}/${step.id} replay finish`,
      );
      if (event.finishNs !== expectedFinishNs) {
        replayFail(`invalid finish for ${event.executionId}/${step.id}`);
      }
      leases.reserve(
        event.executionId,
        step,
        event.startNs,
        event.finishNs,
      );
      reserveResources(resourceLanes, expectedResources, event.finishNs);
      if (step.operation.kind === "collective") {
        sequencer.submit(
          event.executionId,
          step.operation.groupId,
          step.operation.commSequenceId,
        );
        const count = collectiveCounts(request.plan)[step.operation.groupId] ?? 0;
        if (step.operation.commSequenceId === count - 1) {
          finalCollectiveSubmission.set(
            identityKey(event.executionId, step.operation.groupId),
            event.submittedAtNs,
          );
        }
      }
      executionEvents.set(step.id, event);
      arbitrationPhaseByStep.set(
        identityKey(event.executionId, step.id),
        arbitrationPhase,
      );
      localCounts.set(event.executionId, localSequence + 1);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new PlanReplayError(`concurrent event ${index}: ${message}`);
    }
  }

  if (options.nodeFailure) {
    return replayConcurrentNodeFailureTerminals(
      scenario,
      ordered,
      trace,
      options.nodeFailure,
      eventsByExecution,
      priorGroupOwner,
      finalCollectiveSubmission,
    );
  }

  const expectedTerminals: PlanTerminalEvent[] = [];
  const replayExecutions: PlanReplayResult[] = [];
  for (const request of ordered) {
    const executionEvents = eventsByExecution.get(request.plan.executionId);
    if (!executionEvents || executionEvents.size !== request.plan.steps.length) {
      throw new PlanReplayError(
        `${request.plan.executionId} has ${executionEvents?.size ?? 0}/${request.plan.steps.length} operations`,
      );
    }
    sequencer.completeExecution(request.plan.executionId);
    const localOperations = [...executionEvents.values()]
      .sort((left, right) => left.sourceSequence - right.sourceSequence);
    const terminal = successfulTerminal(request.plan, localOperations);
    expectedTerminals.push(terminal);
    replayExecutions.push({
      status: "succeeded",
      appliedEvents: localOperations.length + 1,
      completedAtNs: terminal.timestampNs,
      rankCompletions: terminal.rankStates.map((state) => ({
        rankId: state.rankId,
        completedAtNs: state.terminalAtNs,
      })),
      rankStates: terminal.rankStates,
    });
  }
  expectedTerminals.sort(compareTerminals(ordered));
  if (!terminalsEqual(trace.terminals, expectedTerminals)) {
    throw new PlanReplayError("concurrent terminal events do not match execution");
  }
  const completedAtNs = expectedTerminals.reduce(
    (maximum, terminal) => Math.max(maximum, terminal.timestampNs),
    0,
  );
  return {
    appliedEvents:
      trace.admissions.length + trace.operations.length + trace.terminals.length,
    completedAtNs,
    maximumConcurrentExecutions: maximumConcurrency(
      trace.admissions,
      trace.terminals,
    ),
    executions: replayExecutions,
  };
}

export function runSeededConcurrentPlanCampaign(
  scenario: SimulationScenario,
  template: FrozenPlan,
  options: ConcurrentPlanCampaignOptions,
): ConcurrentPlanCampaignResult {
  validateCampaignInputs(scenario, template, options);
  const requests = buildSeededRequests(template, options);
  const execution = executeConcurrentFrozenPlans(scenario, requests);
  const replay = replayConcurrentPlanTrace(scenario, requests, execution.trace);
  return {
    options: { ...options },
    requests: execution.trace.admissions,
    execution,
    replay,
  };
}

export function runSeededConcurrentNodeFailureCampaign(
  scenario: SimulationScenario,
  template: FrozenPlan,
  options: ConcurrentPlanCampaignOptions,
  fault: ConcurrentNodeFailure,
): ConcurrentNodeFailureCampaignResult {
  validateCampaignInputs(scenario, template, options);
  const requests = buildSeededRequests(template, options);
  const execution = executeConcurrentNodeFailure(
    scenario,
    requests,
    fault,
  );
  const replay = replayConcurrentPlanTrace(
    scenario,
    requests,
    execution.trace,
    { nodeFailure: fault },
  );
  return {
    options: { ...options },
    fault: { ...fault },
    requests: execution.trace.admissions,
    execution,
    replay,
  };
}

function validateConcurrentRequests(
  scenario: SimulationScenario,
  requests: readonly ConcurrentPlanRequest[],
): ConcurrentPlanRequest[] {
  if (requests.length === 0) {
    throw new FrozenPlanExecutionError(
      "concurrent execution requires at least one plan",
    );
  }
  const executionIds = new Set<string>();
  const admissionOrders = new Set<number>();
  let totalSteps = 0;
  let releaseEvents = 0;
  for (const request of requests) {
    if (
      !Number.isSafeInteger(request.arrivalNs)
      || request.arrivalNs < 0
      || !Number.isSafeInteger(request.admissionOrder)
      || request.admissionOrder < 0
    ) {
      throw new FrozenPlanExecutionError(
        "arrival and admission order must be non-negative safe integers",
      );
    }
    if (
      executionIds.has(request.plan.executionId)
      || admissionOrders.has(request.admissionOrder)
    ) {
      throw new FrozenPlanExecutionError(
        "concurrent execution ids and admission orders must be unique",
      );
    }
    const validation = validateFrozenPlan(scenario, request.plan);
    if (!validation.valid) {
      throw new FrozenPlanExecutionError(
        `invalid concurrent plan ${request.plan.executionId}: ${
          validation.issues.map((issue) => issue.message).join("; ")
        }`,
      );
    }
    const stepIds = new Set(request.plan.steps.map((step) => step.id));
    for (const [stepIdText, releaseNs] of Object.entries(
      request.stepNotBeforeNs ?? {},
    )) {
      const stepId = Number(stepIdText);
      if (
        !Number.isSafeInteger(stepId)
        || !stepIds.has(stepId)
        || !Number.isSafeInteger(releaseNs)
        || releaseNs < request.arrivalNs
      ) {
        throw new FrozenPlanExecutionError(
          `invalid not-before constraint ${stepIdText}:${releaseNs} for ${request.plan.executionId}`,
        );
      }
      if (releaseNs > request.arrivalNs) {
        releaseEvents++;
      }
    }
    executionIds.add(request.plan.executionId);
    admissionOrders.add(request.admissionOrder);
    totalSteps += request.plan.steps.length;
  }
  const traceEvents = totalSteps + requests.length * 2;
  const runtimeEvents = traceEvents + releaseEvents;
  if (
    !Number.isSafeInteger(totalSteps)
    || !Number.isSafeInteger(traceEvents)
    || !Number.isSafeInteger(runtimeEvents)
    || runtimeEvents > scenario.execution.maxEvents
  ) {
    throw new FrozenPlanExecutionError(
      `concurrent plans require ${runtimeEvents} runtime events, limit is ${scenario.execution.maxEvents}`,
    );
  }
  const ordered = [...requests].sort((left, right) => (
    left.arrivalNs - right.arrivalNs
    || left.admissionOrder - right.admissionOrder
    || left.plan.executionId.localeCompare(right.plan.executionId)
  ));
  for (let index = 0; index < ordered.length; index++) {
    if (ordered[index].admissionOrder !== index) {
      throw new FrozenPlanExecutionError(
        "admission order must be the contiguous arrival-ordered sequence 0..N-1",
      );
    }
  }
  return ordered;
}

function validateStreamingAdmission(
  scenario: SimulationScenario,
  request: ConcurrentPlanRequest,
  duplicateExecutionId: boolean,
  expectedAdmissionOrder: number,
  priorRuntimeEvents: number,
): number {
  if (
    !Number.isSafeInteger(request.arrivalNs)
    || request.arrivalNs < 0
    || !Number.isSafeInteger(request.admissionOrder)
    || request.admissionOrder !== expectedAdmissionOrder
  ) {
    throw new FrozenPlanExecutionError(
      "streaming arrival must be non-negative and admission order must be contiguous",
    );
  }
  if (duplicateExecutionId) {
    throw new FrozenPlanExecutionError(
      `duplicate streaming execution id ${request.plan.executionId}`,
    );
  }
  const validation = validateFrozenPlan(scenario, request.plan);
  if (!validation.valid) {
    throw new FrozenPlanExecutionError(
      `invalid concurrent plan ${request.plan.executionId}: ${
        validation.issues.map((issue) => issue.message).join("; ")
      }`,
    );
  }
  const stepIds = new Set(request.plan.steps.map((step) => step.id));
  let releaseEvents = 0;
  for (const [stepIdText, releaseNs] of Object.entries(
    request.stepNotBeforeNs ?? {},
  )) {
    const stepId = Number(stepIdText);
    if (
      !Number.isSafeInteger(stepId)
      || !stepIds.has(stepId)
      || !Number.isSafeInteger(releaseNs)
      || releaseNs < request.arrivalNs
    ) {
      throw new FrozenPlanExecutionError(
        `invalid not-before constraint ${stepIdText}:${releaseNs} for ${request.plan.executionId}`,
      );
    }
    if (releaseNs > request.arrivalNs) {
      releaseEvents++;
    }
  }
  const runtimeEvents = priorRuntimeEvents
    + request.plan.steps.length
    + 2
    + releaseEvents;
  if (
    !Number.isSafeInteger(runtimeEvents)
    || runtimeEvents > scenario.execution.maxEvents
  ) {
    throw new FrozenPlanExecutionError(
      `concurrent plans require ${runtimeEvents} runtime events, limit is ${scenario.execution.maxEvents}`,
    );
  }
  return runtimeEvents;
}

function buildExecutionState(request: ConcurrentPlanRequest): ExecutionState {
  const dependents = new Map<number, number[]>();
  for (const step of request.plan.steps) {
    for (const dependency of step.dependencies) {
      const values = dependents.get(dependency) ?? [];
      values.push(step.id);
      dependents.set(dependency, values);
    }
  }
  const releasedSteps = new Set(request.plan.steps
    .filter((step) => (
      (request.stepNotBeforeNs?.[step.id] ?? request.arrivalNs)
        <= request.arrivalNs
    ))
    .map((step) => step.id));
  return {
    request,
    stepMap: new Map(request.plan.steps.map((step) => [step.id, step])),
    planOrder: new Map(
      request.plan.steps.map((step, index) => [step.id, index]),
    ),
    dependents,
    remainingDependencies: new Map(
      request.plan.steps.map((step) => [
        step.id,
        step.dependencies.length,
      ]),
    ),
    releasedSteps,
    ready: request.plan.steps
      .filter((step) => (
        step.dependencies.length === 0 && releasedSteps.has(step.id)
      ))
      .map((step) => step.id),
    completedSteps: new Set(),
    localOperations: [],
    rankCompletions: new Map(),
    admitted: false,
  };
}

function compareReadyCandidates(
  left: { readonly state: ExecutionState; readonly stepId: number },
  right: { readonly state: ExecutionState; readonly stepId: number },
): number {
  return left.state.request.admissionOrder
    - right.state.request.admissionOrder
    || (left.state.planOrder.get(left.stepId) ?? 0)
      - (right.state.planOrder.get(right.stepId) ?? 0)
    || left.state.request.plan.executionId.localeCompare(
      right.state.request.plan.executionId,
    );
}

function compareArbitration(
  left: {
    readonly phase: number;
    readonly admissionOrder: number;
    readonly planOrder: number;
  },
  right: {
    readonly phase: number;
    readonly admissionOrder: number;
    readonly planOrder: number;
  },
): number {
  return left.phase - right.phase
    || left.admissionOrder - right.admissionOrder
    || left.planOrder - right.planOrder;
}

function successfulExecutionResult(
  state: ExecutionState,
  completedAtNs: number,
): FrozenPlanExecutionResult {
  const terminal = successfulTerminal(
    state.request.plan,
    state.localOperations,
  );
  if (terminal.timestampNs !== completedAtNs) {
    throw new FrozenPlanExecutionError(
      `${state.request.plan.executionId} completed at ${completedAtNs}ns but terminal is ${terminal.timestampNs}ns`,
    );
  }
  const rankCompletions = terminal.rankStates.map((rank): RankCompletion => ({
    rankId: rank.rankId,
    completedAtNs: rank.terminalAtNs,
  }));
  const trace: PlanExecutionTrace = {
    operations: state.localOperations,
    terminal,
  };
  return {
    status: "succeeded",
    executionId: state.request.plan.executionId,
    completedAtNs,
    trace,
    rankCompletions,
    rankStates: terminal.rankStates,
  };
}

function successfulTerminal(
  plan: FrozenPlan,
  operations: readonly PlanTraceEvent[],
): PlanTerminalEvent {
  const rankTimes = new Map<string, number>();
  for (const event of operations) {
    for (const rankId of event.participants) {
      rankTimes.set(
        rankId,
        Math.max(rankTimes.get(rankId) ?? 0, event.finishNs),
      );
    }
  }
  const rankStates = [...rankTimes.entries()]
    .map(([rankId, terminalAtNs]): RankTerminalState => ({
      rankId,
      status: "succeeded",
      terminalAtNs,
    }))
    .sort((left, right) => left.rankId.localeCompare(right.rankId));
  return {
    contractRevision: PLAN_CONTRACT_REVISION,
    executionId: plan.executionId,
    topologyEpoch: plan.topologyEpoch,
    sourceSequence: operations.length,
    kind: "execution_terminal",
    status: "succeeded",
    timestampNs: operations.reduce(
      (maximum, event) => Math.max(maximum, event.finishNs),
      0,
    ),
    rankStates,
  };
}

function validateConcurrentNodeFailure(
  scenario: SimulationScenario,
  requests: readonly ConcurrentPlanRequest[],
  fault: ConcurrentNodeFailure,
): ConcurrentPlanRequest[] {
  const ordered = validateConcurrentRequests(scenario, requests);
  if (
    !Number.isSafeInteger(fault.atNs)
    || fault.atNs <= 0
    || typeof fault.nodeId !== "string"
    || fault.nodeId.length === 0
    || typeof fault.reason !== "string"
    || fault.reason.length === 0
    || !scenario.devices.some((device) => device.nodeId === fault.nodeId)
  ) {
    throw new FrozenPlanExecutionError(
      "concurrent node fault must identify a known node, positive safe time, and non-empty reason",
    );
  }
  const rankDevices = buildRankDeviceMap(scenario.groups);
  const deviceNodes = new Map(
    scenario.devices.map((device) => [device.id, device.nodeId]),
  );
  for (const request of ordered) {
    if (request.arrivalNs >= fault.atNs) {
      throw new FrozenPlanExecutionError(
        `execution ${request.plan.executionId} is not admitted before node fault at ${fault.atNs}ns`,
      );
    }
    if (!planRanks(request.plan).some((rankId) => (
      deviceNodes.get(rankDevices.get(rankId) ?? "") === fault.nodeId
    ))) {
      throw new FrozenPlanExecutionError(
        `execution ${request.plan.executionId} does not participate on failed node ${fault.nodeId}`,
      );
    }
  }
  return ordered;
}

function nodeFailureTerminal(
  scenario: SimulationScenario,
  plan: FrozenPlan,
  operations: readonly PlanTraceEvent[],
  fault: ConcurrentNodeFailure,
): PlanTerminalEvent {
  const submittedStepIds = new Set(operations.map((event) => event.stepId));
  const unsubmittedStepIds = plan.steps
    .filter((step) => !submittedStepIds.has(step.id))
    .map((step) => step.id);
  const quiescedAtNs = operations.reduce(
    (maximum, event) => Math.max(maximum, event.finishNs),
    fault.atNs,
  );
  const rankCompletions = new Map<string, number>();
  for (const event of operations) {
    for (const rankId of event.participants) {
      rankCompletions.set(
        rankId,
        Math.max(rankCompletions.get(rankId) ?? 0, event.finishNs),
      );
    }
  }
  const rankDevices = buildRankDeviceMap(scenario.groups);
  const deviceNodes = new Map(
    scenario.devices.map((device) => [device.id, device.nodeId]),
  );
  const failedRanks = new Set(planRanks(plan).filter((rankId) => (
    deviceNodes.get(rankDevices.get(rankId) ?? "") === fault.nodeId
  )));
  const rankStates = planRanks(plan).map((rankId): RankTerminalState => {
    const rankStepIds = plan.steps
      .filter((step) => step.participants.includes(rankId))
      .map((step) => step.id);
    const allStepsSubmitted = rankStepIds.every(
      (stepId) => submittedStepIds.has(stepId),
    );
    const completedAtNs = rankCompletions.get(rankId) ?? 0;
    if (allStepsSubmitted && completedAtNs < fault.atNs) {
      return { rankId, status: "succeeded", terminalAtNs: completedAtNs };
    }
    if (failedRanks.has(rankId)) {
      return { rankId, status: "failed", terminalAtNs: fault.atNs };
    }
    if (allStepsSubmitted) {
      return { rankId, status: "succeeded", terminalAtNs: completedAtNs };
    }
    return { rankId, status: "aborted", terminalAtNs: quiescedAtNs };
  });
  return {
    contractRevision: PLAN_CONTRACT_REVISION,
    executionId: plan.executionId,
    topologyEpoch: plan.topologyEpoch,
    sourceSequence: operations.length,
    kind: "execution_terminal",
    status: "failed",
    timestampNs: quiescedAtNs,
    failureAtNs: fault.atNs,
    reason: fault.reason,
    fault: { ...fault },
    unsubmittedStepIds,
    rankStates,
  };
}

function replayConcurrentNodeFailureTerminals(
  scenario: SimulationScenario,
  ordered: readonly ConcurrentPlanRequest[],
  trace: ConcurrentPlanExecutionTrace,
  fault: ConcurrentNodeFailure,
  eventsByExecution: ReadonlyMap<
    string,
    ReadonlyMap<number, PlanTraceEvent>
  >,
  priorGroupOwner: ReadonlyMap<string, string>,
  finalCollectiveSubmission: ReadonlyMap<string, number>,
): ConcurrentPlanReplayResult {
  validateConcurrentNodeFailure(scenario, ordered, fault);
  if (trace.operations.some(({ event }) => event.submittedAtNs >= fault.atNs)) {
    throw new PlanReplayError(
      "concurrent operation was submitted at or after node fault",
    );
  }
  const expectedTerminals: PlanTerminalEvent[] = [];
  const replayExecutions: PlanReplayResult[] = [];
  for (const request of ordered) {
    const eventMap = eventsByExecution.get(request.plan.executionId);
    if (!eventMap) {
      throw new PlanReplayError(
        `missing operation prefix for ${request.plan.executionId}`,
      );
    }
    for (const step of request.plan.steps) {
      if (eventMap.has(step.id)) {
        continue;
      }
      let readyAtNs = Math.max(
        request.arrivalNs,
        request.stepNotBeforeNs?.[step.id] ?? request.arrivalNs,
      );
      let dependenciesComplete = true;
      for (const dependency of step.dependencies) {
        const dependencyEvent = eventMap.get(dependency);
        if (!dependencyEvent) {
          dependenciesComplete = false;
          break;
        }
        readyAtNs = Math.max(readyAtNs, dependencyEvent.finishNs);
      }
      if (!dependenciesComplete) {
        continue;
      }
      if (step.operation.kind === "collective") {
        const priorOwner = priorGroupOwner.get(identityKey(
          request.plan.executionId,
          step.operation.groupId,
        ));
        if (priorOwner) {
          const priorFinal = finalCollectiveSubmission.get(identityKey(
            priorOwner,
            step.operation.groupId,
          ));
          if (priorFinal === undefined) {
            continue;
          }
          readyAtNs = Math.max(readyAtNs, priorFinal);
        }
      }
      if (readyAtNs < fault.atNs) {
        throw new PlanReplayError(
          `${request.plan.executionId}/${step.id} was ready before node fault but omitted`,
        );
      }
    }
    const operations = [...eventMap.values()]
      .sort((left, right) => left.sourceSequence - right.sourceSequence);
    if (
      operations.length === request.plan.steps.length
      && operations.every((event) => event.finishNs < fault.atNs)
    ) {
      throw new PlanReplayError(
        `${request.plan.executionId} completed before node fault`,
      );
    }
    const terminal = nodeFailureTerminal(
      scenario,
      request.plan,
      operations,
      fault,
    );
    expectedTerminals.push(terminal);
    replayExecutions.push({
      status: "failed",
      appliedEvents: operations.length + 1,
      completedAtNs: terminal.timestampNs,
      rankCompletions: terminal.rankStates
        .filter((state) => state.status === "succeeded")
        .map((state) => ({
          rankId: state.rankId,
          completedAtNs: state.terminalAtNs,
        })),
      rankStates: terminal.rankStates,
    });
  }
  expectedTerminals.sort(compareTerminals(ordered));
  if (!terminalsEqual(trace.terminals, expectedTerminals)) {
    throw new PlanReplayError(
      "concurrent node-failure terminals do not match global quiescence",
    );
  }
  const completedAtNs = expectedTerminals.reduce(
    (maximum, terminal) => Math.max(maximum, terminal.timestampNs),
    fault.atNs,
  );
  return {
    appliedEvents:
      trace.admissions.length + trace.operations.length + trace.terminals.length,
    completedAtNs,
    maximumConcurrentExecutions: maximumConcurrency(
      trace.admissions,
      trace.terminals,
    ),
    executions: replayExecutions,
  };
}

function collectiveCounts(plan: FrozenPlan): Readonly<Record<string, number>> {
  const counts = Object.create(null) as Record<string, number>;
  for (const step of plan.steps) {
    if (step.operation.kind === "collective") {
      counts[step.operation.groupId] =
        (counts[step.operation.groupId] ?? 0) + 1;
    }
  }
  return counts;
}

function sequencerForEpoch(topologyEpoch: number): CollectiveSubmitSequencer {
  const sequencer = new CollectiveSubmitSequencer();
  if (topologyEpoch > 0) {
    sequencer.advanceTopologyEpoch(topologyEpoch);
  }
  return sequencer;
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

function resourceIds(operation: PlanOperation): readonly string[] {
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
  return resourceIds(operation).map((resourceId) => {
    const lanes = resourceLanes.get(resourceId);
    if (!lanes || lanes.length === 0) {
      throw new FrozenPlanExecutionError(`missing resource ${resourceId}`);
    }
    let resourceLane = 0;
    for (let index = 1; index < lanes.length; index++) {
      if (lanes[index] < lanes[resourceLane]) {
        resourceLane = index;
      }
    }
    return { resourceId, resourceLane };
  });
}

function reserveResources(
  resourceLanes: Map<string, number[]>,
  resources: readonly PlanResourceReservation[],
  finishNs: number,
): void {
  for (const reservation of resources) {
    const lanes = resourceLanes.get(reservation.resourceId);
    if (!lanes || reservation.resourceLane >= lanes.length) {
      throw new FrozenPlanExecutionError(
        `invalid resource reservation ${reservation.resourceId}#${reservation.resourceLane}`,
      );
    }
    lanes[reservation.resourceLane] = finishNs;
  }
}

interface LeaseInterval {
  readonly executionId: string;
  readonly stepId: number;
  readonly startNs: number;
  readonly finishNs: number;
  readonly mode: "read" | "write";
}

class ConcurrentLeaseTimeline {
  private readonly intervals = new Map<string, LeaseInterval[]>();

  earliestStart(
    step: PlanStep,
    earliestNs: number,
    durationNs: number,
    completedThroughNs: number,
  ): number {
    for (const [allocationId] of accessModes(step)) {
      const active = (this.intervals.get(allocationId) ?? []).filter(
        (interval) => interval.finishNs > completedThroughNs,
      );
      if (active.length === 0) {
        this.intervals.delete(allocationId);
      } else {
        this.intervals.set(allocationId, active);
      }
    }
    let startNs = earliestNs;
    while (true) {
      const finishNs = checkedAdd(startNs, durationNs, "lease finish");
      let delayedUntilNs = startNs;
      for (const [allocationId, mode] of accessModes(step)) {
        for (const interval of this.intervals.get(allocationId) ?? []) {
          if (
            (mode === "write" || interval.mode === "write")
            && startNs < interval.finishNs
            && interval.startNs < finishNs
          ) {
            delayedUntilNs = Math.max(delayedUntilNs, interval.finishNs);
          }
        }
      }
      if (delayedUntilNs === startNs) {
        return startNs;
      }
      startNs = delayedUntilNs;
    }
  }

  reserve(
    executionId: string,
    step: PlanStep,
    startNs: number,
    finishNs: number,
  ): void {
    for (const [allocationId, mode] of accessModes(step)) {
      const intervals = this.intervals.get(allocationId) ?? [];
      if (intervals.some((interval) => (
        (mode === "write" || interval.mode === "write")
        && startNs < interval.finishNs
        && interval.startNs < finishNs
      ))) {
        throw new FrozenPlanExecutionError(
          `${executionId}/${step.id} overlaps a ${mode} lease on ${allocationId}`,
        );
      }
      intervals.push({
        executionId,
        stepId: step.id,
        startNs,
        finishNs,
        mode,
      });
      this.intervals.set(allocationId, intervals);
    }
  }
}

function accessModes(
  step: PlanStep,
): Array<readonly [string, "read" | "write"]> {
  return [
    ...step.reads.map(
      (allocation): readonly [string, "read"] => [allocation, "read"],
    ),
    ...step.writes.map(
      (allocation): readonly [string, "write"] => [allocation, "write"],
    ),
  ];
}

function buildTraceEvent(
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
    resources: resources.map((reservation) => ({ ...reservation })),
    participants: [...step.participants],
    reads: [...step.reads],
    writes: [...step.writes],
    ...(step.operation.kind === "collective"
      ? {
          groupId: step.operation.groupId,
          commSequenceId: step.operation.commSequenceId,
          collectiveAlgorithm: step.operation.algorithm,
        }
      : {}),
  };
}

function assertEventMatchesStep(
  plan: FrozenPlan,
  step: PlanStep,
  event: PlanTraceEvent,
): void {
  if (
    event.contractRevision !== PLAN_CONTRACT_REVISION
    || event.executionId !== plan.executionId
    || event.topologyEpoch !== plan.topologyEpoch
    || event.kind !== step.operation.kind
    || !arraysEqual(event.participants, step.participants)
    || !arraysEqual(event.reads, step.reads)
    || !arraysEqual(event.writes, step.writes)
  ) {
    replayFail(`event payload does not match ${plan.executionId}/${step.id}`);
  }
  if (step.operation.kind === "collective") {
    if (
      event.groupId !== step.operation.groupId
      || event.commSequenceId !== step.operation.commSequenceId
      || event.collectiveAlgorithm !== step.operation.algorithm
    ) {
      replayFail(
        `collective metadata does not match ${plan.executionId}/${step.id}`,
      );
    }
  } else if (
    event.groupId !== undefined
    || event.commSequenceId !== undefined
    || event.collectiveAlgorithm !== undefined
  ) {
    replayFail(
      `non-collective ${plan.executionId}/${step.id} has collective metadata`,
    );
  }
}

function priorGroupOwners(
  ordered: readonly ConcurrentPlanRequest[],
): ReadonlyMap<string, string> {
  const result = new Map<string, string>();
  const latest = new Map<string, string>();
  for (const request of ordered) {
    for (const groupId of Object.keys(collectiveCounts(request.plan)).sort()) {
      const prior = latest.get(groupId);
      if (prior) {
        result.set(identityKey(request.plan.executionId, groupId), prior);
      }
      latest.set(groupId, request.plan.executionId);
    }
  }
  return result;
}

function admissionFor(request: ConcurrentPlanRequest): ConcurrentPlanAdmission {
  return {
    executionId: request.plan.executionId,
    arrivalNs: request.arrivalNs,
    admissionOrder: request.admissionOrder,
  };
}

function admissionsEqual(
  actual: readonly ConcurrentPlanAdmission[],
  expected: readonly ConcurrentPlanAdmission[],
): boolean {
  return actual.length === expected.length
    && actual.every((entry, index) => (
      entry.executionId === expected[index].executionId
      && entry.arrivalNs === expected[index].arrivalNs
      && entry.admissionOrder === expected[index].admissionOrder
    ));
}

function reservationsEqual(
  actual: readonly PlanResourceReservation[],
  expected: readonly PlanResourceReservation[],
): boolean {
  return actual.length === expected.length
    && actual.every((entry, index) => (
      entry.resourceId === expected[index].resourceId
      && entry.resourceLane === expected[index].resourceLane
    ));
}

function terminalsEqual(
  actual: readonly PlanTerminalEvent[],
  expected: readonly PlanTerminalEvent[],
): boolean {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

function compareTerminals(
  ordered: readonly ConcurrentPlanRequest[],
): (left: PlanTerminalEvent, right: PlanTerminalEvent) => number {
  const admissionOrder = new Map(
    ordered.map((request) => [
      request.plan.executionId,
      request.admissionOrder,
    ]),
  );
  return (left, right) => (
    left.timestampNs - right.timestampNs
    || (admissionOrder.get(left.executionId) ?? 0)
      - (admissionOrder.get(right.executionId) ?? 0)
    || left.executionId.localeCompare(right.executionId)
  );
}

function maximumConcurrency(
  admissions: readonly ConcurrentPlanAdmission[],
  terminals: readonly PlanTerminalEvent[],
): number {
  const arrivalByExecution = new Map(
    admissions.map((admission) => [
      admission.executionId,
      admission.arrivalNs,
    ]),
  );
  const points = [
    ...admissions.map((admission) => ({
      atNs: admission.arrivalNs,
      delta: 1,
      phase: 1,
      order: admission.admissionOrder,
    })),
    ...terminals.map((terminal) => ({
      atNs: terminal.timestampNs,
      delta: -1,
      phase: arrivalByExecution.get(terminal.executionId)
          === terminal.timestampNs
        ? 2
        : 0,
      order: Number.MAX_SAFE_INTEGER,
    })),
  ].sort((left, right) => (
    left.atNs - right.atNs
    || left.phase - right.phase
    || left.order - right.order
  ));
  let active = 0;
  let maximum = 0;
  for (const point of points) {
    active += point.delta;
    maximum = Math.max(maximum, active);
  }
  return maximum;
}

function validateCampaignOptions(options: ConcurrentPlanCampaignOptions): void {
  if (
    !Number.isSafeInteger(options.executionCount)
    || options.executionCount <= 0
    || !Number.isSafeInteger(options.seed)
    || options.seed < 0
    || options.seed > 0xffff_ffff
    || !Number.isSafeInteger(options.arrivalWindowNs)
    || options.arrivalWindowNs < 0
    || options.arrivalWindowNs >= Number.MAX_SAFE_INTEGER
  ) {
    throw new FrozenPlanExecutionError(
      "campaign count must be positive, seed must be uint32, and arrival window must be a non-negative safe integer below MAX_SAFE_INTEGER",
    );
  }
}

function validateCampaignInputs(
  scenario: SimulationScenario,
  template: FrozenPlan,
  options: ConcurrentPlanCampaignOptions,
): void {
  validateCampaignOptions(options);
  const validation = validateFrozenPlan(scenario, template);
  if (!validation.valid) {
    throw new FrozenPlanExecutionError(
      `invalid campaign template: ${
        validation.issues.map((issue) => issue.message).join("; ")
      }`,
    );
  }
  const completionEvents = options.executionCount * template.steps.length;
  const traceEvents = completionEvents + options.executionCount * 2;
  if (
    !Number.isSafeInteger(completionEvents)
    || !Number.isSafeInteger(traceEvents)
    || traceEvents > scenario.execution.maxEvents
  ) {
    throw new FrozenPlanExecutionError(
      `campaign requires ${traceEvents} trace events, limit is ${scenario.execution.maxEvents}`,
    );
  }
}

function buildSeededRequests(
  template: FrozenPlan,
  options: ConcurrentPlanCampaignOptions,
): ConcurrentPlanRequest[] {
  const random = new SeededRandom(options.seed);
  const admissionTieBreakers = shuffledIndices(
    options.executionCount,
    random,
  );
  const arrivalRange = checkedAdd(
    options.arrivalWindowNs,
    1,
    "campaign arrival range",
  );
  const arrivals = Array.from({ length: options.executionCount }, () => (
    Math.floor(random.nextFloat() * arrivalRange)
  ));
  const firstArrival = arrivals.reduce(
    (minimum, arrivalNs) => Math.min(minimum, arrivalNs),
    Number.MAX_SAFE_INTEGER,
  );
  const normalizedArrivals = arrivals.map(
    (arrivalNs) => arrivalNs - firstArrival,
  );
  const admissionOrderByIndex = new Array<number>(options.executionCount);
  [...normalizedArrivals.keys()]
    .sort((left, right) => (
      normalizedArrivals[left] - normalizedArrivals[right]
      || admissionTieBreakers[left] - admissionTieBreakers[right]
      || left - right
    ))
    .forEach((requestIndex, admissionOrder) => {
      admissionOrderByIndex[requestIndex] = admissionOrder;
    });
  return arrivals.map((_arrivalNs, index): ConcurrentPlanRequest => ({
    plan: {
      ...template,
      id: `${template.id}:campaign:${index}`,
      executionId: `${template.executionId}:campaign:${index}`,
      steps: template.steps,
    },
    arrivalNs: normalizedArrivals[index],
    admissionOrder: admissionOrderByIndex[index],
  }));
}

function shuffledIndices(
  count: number,
  random: SeededRandom,
): number[] {
  const values = Array.from({ length: count }, (_, index) => index);
  for (let index = values.length - 1; index > 0; index--) {
    const swap = Math.floor(random.nextFloat() * (index + 1));
    [values[index], values[swap]] = [values[swap], values[index]];
  }
  const ranks = new Array<number>(count);
  for (let rank = 0; rank < values.length; rank++) {
    ranks[values[rank]] = rank;
  }
  return ranks;
}

class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  nextFloat(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  }
}

function arraysEqual<T>(
  left: readonly T[],
  right: readonly T[],
): boolean {
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
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

function planRanks(plan: FrozenPlan): string[] {
  return [...new Set(plan.steps.flatMap((step) => step.participants))].sort();
}

function identityKey(...parts: readonly (string | number)[]): string {
  return JSON.stringify(parts);
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
