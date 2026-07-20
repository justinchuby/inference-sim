import {
  DiscreteEventSimulator,
  type ScheduledEvent,
} from "./event-loop.js";
import {
  SpeculativeTransactionSimulator,
} from "./speculative.js";
import {
  buildSpeculativeStateGroups,
  speculativeFamilyContract,
  validateSpeculativeEligibility,
  type SpeculativeEligibility,
  type SpeculativeProposalPrefix,
  type SpeculativeProposerFamily,
} from "./speculative-family.js";
import {
  SpeculativeAcceptanceCursor,
  validateAcceptanceCoverage,
  type SpeculativeAcceptanceModel,
} from "./speculative-acceptance.js";
import {
  decideSpeculativeIteration,
  planSpeculativeProposal,
} from "./speculative-iteration.js";

export const SERVING_TRACE_CONTRACT_REVISION = 3;

export interface ServingRequestSpec {
  readonly id: string;
  readonly arrivalNs: number;
  readonly promptTokens: number;
  readonly outputTokens: number;
  readonly priority?: number;
}

export interface ServingSchedulerConfig {
  readonly requests: readonly ServingRequestSpec[];
  readonly maxBatchSize: number;
  readonly maxBatchTokens: number;
  readonly prefillChunkTokens: number;
  readonly maxKvTokens: number;
  readonly speculative?: ServingSpeculativeConfig;
  readonly maxEvents?: number;
}

export type ServingSpeculativeAcceptanceModel =
  | {
      readonly kind: "replay";
      readonly acceptedDraftTokensByRequest: Readonly<
        Record<string, readonly number[]>
      >;
    }
  | {
      readonly kind: "conditional_empirical";
      readonly matchProbabilityByPosition: readonly number[];
      readonly seed: number;
    }
  | {
      readonly kind: "conditional_heuristic";
      readonly matchProbabilityByPosition: readonly number[];
      readonly seed: number;
    };

export interface ServingSpeculativeConfig {
  readonly family: SpeculativeProposerFamily;
  readonly eligibility: SpeculativeEligibility;
  readonly maxAdditionalTokens: number;
  readonly acceptance: ServingSpeculativeAcceptanceModel;
}

export interface ServingPrefillSlice {
  readonly requestId: string;
  readonly tokens: number;
}

export interface ServingDecodeSlice {
  readonly requestId: string;
  readonly mode: "target_only" | "speculative";
  readonly guaranteedTargetTokens: 0 | 1;
  readonly proposedAdditionalTokens: number;
  readonly acceptedAdditionalTokens: number;
  readonly proposedDraftTokens: number;
  readonly acceptedDraftTokens: number;
  readonly targetAuthoritativeTokens: 0 | 1;
  readonly committedTokens: number;
  readonly targetTokenWidth: number;
  readonly outcome:
    | "target_only"
    | "correction"
    | "bonus"
    | "accepted_tail";
}

export interface ServingBatchWork {
  readonly batchId: number;
  readonly prefill: readonly ServingPrefillSlice[];
  readonly decode: readonly ServingDecodeSlice[];
  readonly tokenWork: number;
  readonly sequenceCount: number;
  readonly expectedOutputTokens: number;
}

export type ServingBatchDurationEstimator = (
  batch: ServingBatchWork,
  startedAtNs: number,
) => number;

export interface ServingTokenEmission {
  readonly requestId: string;
  readonly tokenIndex: number;
  readonly source:
    | "prefill"
    | "target_decode"
    | "target_guaranteed"
    | "speculative_accepted"
    | "speculative_authoritative";
}

interface ServingTraceBase {
  readonly contractRevision: typeof SERVING_TRACE_CONTRACT_REVISION;
  readonly sourceSequence: number;
  readonly atNs: number;
}

export type ServingTraceEvent =
  | ServingTraceBase & {
      readonly kind: "arrival";
      readonly requestId: string;
    }
  | ServingTraceBase & {
      readonly kind: "batch_start";
      readonly batch: ServingBatchWork;
      readonly durationNs: number;
      readonly kvTokensBefore: number;
      readonly transientKvTokens: number;
    }
  | ServingTraceBase & {
      readonly kind: "batch_finish";
      readonly batchId: number;
      readonly emittedTokens: readonly ServingTokenEmission[];
      readonly completedRequestIds: readonly string[];
      readonly kvTokensAfter: number;
    }
  | ServingTraceBase & {
      readonly kind: "terminal";
      readonly completedRequests: number;
      readonly totalOutputTokens: number;
      readonly kvTokensAfter: 0;
    };

type ServingTraceInput = ServingTraceEvent extends infer Event
  ? Event extends ServingTraceEvent
    ? Omit<Event, "contractRevision" | "sourceSequence">
    : never
  : never;

export interface ServingRequestResult {
  readonly id: string;
  readonly arrivalNs: number;
  readonly firstTokenNs: number;
  readonly completedAtNs: number;
  readonly timeToFirstTokenNs: number;
  readonly latencyNs: number;
  readonly tokenTimestampsNs: readonly number[];
}

export interface ServingMetrics {
  readonly requests: number;
  readonly batches: number;
  readonly prefillTokens: number;
  readonly decodeTokens: number;
  readonly outputTokens: number;
  readonly targetForwards: number;
  readonly targetVerificationTokens: number;
  readonly guaranteedTargetTokens: number;
  readonly proposedAdditionalTokens: number;
  readonly acceptedAdditionalTokens: number;
  readonly proposedDraftTokens: number;
  readonly acceptedDraftTokens: number;
  readonly rejectedDraftTokens: number;
  readonly targetAuthoritativeTokens: number;
  readonly committedTokensPerTargetForward: number;
  readonly totalDurationNs: number;
  readonly batchServiceNs: number;
  readonly throughputTokensPerSecond: number;
  readonly averageTimeToFirstTokenNs: number;
  readonly p50TimeToFirstTokenNs: number;
  readonly p95TimeToFirstTokenNs: number;
  readonly averageInterTokenLatencyNs: number;
  readonly p50InterTokenLatencyNs: number;
  readonly p95InterTokenLatencyNs: number;
  readonly averageRequestLatencyNs: number;
  readonly sequenceBatchUtilization: number;
  readonly tokenBatchUtilization: number;
  readonly kvHighWaterTokens: number;
}

export interface ServingReplayResult {
  readonly appliedEvents: number;
  readonly completedRequests: number;
  readonly finalKvTokens: number;
}

export interface ServingSimulationResult {
  readonly trace: readonly ServingTraceEvent[];
  readonly requests: readonly ServingRequestResult[];
  readonly metrics: ServingMetrics;
  readonly replay: ServingReplayResult;
}

export class ServingProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ServingProtocolError";
  }
}

type RequestPhase = "unarrived" | "waiting" | "prefilling" | "decoding" | "completed";

interface MutableRequest {
  spec: ServingRequestSpec;
  phase: RequestPhase;
  promptProcessed: number;
  outputEmitted: number;
  kvTokens: number;
  tokenTimestampsNs: number[];
  completedAtNs?: number;
  speculative?: {
    transaction: SpeculativeTransactionSimulator;
    acceptance: SpeculativeAcceptanceCursor;
    proposalPrefix: SpeculativeProposalPrefix;
  };
}

interface RunningBatch {
  work: ServingBatchWork;
  startedAtNs: number;
  durationNs: number;
}

type InternalEvent =
  | { readonly kind: "arrival"; readonly requestId: string }
  | { readonly kind: "dispatch" }
  | { readonly kind: "batch_finish"; readonly batchId: number };

export const DEFAULT_SERVING_BATCH_DURATION: ServingBatchDurationEstimator = (
  batch,
) => checkedAdd(
  25_000,
  checkedAdd(
    checkedMultiply(
      batch.prefill.reduce((sum, entry) => sum + entry.tokens, 0),
      35_000,
      "default prefill duration",
    ),
    checkedAdd(
      checkedMultiply(
        batch.decode.reduce(
          (sum, entry) => sum + entry.targetTokenWidth,
          0,
        ),
        70_000,
        "default decode duration",
      ),
      checkedMultiply(
        batch.decode.reduce(
          (sum, entry) => sum + entry.proposedAdditionalTokens,
          0,
        ),
        25_000,
        "default proposer duration",
      ),
      "default decode/proposer duration",
    ),
    "default batch token duration",
  ),
  "default batch duration",
);

export function simulateServingWorkload(
  config: ServingSchedulerConfig,
  estimateDuration: ServingBatchDurationEstimator =
    DEFAULT_SERVING_BATCH_DURATION,
): ServingSimulationResult {
  validateServingConfig(config);
  const engine = new ServingSimulator(config, estimateDuration);
  const result = engine.run();
  const replay = replayServingTrace(config, result.trace, estimateDuration);
  return { ...result, replay };
}

class ServingSimulator {
  private readonly eventLoop = new DiscreteEventSimulator<InternalEvent>();
  private readonly requests = new Map<string, MutableRequest>();
  private readonly traceEvents: ServingTraceEvent[] = [];
  private sourceSequence = 0;
  private nextBatchId = 0;
  private running?: RunningBatch;
  private arrivedRequests = 0;
  private completedRequests = 0;
  private kvTokens = 0;
  private kvHighWaterTokens = 0;
  private batchServiceNs = 0;
  private scheduledTokenWork = 0;
  private scheduledSequenceWork = 0;
  private targetForwards = 0;
  private targetVerificationTokens = 0;
  private guaranteedTargetTokens = 0;
  private proposedAdditionalTokens = 0;
  private acceptedAdditionalTokens = 0;
  private proposedDraftTokens = 0;
  private acceptedDraftTokens = 0;
  private targetAuthoritativeTokens = 0;
  private dispatchAtNs?: number;

  constructor(
    private readonly config: ServingSchedulerConfig,
    private readonly estimateDuration: ServingBatchDurationEstimator,
  ) {
    for (const spec of sortedRequests(config.requests)) {
      this.requests.set(spec.id, createMutableRequest(config, spec));
      this.eventLoop.scheduleAt(spec.arrivalNs, {
        kind: "arrival",
        requestId: spec.id,
      });
    }
  }

  run(): Omit<ServingSimulationResult, "replay"> {
    this.eventLoop.run(
      (event) => this.handleEvent(event),
      { maxEvents: this.config.maxEvents ?? 1_000_000 },
    );
    if (this.completedRequests !== this.config.requests.length) {
      throw new ServingProtocolError(
        `simulation ended with ${this.completedRequests}/${this.config.requests.length} requests completed`,
      );
    }
    const requestResults = this.requestResults();
    return {
      trace: this.traceEvents,
      requests: requestResults,
      metrics: this.metrics(requestResults),
    };
  }

  private handleEvent(event: ScheduledEvent<InternalEvent>): void {
    if (event.payload.kind === "arrival") {
      this.handleArrival(event.payload.requestId, event.timestampNs);
    } else if (event.payload.kind === "batch_finish") {
      this.handleBatchFinish(event.payload.batchId, event.timestampNs);
    } else {
      if (this.dispatchAtNs !== event.timestampNs) {
        throw new ServingProtocolError(
          `stale serving dispatch at ${event.timestampNs}ns`,
        );
      }
      this.dispatchAtNs = undefined;
      this.maybeStartBatch(event.timestampNs);
    }
  }

  private handleArrival(requestId: string, atNs: number): void {
    const state = requireRequest(this.requests, requestId);
    if (state.phase !== "unarrived" || state.spec.arrivalNs !== atNs) {
      throw new ServingProtocolError(`invalid arrival for ${requestId}`);
    }
    state.phase = "waiting";
    this.arrivedRequests++;
    this.emit({
      kind: "arrival",
      atNs,
      requestId,
    });
    if (!this.running && this.dispatchAtNs !== atNs) {
      this.dispatchAtNs = atNs;
      this.eventLoop.scheduleAt(atNs, { kind: "dispatch" });
    }
  }

  private maybeStartBatch(atNs: number): void {
    if (this.running) {
      return;
    }
    const work = selectServingBatch(
      this.config,
      this.requests,
      this.kvTokens,
      this.nextBatchId,
    );
    if (!work) {
      if (
        [...this.requests.values()].some((request) => (
          request.phase !== "unarrived" && request.phase !== "completed"
        ))
      ) {
        throw new ServingProtocolError(
          `serving scheduler deadlocked at ${atNs}ns with ${this.kvTokens}/${this.config.maxKvTokens} KV tokens`,
        );
      }
      return;
    }
    const durationNs = this.estimateDuration(work, atNs);
    assertPositiveSafeInteger(durationNs, `batch ${work.batchId} duration`);
    const finishNs = checkedAdd(atNs, durationNs, "batch finish time");
    for (const entry of work.prefill) {
      const request = requireRequest(this.requests, entry.requestId);
      request.phase = "prefilling";
    }
    this.running = { work, startedAtNs: atNs, durationNs };
    this.nextBatchId++;
    this.batchServiceNs = checkedAdd(
      this.batchServiceNs,
      durationNs,
      "batch service time",
    );
    this.scheduledTokenWork = checkedAdd(
      this.scheduledTokenWork,
      work.tokenWork,
      "scheduled token work",
    );
    this.scheduledSequenceWork = checkedAdd(
      this.scheduledSequenceWork,
      work.sequenceCount,
      "scheduled sequence work",
    );
    this.targetForwards = checkedAdd(
      this.targetForwards,
      work.decode.length,
      "target forwards",
    );
    this.targetVerificationTokens = checkedAdd(
      this.targetVerificationTokens,
      sumDecode(work, (entry) => entry.targetTokenWidth),
      "target verification tokens",
    );
    this.guaranteedTargetTokens = checkedAdd(
      this.guaranteedTargetTokens,
      sumDecode(work, (entry) => entry.guaranteedTargetTokens),
      "guaranteed target tokens",
    );
    this.proposedAdditionalTokens = checkedAdd(
      this.proposedAdditionalTokens,
      sumDecode(work, (entry) => entry.proposedAdditionalTokens),
      "proposed additional tokens",
    );
    this.acceptedAdditionalTokens = checkedAdd(
      this.acceptedAdditionalTokens,
      sumDecode(work, (entry) => entry.acceptedAdditionalTokens),
      "accepted additional tokens",
    );
    this.proposedDraftTokens = checkedAdd(
      this.proposedDraftTokens,
      sumDecode(work, (entry) => entry.proposedDraftTokens),
      "proposed draft tokens",
    );
    this.acceptedDraftTokens = checkedAdd(
      this.acceptedDraftTokens,
      sumDecode(work, (entry) => entry.acceptedDraftTokens),
      "accepted draft tokens",
    );
    this.targetAuthoritativeTokens = checkedAdd(
      this.targetAuthoritativeTokens,
      sumDecode(work, (entry) => entry.targetAuthoritativeTokens),
      "target authoritative tokens",
    );
    const transientKvTokens = checkedAdd(
      this.kvTokens,
      batchKvReservation(work),
      `batch ${work.batchId} transient KV`,
    );
    this.updateKvHighWater(transientKvTokens);
    this.emit({
      kind: "batch_start",
      atNs,
      batch: work,
      durationNs,
      kvTokensBefore: this.kvTokens,
      transientKvTokens,
    });
    this.eventLoop.scheduleAt(finishNs, {
      kind: "batch_finish",
      batchId: work.batchId,
    });
  }

  private handleBatchFinish(batchId: number, atNs: number): void {
    const running = this.running;
    if (
      !running
      || running.work.batchId !== batchId
      || atNs !== running.startedAtNs + running.durationNs
    ) {
      throw new ServingProtocolError(`invalid finish for batch ${batchId}`);
    }
    const emittedTokens: ServingTokenEmission[] = [];
    const completedRequestIds: string[] = [];

    for (const entry of running.work.prefill) {
      const request = requireRequest(this.requests, entry.requestId);
      request.promptProcessed += entry.tokens;
      request.kvTokens += entry.tokens;
      this.kvTokens += entry.tokens;
      this.updateKvHighWater();
      if (request.promptProcessed === request.spec.promptTokens) {
        request.phase = "decoding";
        this.emitToken(request, "prefill", atNs, emittedTokens);
        this.completeIfDone(request, atNs, completedRequestIds);
      } else {
        request.phase = "waiting";
      }
    }
    for (const decode of running.work.decode) {
      const request = requireRequest(this.requests, decode.requestId);
      applyDecodeTransaction(request, decode);
      request.kvTokens += decode.committedTokens;
      this.kvTokens += decode.committedTokens;
      this.updateKvHighWater();
      for (
        let tokenOffset = 0;
        tokenOffset < decode.committedTokens;
        tokenOffset++
      ) {
        const source = decodeEmissionSource(decode, tokenOffset);
        this.emitToken(request, source, atNs, emittedTokens);
      }
      this.completeIfDone(request, atNs, completedRequestIds);
    }

    this.emit({
      kind: "batch_finish",
      atNs,
      batchId,
      emittedTokens,
      completedRequestIds,
      kvTokensAfter: this.kvTokens,
    });
    this.running = undefined;
    if (
      this.completedRequests === this.config.requests.length
      && this.arrivedRequests === this.config.requests.length
    ) {
      if (this.kvTokens !== 0) {
        throw new ServingProtocolError(
          `terminal serving state retains ${this.kvTokens} KV tokens`,
        );
      }
      this.emit({
        kind: "terminal",
        atNs,
        completedRequests: this.completedRequests,
        totalOutputTokens: this.config.requests.reduce(
          (sum, request) => sum + request.outputTokens,
          0,
        ),
        kvTokensAfter: 0,
      });
      return;
    }
    this.maybeStartBatch(atNs);
  }

  private emitToken(
    request: MutableRequest,
    source: ServingTokenEmission["source"],
    atNs: number,
    output: ServingTokenEmission[],
  ): void {
    if (request.outputEmitted >= request.spec.outputTokens) {
      throw new ServingProtocolError(
        `request ${request.spec.id} emitted beyond output budget`,
      );
    }
    const tokenIndex = request.outputEmitted++;
    request.tokenTimestampsNs.push(atNs);
    output.push({ requestId: request.spec.id, tokenIndex, source });
  }

  private completeIfDone(
    request: MutableRequest,
    atNs: number,
    completed: string[],
  ): void {
    if (request.outputEmitted !== request.spec.outputTokens) {
      return;
    }
    request.phase = "completed";
    request.completedAtNs = atNs;
    this.kvTokens -= request.kvTokens;
    request.kvTokens = 0;
    this.completedRequests++;
    completed.push(request.spec.id);
  }

  private updateKvHighWater(candidateKvTokens = this.kvTokens): void {
    if (candidateKvTokens > this.config.maxKvTokens) {
      throw new ServingProtocolError(
        `KV usage ${candidateKvTokens} exceeds capacity ${this.config.maxKvTokens}`,
      );
    }
    this.kvHighWaterTokens = Math.max(
      this.kvHighWaterTokens,
      candidateKvTokens,
    );
  }

  private requestResults(): readonly ServingRequestResult[] {
    return sortedRequests(this.config.requests).map((spec) => {
      const state = requireRequest(this.requests, spec.id);
      const firstTokenNs = state.tokenTimestampsNs[0];
      if (
        firstTokenNs === undefined
        || state.completedAtNs === undefined
        || state.tokenTimestampsNs.length !== spec.outputTokens
      ) {
        throw new ServingProtocolError(
          `request ${spec.id} lacks terminal token timing`,
        );
      }
      return {
        id: spec.id,
        arrivalNs: spec.arrivalNs,
        firstTokenNs,
        completedAtNs: state.completedAtNs,
        timeToFirstTokenNs: firstTokenNs - spec.arrivalNs,
        latencyNs: state.completedAtNs - spec.arrivalNs,
        tokenTimestampsNs: state.tokenTimestampsNs,
      };
    });
  }

  private metrics(
    requests: readonly ServingRequestResult[],
  ): ServingMetrics {
    const firstArrival = Math.min(...requests.map((request) => request.arrivalNs));
    const terminal = Math.max(...requests.map((request) => request.completedAtNs));
    const totalDurationNs = terminal - firstArrival;
    const outputTokens = this.config.requests.reduce(
      (sum, request) => sum + request.outputTokens,
      0,
    );
    const prefillTokens = this.config.requests.reduce(
      (sum, request) => sum + request.promptTokens,
      0,
    );
    const decodeTokens = this.config.requests.reduce(
      (sum, request) => sum + Math.max(0, request.outputTokens - 1),
      0,
    );
    const ttft = requests.map((request) => request.timeToFirstTokenNs);
    const latency = requests.map((request) => request.latencyNs);
    const itl = requests.flatMap((request) => (
      request.tokenTimestampsNs.slice(1).map((timestamp, index) => (
        timestamp - request.tokenTimestampsNs[index]
      ))
    ));
    const batches = this.nextBatchId;
    return {
      requests: requests.length,
      batches,
      prefillTokens,
      decodeTokens,
      outputTokens,
      targetForwards: this.targetForwards,
      targetVerificationTokens: this.targetVerificationTokens,
      guaranteedTargetTokens: this.guaranteedTargetTokens,
      proposedAdditionalTokens: this.proposedAdditionalTokens,
      acceptedAdditionalTokens: this.acceptedAdditionalTokens,
      proposedDraftTokens: this.proposedDraftTokens,
      acceptedDraftTokens: this.acceptedDraftTokens,
      rejectedDraftTokens:
        this.proposedDraftTokens - this.acceptedDraftTokens,
      targetAuthoritativeTokens: this.targetAuthoritativeTokens,
      committedTokensPerTargetForward:
        this.targetForwards === 0 ? 0 : decodeTokens / this.targetForwards,
      totalDurationNs,
      batchServiceNs: this.batchServiceNs,
      throughputTokensPerSecond: totalDurationNs === 0
        ? 0
        : outputTokens * 1_000_000_000 / totalDurationNs,
      averageTimeToFirstTokenNs: average(ttft),
      p50TimeToFirstTokenNs: percentile(ttft, 0.5),
      p95TimeToFirstTokenNs: percentile(ttft, 0.95),
      averageInterTokenLatencyNs: average(itl),
      p50InterTokenLatencyNs: percentile(itl, 0.5),
      p95InterTokenLatencyNs: percentile(itl, 0.95),
      averageRequestLatencyNs: average(latency),
      sequenceBatchUtilization:
        this.scheduledSequenceWork / (batches * this.config.maxBatchSize),
      tokenBatchUtilization:
        this.scheduledTokenWork / (batches * this.config.maxBatchTokens),
      kvHighWaterTokens: this.kvHighWaterTokens,
    };
  }

  private emit(
    event: ServingTraceInput,
  ): void {
    this.traceEvents.push({
      contractRevision: SERVING_TRACE_CONTRACT_REVISION,
      sourceSequence: this.sourceSequence++,
      ...event,
    } as ServingTraceEvent);
  }
}

export function replayServingTrace(
  config: ServingSchedulerConfig,
  trace: readonly ServingTraceEvent[],
  estimateDuration: ServingBatchDurationEstimator =
    DEFAULT_SERVING_BATCH_DURATION,
): ServingReplayResult {
  validateServingConfig(config);
  const requests = new Map<string, MutableRequest>();
  for (const spec of sortedRequests(config.requests)) {
    requests.set(spec.id, createMutableRequest(config, spec));
  }
  let running: RunningBatch | undefined;
  let kvTokens = 0;
  let completedRequests = 0;
  let nextBatchId = 0;
  let previousAtNs = 0;
  let terminalSeen = false;

  for (let index = 0; index < trace.length; index++) {
    const event = trace[index];
    if (
      event.contractRevision !== SERVING_TRACE_CONTRACT_REVISION
      || event.sourceSequence !== index
      || event.atNs < previousAtNs
      || terminalSeen
    ) {
      replayFail(`invalid trace envelope at event ${index}`);
    }
    previousAtNs = event.atNs;
    if (event.kind === "arrival") {
      const request = requireRequest(requests, event.requestId);
      if (
        request.phase !== "unarrived"
        || request.spec.arrivalNs !== event.atNs
      ) {
        replayFail(`invalid arrival for ${event.requestId}`);
      }
      request.phase = "waiting";
      continue;
    }
    if (event.kind === "batch_start") {
      if (running) {
        replayFail(`batch ${event.batch.batchId} overlaps active batch`);
      }
      const expected = selectServingBatch(
        config,
        requests,
        kvTokens,
        nextBatchId,
      );
      if (!expected || !equalBatch(expected, event.batch)) {
        replayFail(`batch ${event.batch.batchId} violates scheduler decision`);
      }
      const durationNs = estimateDuration(expected, event.atNs);
      const transientKvTokens = checkedAdd(
        kvTokens,
        batchKvReservation(expected),
        `batch ${expected.batchId} transient KV`,
      );
      if (
        event.durationNs !== durationNs
        || event.kvTokensBefore !== kvTokens
        || event.transientKvTokens !== transientKvTokens
        || transientKvTokens > config.maxKvTokens
      ) {
        replayFail(`batch ${event.batch.batchId} timing/KV mismatch`);
      }
      for (const entry of expected.prefill) {
        requireRequest(requests, entry.requestId).phase = "prefilling";
      }
      running = {
        work: expected,
        startedAtNs: event.atNs,
        durationNs,
      };
      nextBatchId++;
      continue;
    }
    if (event.kind === "batch_finish") {
      if (
        !running
        || running.work.batchId !== event.batchId
        || event.atNs !== running.startedAtNs + running.durationNs
      ) {
        replayFail(`invalid finish for batch ${event.batchId}`);
      }
      const emittedTokens: ServingTokenEmission[] = [];
      const completedRequestIds: string[] = [];
      for (const entry of running.work.prefill) {
        const request = requireRequest(requests, entry.requestId);
        request.promptProcessed += entry.tokens;
        request.kvTokens += entry.tokens;
        kvTokens += entry.tokens;
        if (request.promptProcessed === request.spec.promptTokens) {
          request.phase = "decoding";
          replayEmitToken(request, "prefill", event.atNs, emittedTokens);
          if (request.outputEmitted === request.spec.outputTokens) {
            request.phase = "completed";
            request.completedAtNs = event.atNs;
            kvTokens -= request.kvTokens;
            request.kvTokens = 0;
            completedRequests++;
            completedRequestIds.push(request.spec.id);
          }
        } else {
          request.phase = "waiting";
        }
      }
      for (const decode of running.work.decode) {
        const request = requireRequest(requests, decode.requestId);
        applyDecodeTransaction(request, decode, true);
        request.kvTokens += decode.committedTokens;
        kvTokens += decode.committedTokens;
        for (
          let tokenOffset = 0;
          tokenOffset < decode.committedTokens;
          tokenOffset++
        ) {
          const source = decodeEmissionSource(decode, tokenOffset);
          replayEmitToken(request, source, event.atNs, emittedTokens);
        }
        if (request.outputEmitted === request.spec.outputTokens) {
          request.phase = "completed";
          request.completedAtNs = event.atNs;
          kvTokens -= request.kvTokens;
          request.kvTokens = 0;
          completedRequests++;
          completedRequestIds.push(request.spec.id);
        }
      }
      if (
        kvTokens > config.maxKvTokens
        || event.kvTokensAfter !== kvTokens
        || !equalEmissions(event.emittedTokens, emittedTokens)
        || !equalStrings(event.completedRequestIds, completedRequestIds)
      ) {
        replayFail(`batch ${event.batchId} completion mismatch`);
      }
      running = undefined;
      continue;
    }
    if (
      running
      || completedRequests !== config.requests.length
      || event.completedRequests !== completedRequests
      || event.totalOutputTokens !== config.requests.reduce(
        (sum, request) => sum + request.outputTokens,
        0,
      )
      || event.kvTokensAfter !== 0
      || kvTokens !== 0
    ) {
      replayFail("invalid serving terminal event");
    }
    terminalSeen = true;
  }
  if (!terminalSeen) {
    replayFail("serving trace lacks terminal event");
  }
  return {
    appliedEvents: trace.length,
    completedRequests,
    finalKvTokens: kvTokens,
  };
}

function selectServingBatch(
  config: ServingSchedulerConfig,
  requests: ReadonlyMap<string, MutableRequest>,
  currentKvTokens: number,
  batchId: number,
): ServingBatchWork | undefined {
  let tokenBudget = config.maxBatchTokens;
  let sequenceBudget = config.maxBatchSize;
  let reservedKv = 0;
  const decode: ServingDecodeSlice[] = [];
  const prefill: ServingPrefillSlice[] = [];
  const candidates = [...requests.values()].sort(compareMutableRequests);

  for (const request of candidates) {
    if (
      request.phase !== "decoding"
      || sequenceBudget === 0
      || tokenBudget === 0
    ) {
      continue;
    }
    const remainingOutput =
      request.spec.outputTokens - request.outputEmitted;
    const availableKv =
      config.maxKvTokens - currentKvTokens - reservedKv;
    const maxTargetTokenWidth = Math.min(tokenBudget, availableKv);
    if (remainingOutput <= 0 || maxTargetTokenWidth <= 0) {
      continue;
    }
    const proposal = planSpeculativeProposal({
      proposalPrefix: request.speculative?.proposalPrefix ?? "none",
      remainingOutputTokens: remainingOutput,
      maxAdditionalTokens: config.speculative?.maxAdditionalTokens ?? 0,
      maxTargetTokenWidth,
    });
    const acceptedAdditionalTokens = request.speculative?.acceptance.next(
      proposal.proposedAdditionalTokens,
      request.outputEmitted,
    ) ?? 0;
    const decision = decideSpeculativeIteration(
      proposal,
      acceptedAdditionalTokens,
      remainingOutput,
    );
    decode.push({
      requestId: request.spec.id,
      mode: config.speculative ? "speculative" : "target_only",
      ...decision,
    });
    reservedKv += decision.targetTokenWidth;
    sequenceBudget--;
    tokenBudget -= decision.targetTokenWidth;
  }
  for (const request of candidates) {
    if (
      (request.phase !== "waiting" && request.phase !== "prefilling")
      || sequenceBudget === 0
      || tokenBudget === 0
    ) {
      continue;
    }
    const remainingPrompt = request.spec.promptTokens - request.promptProcessed;
    const availableKv = config.maxKvTokens - currentKvTokens - reservedKv;
    const tokens = Math.min(
      remainingPrompt,
      config.prefillChunkTokens,
      tokenBudget,
      availableKv,
    );
    if (tokens <= 0) {
      continue;
    }
    prefill.push({ requestId: request.spec.id, tokens });
    reservedKv += tokens;
    sequenceBudget--;
    tokenBudget -= tokens;
  }
  const tokenWork = config.maxBatchTokens - tokenBudget;
  if (tokenWork === 0) {
    return undefined;
  }
  return {
    batchId,
    prefill,
    decode,
    tokenWork,
    sequenceCount: prefill.length + decode.length,
    expectedOutputTokens: decode.reduce(
      (sum, entry) => sum + entry.committedTokens,
      0,
    ) + prefill.filter((entry) => {
      const request = requireRequest(requests, entry.requestId);
      return request.promptProcessed + entry.tokens === request.spec.promptTokens;
    }).length,
  };
}

function validateServingConfig(config: ServingSchedulerConfig): void {
  if (config.requests.length === 0) {
    throw new ServingProtocolError("serving workload requires requests");
  }
  assertPositiveSafeInteger(config.maxBatchSize, "maxBatchSize");
  assertPositiveSafeInteger(config.maxBatchTokens, "maxBatchTokens");
  assertPositiveSafeInteger(config.prefillChunkTokens, "prefillChunkTokens");
  assertPositiveSafeInteger(config.maxKvTokens, "maxKvTokens");
  if (config.speculative) {
    assertNonNegativeSafeInteger(
      config.speculative.maxAdditionalTokens,
      "speculative.maxAdditionalTokens",
    );
    validateSpeculativeEligibility(
      config.speculative.family,
      config.speculative.eligibility,
    );
    if (config.speculative.acceptance.kind === "replay") {
      const requestIds = new Set(config.requests.map((request) => request.id));
      for (const request of config.requests) {
        if (
          config.speculative.acceptance
            .acceptedDraftTokensByRequest[request.id] === undefined
        ) {
          throw new ServingProtocolError(
            `speculative acceptance replay lacks request ${request.id}`,
          );
        }
      }
      for (
        const requestId
        of Object.keys(
          config.speculative.acceptance.acceptedDraftTokensByRequest,
        )
      ) {
        if (!requestIds.has(requestId)) {
          throw new ServingProtocolError(
            `speculative acceptance replay references unknown request ${requestId}`,
          );
        }
      }
    } else {
      validateAcceptanceCoverage(
        config.speculative.acceptance,
        config.speculative.maxAdditionalTokens,
      );
    }
  }
  if (config.maxEvents !== undefined) {
    assertPositiveSafeInteger(config.maxEvents, "maxEvents");
  }
  const ids = new Set<string>();
  for (const request of config.requests) {
    if (request.id.length === 0 || ids.has(request.id)) {
      throw new ServingProtocolError(
        `request id must be non-empty and unique; got ${request.id}`,
      );
    }
    ids.add(request.id);
    assertNonNegativeSafeInteger(request.arrivalNs, `${request.id} arrivalNs`);
    assertPositiveSafeInteger(request.promptTokens, `${request.id} promptTokens`);
    assertPositiveSafeInteger(request.outputTokens, `${request.id} outputTokens`);
    if (
      request.priority !== undefined
      && !Number.isSafeInteger(request.priority)
    ) {
      throw new ServingProtocolError(
        `${request.id} priority must be a safe integer`,
      );
    }
    const peakKv = checkedAdd(
      request.promptTokens,
      request.outputTokens - 1,
      `${request.id} peak KV`,
    );
    if (peakKv > config.maxKvTokens) {
      throw new ServingProtocolError(
        `${request.id} requires ${peakKv} KV tokens but capacity is ${config.maxKvTokens}`,
      );
    }
  }
}

function sortedRequests(
  requests: readonly ServingRequestSpec[],
): readonly ServingRequestSpec[] {
  return [...requests].sort((left, right) => (
    left.arrivalNs - right.arrivalNs
    || (right.priority ?? 0) - (left.priority ?? 0)
    || left.id.localeCompare(right.id)
  ));
}

function createMutableRequest(
  config: ServingSchedulerConfig,
  spec: ServingRequestSpec,
): MutableRequest {
  const speculative = config.speculative;
  const peakKv = checkedAdd(
    spec.promptTokens,
    spec.outputTokens - 1,
    `${spec.id} peak KV`,
  );
  return {
    spec,
    phase: "unarrived",
    promptProcessed: 0,
    outputEmitted: 0,
    kvTokens: 0,
    tokenTimestampsNs: [],
    ...(speculative
      ? {
          speculative: {
            transaction: new SpeculativeTransactionSimulator(
              spec.promptTokens,
              buildSpeculativeStateGroups(
                speculative.family,
                peakKv,
                speculative.maxAdditionalTokens,
              ),
            ),
            acceptance: new SpeculativeAcceptanceCursor(
              acceptanceForRequest(speculative.acceptance, spec.id),
              speculative.acceptance.kind === "replay" ? undefined : spec.id,
            ),
            proposalPrefix:
              speculativeFamilyContract(speculative.family).proposalPrefix,
          },
        }
      : {}),
  };
}

function acceptanceForRequest(
  acceptance: ServingSpeculativeAcceptanceModel,
  requestId: string,
): SpeculativeAcceptanceModel {
  if (acceptance.kind !== "replay") {
    return acceptance;
  }
  return {
    kind: "replay",
    acceptedDraftTokens:
      acceptance.acceptedDraftTokensByRequest[requestId] ?? [],
  };
}

function applyDecodeTransaction(
  request: MutableRequest,
  decode: ServingDecodeSlice,
  replay = false,
): void {
  if (decode.mode === "target_only") {
    if (
      request.speculative
      || decode.guaranteedTargetTokens !== 0
      || decode.proposedAdditionalTokens !== 0
      || decode.acceptedAdditionalTokens !== 0
      || decode.proposedDraftTokens !== 0
      || decode.acceptedDraftTokens !== 0
      || decode.targetAuthoritativeTokens !== 1
      || decode.committedTokens !== 1
      || decode.targetTokenWidth !== 1
      || decode.outcome !== "target_only"
    ) {
      decodeFail(replay, `invalid target-only decode for ${request.spec.id}`);
    }
    return;
  }
  const speculative = request.speculative;
  if (!speculative) {
    decodeFail(replay, `request ${request.spec.id} lacks speculative state`);
  }
  const expectedGuaranteedTargetTokens: 0 | 1 =
    speculative.proposalPrefix === "guaranteed_target"
    && decode.proposedDraftTokens > 0
      ? 1
      : 0;
  const result = speculative.transaction.runIteration({
    draftTokenCount: decode.proposedDraftTokens,
    acceptedDraftTokenCount: decode.acceptedDraftTokens,
    proposalLocalTokenCount: decode.proposedAdditionalTokens,
    targetAuthoritativeTokenCount: decode.targetAuthoritativeTokens,
  });
  if (
    decode.guaranteedTargetTokens !== expectedGuaranteedTargetTokens
    || decode.proposedDraftTokens
      !== decode.guaranteedTargetTokens + decode.proposedAdditionalTokens
    || decode.acceptedDraftTokens
      !== decode.guaranteedTargetTokens + decode.acceptedAdditionalTokens
    || result.committedTokenCount !== decode.committedTokens
    || decode.targetTokenWidth !== decode.proposedDraftTokens + 1
    || result.finalTokenLength !== request.kvTokens + decode.committedTokens
  ) {
    decodeFail(replay, `speculative transaction mismatch for ${request.spec.id}`);
  }
}

function decodeEmissionSource(
  decode: ServingDecodeSlice,
  tokenOffset: number,
): ServingTokenEmission["source"] {
  if (decode.mode === "target_only" || decode.outcome === "target_only") {
    return "target_decode";
  }
  if (tokenOffset < decode.guaranteedTargetTokens) {
    return "target_guaranteed";
  }
  if (tokenOffset < decode.acceptedDraftTokens) {
    return "speculative_accepted";
  }
  return "speculative_authoritative";
}

function decodeFail(replay: boolean, message: string): never {
  if (replay) {
    replayFail(message);
  }
  throw new ServingProtocolError(message);
}

function batchKvReservation(batch: ServingBatchWork): number {
  return batch.prefill.reduce((sum, entry) => sum + entry.tokens, 0)
    + sumDecode(batch, (entry) => entry.targetTokenWidth);
}

function sumDecode(
  batch: ServingBatchWork,
  select: (entry: ServingDecodeSlice) => number,
): number {
  return batch.decode.reduce((sum, entry) => sum + select(entry), 0);
}

function compareMutableRequests(
  left: MutableRequest,
  right: MutableRequest,
): number {
  return (
    (right.spec.priority ?? 0) - (left.spec.priority ?? 0)
    || left.spec.arrivalNs - right.spec.arrivalNs
    || left.spec.id.localeCompare(right.spec.id)
  );
}

function replayEmitToken(
  request: MutableRequest,
  source: ServingTokenEmission["source"],
  atNs: number,
  output: ServingTokenEmission[],
): void {
  if (request.outputEmitted >= request.spec.outputTokens) {
    replayFail(`request ${request.spec.id} exceeds output budget`);
  }
  const tokenIndex = request.outputEmitted++;
  request.tokenTimestampsNs.push(atNs);
  output.push({ requestId: request.spec.id, tokenIndex, source });
}

function requireRequest(
  requests: ReadonlyMap<string, MutableRequest>,
  id: string,
): MutableRequest {
  const request = requests.get(id);
  if (!request) {
    throw new ServingProtocolError(`unknown serving request ${id}`);
  }
  return request;
}

function equalBatch(left: ServingBatchWork, right: ServingBatchWork): boolean {
  return left.batchId === right.batchId
    && left.tokenWork === right.tokenWork
    && left.sequenceCount === right.sequenceCount
    && left.expectedOutputTokens === right.expectedOutputTokens
    && left.decode.length === right.decode.length
    && left.decode.every((entry, index) => (
      equalDecode(entry, right.decode[index])
    ))
    && left.prefill.length === right.prefill.length
    && left.prefill.every((entry, index) => (
      entry.requestId === right.prefill[index]?.requestId
      && entry.tokens === right.prefill[index]?.tokens
    ));
}

function equalDecode(
  left: ServingDecodeSlice,
  right: ServingDecodeSlice | undefined,
): boolean {
  return right !== undefined
    && left.requestId === right.requestId
    && left.mode === right.mode
    && left.guaranteedTargetTokens === right.guaranteedTargetTokens
    && left.proposedAdditionalTokens === right.proposedAdditionalTokens
    && left.acceptedAdditionalTokens === right.acceptedAdditionalTokens
    && left.proposedDraftTokens === right.proposedDraftTokens
    && left.acceptedDraftTokens === right.acceptedDraftTokens
    && left.targetAuthoritativeTokens === right.targetAuthoritativeTokens
    && left.committedTokens === right.committedTokens
    && left.targetTokenWidth === right.targetTokenWidth
    && left.outcome === right.outcome;
}

function equalEmissions(
  left: readonly ServingTokenEmission[],
  right: readonly ServingTokenEmission[],
): boolean {
  return left.length === right.length && left.every((entry, index) => (
    entry.requestId === right[index]?.requestId
    && entry.tokenIndex === right[index]?.tokenIndex
    && entry.source === right[index]?.source
  ));
}

function equalStrings(
  left: readonly string[],
  right: readonly string[],
): boolean {
  return left.length === right.length
    && left.every((entry, index) => entry === right[index]);
}

function average(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values: readonly number[], quantile: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(quantile * sorted.length) - 1)];
}

function replayFail(message: string): never {
  throw new ServingProtocolError(`serving replay: ${message}`);
}

function checkedAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    throw new ServingProtocolError(`${label} exceeds Number.MAX_SAFE_INTEGER`);
  }
  return result;
}

function checkedMultiply(left: number, right: number, label: string): number {
  const result = left * right;
  if (!Number.isSafeInteger(result)) {
    throw new ServingProtocolError(`${label} exceeds Number.MAX_SAFE_INTEGER`);
  }
  return result;
}

function assertNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new ServingProtocolError(
      `${label} must be a non-negative safe integer; got ${value}`,
    );
  }
}

function assertPositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new ServingProtocolError(
      `${label} must be a positive safe integer; got ${value}`,
    );
  }
}
