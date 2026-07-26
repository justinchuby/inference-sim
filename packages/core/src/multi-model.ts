/**
 * Co-residency: several models served from one device.
 *
 * On a personal machine the binding constraint for running more than one model
 * is residency rather than compute. A device holds a fixed number of bytes,
 * each model occupies its weights plus a preallocated KV arena, and a model
 * that is not resident cannot answer until it has been loaded over a link that
 * is itself contended.
 *
 * That produces two regimes. When the working set fits, every model stays
 * resident and the only interference is that one model's batch delays
 * another's. When it does not fit, models are evicted and reloaded, and an
 * eviction releases the KV arena, so partially generated requests lose their
 * context and prefill again. The simulator reports which regime a
 * configuration is in and what it costs, rather than assuming either.
 *
 * Residency, eviction, admission, and token accounting are exact. Batch
 * durations come from the caller's cost model and carry its confidence.
 */
import { DiscreteEventSimulator } from "./event-loop.js";
import { compareIds } from "./ordering.js";

export const MULTI_MODEL_TRACE_REVISION = 1;

export class MultiModelProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MultiModelProtocolError";
  }
}

// ============================================================
// Configuration
// ============================================================

export interface MultiModelRequestSpec {
  readonly id: string;
  readonly arrivalNs: number;
  readonly promptTokens: number;
  readonly outputTokens: number;
}

export interface MultiModelTenantSpec {
  readonly id: string;
  readonly displayName: string;
  readonly weightBytes: number;
  readonly kvBytesPerToken: number;
  /** KV arena the runtime preallocates when the model becomes resident. */
  readonly maxKvTokens: number;
  /** Pinned models are never evicted, so they never pay a reload. */
  readonly pinned: boolean;
  readonly requests: readonly MultiModelRequestSpec[];
}

export interface MultiModelConfig {
  readonly tenants: readonly MultiModelTenantSpec[];
  /** Allocatable device bytes shared by every model. */
  readonly deviceMemoryBytes: number;
  /** Effective bandwidth of the link a model is loaded over. */
  readonly loadBandwidthBytesPerSec: number;
  readonly maxBatchTokens: number;
  readonly prefillChunkTokens: number;
}

export interface MultiModelPrefillSlice {
  readonly requestId: string;
  readonly tokens: number;
}

/**
 * Work one model performs in a single batch. Slices are per request so replay
 * can rebuild KV occupancy exactly rather than inferring it from totals.
 */
export interface MultiModelBatchWork {
  readonly batchId: number;
  readonly tenantId: string;
  readonly prefill: readonly MultiModelPrefillSlice[];
  readonly decode: readonly string[];
  readonly prefillTokens: number;
  readonly decodeTokens: number;
  readonly sequenceCount: number;
  readonly tokenWork: number;
}

export type MultiModelBatchDurationEstimator = (
  batch: MultiModelBatchWork,
  tenant: MultiModelTenantSpec,
) => number;

// ============================================================
// Trace
// ============================================================

export type MultiModelTraceEvent =
  | {
      readonly kind: "arrival";
      readonly atNs: number;
      readonly tenantId: string;
      readonly requestId: string;
    }
  | {
      readonly kind: "evict";
      readonly atNs: number;
      readonly tenantId: string;
      readonly bytes: number;
      /** Committed tokens discarded with the KV arena. */
      readonly discardedKvTokens: number;
    }
  | {
      readonly kind: "load_start";
      readonly atNs: number;
      readonly tenantId: string;
      readonly bytes: number;
      readonly completesAtNs: number;
    }
  | {
      readonly kind: "load_complete";
      readonly atNs: number;
      readonly tenantId: string;
      readonly residentBytes: number;
    }
  | {
      readonly kind: "batch_start";
      readonly atNs: number;
      readonly batch: MultiModelBatchWork;
      readonly durationNs: number;
    }
  | {
      readonly kind: "batch_finish";
      readonly atNs: number;
      readonly batchId: number;
      readonly tenantId: string;
      readonly emittedRequestIds: readonly string[];
      readonly completedRequestIds: readonly string[];
    }
  | {
      readonly kind: "terminal";
      readonly atNs: number;
      readonly completedRequests: number;
      readonly residentBytes: number;
    };

// ============================================================
// Results
// ============================================================

export interface MultiModelTenantMetrics {
  readonly tenantId: string;
  readonly displayName: string;
  readonly requests: number;
  readonly outputTokens: number;
  readonly residencyFootprintBytes: number;
  readonly loads: number;
  readonly evictions: number;
  readonly loadedBytes: number;
  readonly loadServiceNs: number;
  /** Time this model's requests spent waiting for it to become resident. */
  readonly residencyWaitNs: number;
  readonly averageTimeToFirstTokenNs: number;
  readonly p95TimeToFirstTokenNs: number;
  readonly averageRequestLatencyNs: number;
  /** Prompt tokens recomputed because an eviction discarded partial work. */
  readonly rePrefilledTokens: number;
  readonly residentNs: number;
}

export interface MultiModelMetrics {
  readonly tenants: readonly MultiModelTenantMetrics[];
  readonly totalDurationNs: number;
  readonly computeServiceNs: number;
  readonly transferServiceNs: number;
  readonly computeUtilization: number;
  readonly transferUtilization: number;
  /** Transfer time that overlapped a batch and therefore cost no wall clock. */
  readonly hiddenTransferNs: number;
  readonly peakResidentBytes: number;
  readonly deviceMemoryBytes: number;
  readonly totalLoads: number;
  readonly totalEvictions: number;
  readonly totalLoadedBytes: number;
  /**
   * Loads beyond the unavoidable first load of each model, per request. Zero
   * means the working set stayed resident; approaching one means the device
   * reloads a model for nearly every request.
   */
  readonly reloadsPerRequest: number;
  /** True when nothing ever had to be evicted to make room. */
  readonly fitsWithoutSwapping: boolean;
}

export interface MultiModelRequestResult {
  readonly tenantId: string;
  readonly requestId: string;
  readonly arrivalNs: number;
  readonly firstTokenNs: number;
  readonly completedAtNs: number;
  readonly timeToFirstTokenNs: number;
  readonly residencyWaitNs: number;
}

export interface MultiModelReplayResult {
  readonly appliedEvents: number;
  readonly completedRequests: number;
  readonly finalResidentBytes: number;
}

export interface MultiModelResult {
  readonly revision: typeof MULTI_MODEL_TRACE_REVISION;
  readonly trace: readonly MultiModelTraceEvent[];
  readonly requests: readonly MultiModelRequestResult[];
  readonly metrics: MultiModelMetrics;
  readonly replay: MultiModelReplayResult;
}

// ============================================================
// Mutable state
// ============================================================

type RequestPhase = "unarrived" | "waiting" | "decoding" | "completed";

interface MutableRequest {
  readonly spec: MultiModelRequestSpec;
  readonly tenantId: string;
  phase: RequestPhase;
  promptProcessed: number;
  outputEmitted: number;
  kvTokens: number;
  firstTokenNs?: number;
  completedAtNs?: number;
  residencyWaitNs: number;
  /** Set while the model is not resident, to attribute the wait. */
  blockedSinceNs?: number;
}

interface MutableTenant {
  readonly spec: MultiModelTenantSpec;
  readonly footprintBytes: number;
  resident: boolean;
  loading: boolean;
  /** Requests retired since the current residency began. */
  completedSinceLoad: number;
  lastUsedSequence: number;
  residentSinceNs?: number;
  residentNs: number;
  loads: number;
  evictions: number;
  loadedBytes: number;
  loadServiceNs: number;
  rePrefilledTokens: number;
  kvTokens: number;
}

type InternalEvent =
  | { readonly kind: "arrival"; readonly requestKey: string }
  | { readonly kind: "load_finish"; readonly tenantId: string }
  | { readonly kind: "batch_finish"; readonly batchId: number };

/** Bytes a model occupies while resident: weights plus its whole KV arena. */
export function tenantFootprintBytes(spec: MultiModelTenantSpec): number {
  return spec.weightBytes + spec.kvBytesPerToken * spec.maxKvTokens;
}

// ============================================================
// Simulator
// ============================================================

class MultiModelSimulator {
  private readonly eventLoop = new DiscreteEventSimulator<InternalEvent>();
  private readonly tenants = new Map<string, MutableTenant>();
  private readonly requests = new Map<string, MutableRequest>();
  private readonly requestsByTenant = new Map<string, MutableRequest[]>();
  private readonly trace: MultiModelTraceEvent[] = [];

  private residentBytes = 0;
  private peakResidentBytes = 0;
  private useClock = 0;
  private nextBatchId = 0;
  private completedRequests = 0;
  private computeServiceNs = 0;
  private transferServiceNs = 0;
  private hiddenTransferNs = 0;

  private runningBatch?: {
    readonly work: MultiModelBatchWork;
    readonly startedAtNs: number;
    readonly durationNs: number;
  };

  private runningLoad?: {
    readonly tenantId: string;
    readonly completesAtNs: number;
  };

  constructor(
    private readonly config: MultiModelConfig,
    private readonly estimateDuration: MultiModelBatchDurationEstimator,
  ) {
    for (const spec of config.tenants) {
      this.tenants.set(spec.id, {
        spec,
        footprintBytes: tenantFootprintBytes(spec),
        resident: false,
        loading: false,
        completedSinceLoad: 0,
        lastUsedSequence: 0,
        residentNs: 0,
        loads: 0,
        evictions: 0,
        loadedBytes: 0,
        loadServiceNs: 0,
        rePrefilledTokens: 0,
        kvTokens: 0,
      });
      const owned: MutableRequest[] = [];
      for (const request of spec.requests) {
        const mutable: MutableRequest = {
          spec: request,
          tenantId: spec.id,
          phase: "unarrived",
          promptProcessed: 0,
          outputEmitted: 0,
          kvTokens: 0,
          residencyWaitNs: 0,
        };
        this.requests.set(requestKey(spec.id, request.id), mutable);
        owned.push(mutable);
      }
      owned.sort((left, right) => (
        left.spec.arrivalNs - right.spec.arrivalNs
        || compareIds(left.spec.id, right.spec.id)
      ));
      this.requestsByTenant.set(spec.id, owned);
    }
  }

  run(): MultiModelResult {
    for (const key of this.requests.keys()) {
      const request = this.requests.get(key)!;
      this.eventLoop.scheduleAt(request.spec.arrivalNs, {
        kind: "arrival",
        requestKey: key,
      });
    }
    const totalRequests = this.requests.size;
    this.eventLoop.run((event, simulation) => {
      this.handleEvent(event.payload, simulation.nowNs);
    }, { maxEvents: 4_000_000 });

    if (this.completedRequests !== totalRequests) {
      throw new MultiModelProtocolError(
        `co-residency quiesced with ${this.completedRequests}/${totalRequests} requests complete`,
      );
    }
    const totalDurationNs = this.eventLoop.nowNs;
    this.emit({
      kind: "terminal",
      atNs: totalDurationNs,
      completedRequests: this.completedRequests,
      residentBytes: this.residentBytes,
    });
    return {
      revision: MULTI_MODEL_TRACE_REVISION,
      trace: this.trace,
      requests: this.requestResults(),
      metrics: this.buildMetrics(totalDurationNs),
      replay: replayMultiModelTrace(
        this.config,
        this.trace,
        this.estimateDuration,
      ),
    };
  }

  private handleEvent(event: InternalEvent, atNs: number): void {
    switch (event.kind) {
      case "arrival":
        this.handleArrival(event.requestKey, atNs);
        break;
      case "load_finish":
        this.handleLoadFinish(event.tenantId, atNs);
        break;
      case "batch_finish":
        this.handleBatchFinish(event.batchId, atNs);
        break;
    }
    // Both lanes are offered work on every state change, and transfers overlap
    // compute so one model can load while another computes. The transfer lane
    // is offered first: a model that is mid-batch cannot be evicted, so
    // starting a batch before considering a load would let whichever model is
    // already resident hold the device for as long as it has work.
    this.startLoadIfPossible(atNs);
    this.startBatchIfPossible(atNs);
  }

  private handleArrival(key: string, atNs: number): void {
    const request = this.requests.get(key);
    if (!request || request.phase !== "unarrived") {
      throw new MultiModelProtocolError(`invalid arrival for ${key}`);
    }
    request.phase = "waiting";
    request.blockedSinceNs = atNs;
    this.emit({
      kind: "arrival",
      atNs,
      tenantId: request.tenantId,
      requestId: request.spec.id,
    });
  }

  private startBatchIfPossible(atNs: number): void {
    if (this.runningBatch) {
      return;
    }
    for (const tenant of this.pendingTenants()) {
      if (!tenant.resident) {
        continue;
      }
      const plan = this.planBatch(tenant);
      if (!plan) {
        continue;
      }
      const work = plan.work;
      const durationNs = this.estimateDuration(work, tenant.spec);
      assertPositiveSafeInteger(durationNs, `batch ${work.batchId} duration`);
      tenant.lastUsedSequence = ++this.useClock;
      this.runningBatch = { work, startedAtNs: atNs, durationNs };
      this.nextBatchId++;
      this.computeServiceNs += durationNs;
      if (this.runningLoad) {
        this.hiddenTransferNs += Math.max(
          0,
          Math.min(this.runningLoad.completesAtNs, atNs + durationNs) - atNs,
        );
      }
      for (const request of this.tenantRequests(tenant.spec.id)) {
        if (request.blockedSinceNs !== undefined) {
          request.residencyWaitNs += atNs - request.blockedSinceNs;
          request.blockedSinceNs = undefined;
        }
      }
      this.emit({ kind: "batch_start", atNs, batch: work, durationNs });
      this.eventLoop.scheduleAt(atNs + durationNs, {
        kind: "batch_finish",
        batchId: work.batchId,
      });
      return;
    }
  }

  private startLoadIfPossible(atNs: number): void {
    if (this.runningLoad) {
      return;
    }
    for (const tenant of this.pendingTenants()) {
      if (tenant.resident || tenant.loading) {
        continue;
      }
      if (!this.makeRoomFor(tenant, atNs)) {
        continue;
      }
      const bytes = tenant.footprintBytes;
      const durationNs = loadDurationNs(
        bytes,
        this.config.loadBandwidthBytesPerSec,
      );
      const completesAtNs = atNs + durationNs;
      tenant.loading = true;
      tenant.loads++;
      tenant.loadedBytes += bytes;
      tenant.loadServiceNs += durationNs;
      this.transferServiceNs += durationNs;
      if (this.runningBatch) {
        const batchEnd = this.runningBatch.startedAtNs
          + this.runningBatch.durationNs;
        this.hiddenTransferNs += Math.max(
          0,
          Math.min(batchEnd, completesAtNs) - atNs,
        );
      }
      this.emit({
        kind: "load_start",
        atNs,
        tenantId: tenant.spec.id,
        bytes,
        completesAtNs,
      });
      this.runningLoad = { tenantId: tenant.spec.id, completesAtNs };
      this.eventLoop.scheduleAt(completesAtNs, {
        kind: "load_finish",
        tenantId: tenant.spec.id,
      });
      return;
    }
  }

  /**
   * Evicts least-recently-used unpinned models until the candidate fits.
   *
   * A model only becomes evictable once it has retired a request since its
   * last load. Gating on a batch would not be enough: an eviction discards
   * partial prefill, so a model could be reloaded and evicted forever while
   * never finishing anything. Retiring a request makes progress monotone, so
   * residencies are bounded by the total request count.
   */
  private makeRoomFor(candidate: MutableTenant, atNs: number): boolean {
    const budget = this.config.deviceMemoryBytes;
    const fits = () => budget - this.residentBytes >= candidate.footprintBytes;
    if (fits()) {
      return true;
    }
    const victims = [...this.tenants.values()]
      .filter((tenant) => (
        tenant.resident
        && !tenant.spec.pinned
        && tenant.spec.id !== candidate.spec.id
        && tenant.completedSinceLoad > 0
        && this.runningBatch?.work.tenantId !== tenant.spec.id
      ))
      .sort((left, right) => (
        left.lastUsedSequence - right.lastUsedSequence
        || compareIds(left.spec.id, right.spec.id)
      ));
    for (const victim of victims) {
      this.evict(victim, atNs);
      if (fits()) {
        return true;
      }
    }
    return fits();
  }

  private evict(tenant: MutableTenant, atNs: number): void {
    const discardedKvTokens = tenant.kvTokens;
    // Releasing the KV arena discards context, so a partially generated
    // request has to prefill again when the model comes back.
    for (const request of this.tenantRequests(tenant.spec.id)) {
      if (request.phase === "completed" || request.phase === "unarrived") {
        continue;
      }
      tenant.rePrefilledTokens += request.promptProcessed;
      request.promptProcessed = 0;
      request.kvTokens = 0;
      request.outputEmitted = 0;
      request.phase = "waiting";
      request.blockedSinceNs ??= atNs;
    }
    tenant.kvTokens = 0;
    tenant.resident = false;
    tenant.evictions++;
    tenant.completedSinceLoad = 0;
    this.residentBytes -= tenant.footprintBytes;
    if (tenant.residentSinceNs !== undefined) {
      tenant.residentNs += atNs - tenant.residentSinceNs;
      tenant.residentSinceNs = undefined;
    }
    this.emit({
      kind: "evict",
      atNs,
      tenantId: tenant.spec.id,
      bytes: tenant.footprintBytes,
      discardedKvTokens,
    });
  }

  private handleLoadFinish(tenantId: string, atNs: number): void {
    const running = this.runningLoad;
    if (
      !running
      || running.tenantId !== tenantId
      || running.completesAtNs !== atNs
    ) {
      throw new MultiModelProtocolError(
        `invalid load completion for ${tenantId}`,
      );
    }
    const tenant = requireTenant(this.tenants, tenantId);
    tenant.loading = false;
    tenant.resident = true;
    tenant.residentSinceNs = atNs;
    tenant.completedSinceLoad = 0;
    this.residentBytes += tenant.footprintBytes;
    if (this.residentBytes > this.config.deviceMemoryBytes) {
      throw new MultiModelProtocolError(
        `resident bytes ${this.residentBytes} exceed device capacity ${this.config.deviceMemoryBytes}`,
      );
    }
    this.peakResidentBytes = Math.max(
      this.peakResidentBytes,
      this.residentBytes,
    );
    this.runningLoad = undefined;
    this.emit({
      kind: "load_complete",
      atNs,
      tenantId,
      residentBytes: this.residentBytes,
    });
  }

  private handleBatchFinish(batchId: number, atNs: number): void {
    const running = this.runningBatch;
    if (
      !running
      || running.work.batchId !== batchId
      || atNs !== running.startedAtNs + running.durationNs
    ) {
      throw new MultiModelProtocolError(`invalid finish for batch ${batchId}`);
    }
    const tenant = requireTenant(this.tenants, running.work.tenantId);
    const emitted: string[] = [];
    const completed: string[] = [];
    this.applyBatch(tenant, running.work, atNs, emitted, completed);
    this.runningBatch = undefined;
    this.emit({
      kind: "batch_finish",
      atNs,
      batchId,
      tenantId: tenant.spec.id,
      emittedRequestIds: emitted,
      completedRequestIds: completed,
    });
  }

  private applyBatch(
    tenant: MutableTenant,
    work: MultiModelBatchWork,
    atNs: number,
    emitted: string[],
    completed: string[],
  ): void {
    const plan = this.planBatch(tenant);
    if (
      plan === undefined
      || plan.work.prefillTokens !== work.prefillTokens
      || plan.work.decodeTokens !== work.decodeTokens
    ) {
      throw new MultiModelProtocolError(
        `batch ${work.batchId} does not match ${tenant.spec.id} state`,
      );
    }
    const emitToken = (request: MutableRequest): void => {
      request.outputEmitted++;
      request.firstTokenNs ??= atNs;
      emitted.push(request.spec.id);
      if (request.outputEmitted === request.spec.outputTokens) {
        request.phase = "completed";
        request.completedAtNs = atNs;
        tenant.kvTokens -= request.kvTokens;
        request.kvTokens = 0;
        this.completedRequests++;
        tenant.completedSinceLoad++;
        completed.push(request.spec.id);
      }
    };
    for (const { request } of plan.decode) {
      request.kvTokens++;
      tenant.kvTokens++;
      emitToken(request);
    }
    for (const { request, tokens } of plan.prefill) {
      request.promptProcessed += tokens;
      request.kvTokens += tokens;
      tenant.kvTokens += tokens;
      if (request.promptProcessed === request.spec.promptTokens) {
        // The final prefill chunk emits the first token from its own logits.
        request.phase = "decoding";
        emitToken(request);
      }
    }
  }

  /**
   * Decode-first batching within one model, mirroring the single-model
   * scheduler: decode reserves first, chunked prefill fills the rest, and KV
   * is only granted to a request whose whole remaining need fits.
   */
  private planBatch(tenant: MutableTenant): BatchPlan | undefined {
    let tokenBudget = this.config.maxBatchTokens;
    let reservedKv = 0;
    const decode: { readonly request: MutableRequest }[] = [];
    const prefill: { readonly request: MutableRequest; readonly tokens: number }[] = [];
    const candidates = this.tenantRequests(tenant.spec.id);
    const availableKv = () => (
      tenant.spec.maxKvTokens - tenant.kvTokens - reservedKv
    );

    for (const request of candidates) {
      if (request.phase !== "decoding" || tokenBudget === 0) {
        continue;
      }
      if (availableKv() <= 0 || remainingKvNeed(request) > availableKv()) {
        continue;
      }
      decode.push({ request });
      reservedKv++;
      tokenBudget--;
    }
    for (const request of candidates) {
      if (request.phase !== "waiting" || tokenBudget === 0) {
        continue;
      }
      if (remainingKvNeed(request) > availableKv()) {
        continue;
      }
      const tokens = Math.min(
        request.spec.promptTokens - request.promptProcessed,
        this.config.prefillChunkTokens,
        tokenBudget,
        availableKv(),
      );
      if (tokens <= 0) {
        continue;
      }
      prefill.push({ request, tokens });
      reservedKv += tokens;
      tokenBudget -= tokens;
    }
    const prefillTokens = prefill.reduce((sum, slice) => sum + slice.tokens, 0);
    const tokenWork = prefillTokens + decode.length;
    if (tokenWork === 0) {
      return undefined;
    }
    return {
      work: {
        batchId: this.nextBatchId,
        tenantId: tenant.spec.id,
        prefill: prefill.map((slice) => ({
          requestId: slice.request.spec.id,
          tokens: slice.tokens,
        })),
        decode: decode.map((slice) => slice.request.spec.id),
        prefillTokens,
        decodeTokens: decode.length,
        sequenceCount: prefill.length + decode.length,
        tokenWork,
      },
      decode,
      prefill,
    };
  }

  /**
   * Models with work to do. A model that has not run recently is offered the
   * device first, so two busy models alternate at request granularity instead
   * of the earliest arrival monopolising the machine.
   */
  private pendingTenants(): MutableTenant[] {
    const oldest = new Map<string, number>();
    for (const request of this.requests.values()) {
      if (request.phase === "unarrived" || request.phase === "completed") {
        continue;
      }
      const current = oldest.get(request.tenantId);
      if (current === undefined || request.spec.arrivalNs < current) {
        oldest.set(request.tenantId, request.spec.arrivalNs);
      }
    }
    return [...oldest.keys()]
      .map((id) => requireTenant(this.tenants, id))
      .sort((left, right) => (
        left.lastUsedSequence - right.lastUsedSequence
        || oldest.get(left.spec.id)! - oldest.get(right.spec.id)!
        || compareIds(left.spec.id, right.spec.id)
      ));
  }

  private tenantRequests(tenantId: string): readonly MutableRequest[] {
    return this.requestsByTenant.get(tenantId) ?? [];
  }

  private emit(event: MultiModelTraceEvent): void {
    this.trace.push(event);
  }

  private requestResults(): MultiModelRequestResult[] {
    return [...this.requests.values()]
      .map((request) => ({
        tenantId: request.tenantId,
        requestId: request.spec.id,
        arrivalNs: request.spec.arrivalNs,
        firstTokenNs: request.firstTokenNs ?? 0,
        completedAtNs: request.completedAtNs ?? 0,
        timeToFirstTokenNs: (request.firstTokenNs ?? 0) - request.spec.arrivalNs,
        residencyWaitNs: request.residencyWaitNs,
      }))
      .sort((left, right) => (
        compareIds(left.tenantId, right.tenantId)
        || compareIds(left.requestId, right.requestId)
      ));
  }

  private buildMetrics(totalDurationNs: number): MultiModelMetrics {
    const results = this.requestResults();
    const tenants = [...this.tenants.values()]
      .sort((left, right) => compareIds(left.spec.id, right.spec.id))
      .map((tenant): MultiModelTenantMetrics => {
        const own = results.filter(
          (result) => result.tenantId === tenant.spec.id,
        );
        const ttfts = own.map((result) => result.timeToFirstTokenNs);
        return {
          tenantId: tenant.spec.id,
          displayName: tenant.spec.displayName,
          requests: own.length,
          outputTokens: tenant.spec.requests.reduce(
            (sum, request) => sum + request.outputTokens,
            0,
          ),
          residencyFootprintBytes: tenant.footprintBytes,
          loads: tenant.loads,
          evictions: tenant.evictions,
          loadedBytes: tenant.loadedBytes,
          loadServiceNs: tenant.loadServiceNs,
          residencyWaitNs: own.reduce(
            (sum, result) => sum + result.residencyWaitNs,
            0,
          ),
          averageTimeToFirstTokenNs: average(ttfts),
          p95TimeToFirstTokenNs: percentile(ttfts, 0.95),
          averageRequestLatencyNs: average(
            own.map((result) => result.completedAtNs - result.arrivalNs),
          ),
          rePrefilledTokens: tenant.rePrefilledTokens,
          residentNs: tenant.residentNs + (
            tenant.residentSinceNs === undefined
              ? 0
              : totalDurationNs - tenant.residentSinceNs
          ),
        };
      });
    const totalLoads = tenants.reduce((sum, tenant) => sum + tenant.loads, 0);
    const everLoaded = tenants.filter((tenant) => tenant.loads > 0).length;
    const totalRequests = results.length;
    return {
      tenants,
      totalDurationNs,
      computeServiceNs: this.computeServiceNs,
      transferServiceNs: this.transferServiceNs,
      computeUtilization: totalDurationNs === 0
        ? 0
        : this.computeServiceNs / totalDurationNs,
      transferUtilization: totalDurationNs === 0
        ? 0
        : this.transferServiceNs / totalDurationNs,
      hiddenTransferNs: this.hiddenTransferNs,
      peakResidentBytes: this.peakResidentBytes,
      deviceMemoryBytes: this.config.deviceMemoryBytes,
      totalLoads,
      totalEvictions: tenants.reduce(
        (sum, tenant) => sum + tenant.evictions,
        0,
      ),
      totalLoadedBytes: tenants.reduce(
        (sum, tenant) => sum + tenant.loadedBytes,
        0,
      ),
      reloadsPerRequest: totalRequests === 0
        ? 0
        : (totalLoads - everLoaded) / totalRequests,
      fitsWithoutSwapping: tenants.every((tenant) => tenant.evictions === 0),
    };
  }
}

interface BatchPlan {
  readonly work: MultiModelBatchWork;
  readonly decode: readonly { readonly request: MutableRequest }[];
  readonly prefill: readonly {
    readonly request: MutableRequest;
    readonly tokens: number;
  }[];
}

// ============================================================
// Entry points
// ============================================================

export function simulateMultiModelWorkload(
  config: MultiModelConfig,
  estimateDuration: MultiModelBatchDurationEstimator,
): MultiModelResult {
  validateMultiModelConfig(config);
  return new MultiModelSimulator(config, estimateDuration).run();
}

/**
 * Independent replay. Residency, memory, and token state are re-derived from
 * the trace alone rather than trusting the simulator's own bookkeeping.
 */
export function replayMultiModelTrace(
  config: MultiModelConfig,
  trace: readonly MultiModelTraceEvent[],
  estimateDuration: MultiModelBatchDurationEstimator,
): MultiModelReplayResult {
  validateMultiModelConfig(config);
  const specs = new Map(config.tenants.map((tenant) => [tenant.id, tenant]));
  const resident = new Map<string, boolean>();
  const kvTokens = new Map<string, number>();
  const requestKv = new Map<string, number>();
  const emitted = new Map<string, number>();
  const lastBatch = new Map<string, MultiModelBatchWork>();
  let residentBytes = 0;
  let currentTimeNs = 0;
  let completedRequests = 0;
  let terminalSeen = false;

  const fail = (message: string): never => {
    throw new MultiModelProtocolError(`co-residency replay: ${message}`);
  };

  for (const [index, event] of trace.entries()) {
    if (terminalSeen) {
      fail(`event ${index} follows the terminal event`);
    }
    if (event.atNs < currentTimeNs) {
      fail(`event ${index} moves time backwards`);
    }
    currentTimeNs = event.atNs;
    switch (event.kind) {
      case "arrival": {
        if (!specs.has(event.tenantId)) {
          fail(`arrival names unknown model ${event.tenantId}`);
        }
        break;
      }
      case "evict": {
        const spec = specs.get(event.tenantId);
        if (!spec) {
          fail(`eviction names unknown model ${event.tenantId}`);
        }
        if (resident.get(event.tenantId) !== true) {
          fail(`evicted model ${event.tenantId} was not resident`);
        }
        if (spec!.pinned) {
          fail(`pinned model ${event.tenantId} was evicted`);
        }
        if (event.bytes !== tenantFootprintBytes(spec!)) {
          fail(`eviction of ${event.tenantId} released the wrong extent`);
        }
        if (event.discardedKvTokens !== (kvTokens.get(event.tenantId) ?? 0)) {
          fail(`eviction of ${event.tenantId} discarded the wrong KV extent`);
        }
        resident.set(event.tenantId, false);
        kvTokens.set(event.tenantId, 0);
        residentBytes -= event.bytes;
        // Losing the arena resets partial generation for that model.
        for (const request of spec!.requests) {
          const key = requestKey(spec!.id, request.id);
          if (emitted.get(key) !== request.outputTokens) {
            emitted.delete(key);
          }
          requestKv.set(key, 0);
        }
        break;
      }
      case "load_start": {
        const spec = specs.get(event.tenantId);
        if (!spec) {
          fail(`load names unknown model ${event.tenantId}`);
        }
        if (resident.get(event.tenantId) === true) {
          fail(`model ${event.tenantId} was loaded while resident`);
        }
        if (event.bytes !== tenantFootprintBytes(spec!)) {
          fail(`load of ${event.tenantId} moved the wrong extent`);
        }
        const expected = loadDurationNs(
          event.bytes,
          config.loadBandwidthBytesPerSec,
        );
        if (event.completesAtNs !== event.atNs + expected) {
          fail(`load of ${event.tenantId} does not match link bandwidth`);
        }
        break;
      }
      case "load_complete": {
        const spec = specs.get(event.tenantId);
        if (!spec) {
          fail(`load completion names unknown model ${event.tenantId}`);
        }
        resident.set(event.tenantId, true);
        residentBytes += tenantFootprintBytes(spec!);
        if (residentBytes > config.deviceMemoryBytes) {
          fail(
            `resident bytes ${residentBytes} exceed device capacity ${config.deviceMemoryBytes}`,
          );
        }
        if (event.residentBytes !== residentBytes) {
          fail(`load of ${event.tenantId} reports the wrong resident extent`);
        }
        break;
      }
      case "batch_start": {
        const spec = specs.get(event.batch.tenantId);
        if (!spec) {
          fail(`batch names unknown model ${event.batch.tenantId}`);
        }
        if (resident.get(event.batch.tenantId) !== true) {
          fail(`batch ran for evicted model ${event.batch.tenantId}`);
        }
        if (
          event.batch.tokenWork
          !== event.batch.prefillTokens + event.batch.decodeTokens
        ) {
          fail(`batch ${event.batch.batchId} token work does not match its slices`);
        }
        if (event.batch.tokenWork > config.maxBatchTokens) {
          fail(`batch ${event.batch.batchId} exceeds the token budget`);
        }
        if (estimateDuration(event.batch, spec!) !== event.durationNs) {
          fail(`batch ${event.batch.batchId} duration does not match the cost model`);
        }
        if (
          event.batch.prefillTokens
          !== event.batch.prefill.reduce((sum, slice) => sum + slice.tokens, 0)
          || event.batch.decodeTokens !== event.batch.decode.length
        ) {
          fail(`batch ${event.batch.batchId} totals do not match its slices`);
        }
        lastBatch.set(event.batch.tenantId, event.batch);
        break;
      }
      case "batch_finish": {
        const spec = specs.get(event.tenantId);
        if (!spec) {
          fail(`batch completion names unknown model ${event.tenantId}`);
        }
        if (resident.get(event.tenantId) !== true) {
          fail(`batch finished for evicted model ${event.tenantId}`);
        }
        const batch = lastBatch.get(event.tenantId);
        if (!batch || batch.batchId !== event.batchId) {
          fail(`completion of batch ${event.batchId} has no matching start`);
        }
        let tokens = kvTokens.get(event.tenantId) ?? 0;
        const chargeKv = (requestId: string, amount: number): void => {
          const key = requestKey(event.tenantId, requestId);
          requestKv.set(key, (requestKv.get(key) ?? 0) + amount);
          tokens += amount;
        };
        for (const slice of batch!.prefill) {
          chargeKv(slice.requestId, slice.tokens);
        }
        for (const requestId of batch!.decode) {
          chargeKv(requestId, 1);
        }
        for (const requestId of event.emittedRequestIds) {
          const key = requestKey(event.tenantId, requestId);
          emitted.set(key, (emitted.get(key) ?? 0) + 1);
        }
        for (const requestId of event.completedRequestIds) {
          const request = spec!.requests.find((r) => r.id === requestId);
          if (!request) {
            fail(`completion names unknown request ${requestId}`);
          }
          const key = requestKey(event.tenantId, requestId);
          if (emitted.get(key) !== request!.outputTokens) {
            fail(`request ${requestId} completed with the wrong token count`);
          }
          // Completion releases the whole extent the request held.
          tokens -= requestKv.get(key) ?? 0;
          requestKv.set(key, 0);
          completedRequests++;
        }
        if (tokens > spec!.maxKvTokens) {
          fail(`model ${event.tenantId} exceeded its KV arena`);
        }
        if (tokens < 0) {
          fail(`model ${event.tenantId} released more KV than it held`);
        }
        kvTokens.set(event.tenantId, tokens);
        break;
      }
      case "terminal": {
        if (event.residentBytes !== residentBytes) {
          fail("terminal event reports the wrong resident extent");
        }
        if (event.completedRequests !== completedRequests) {
          fail("terminal event reports the wrong completion count");
        }
        terminalSeen = true;
        break;
      }
    }
  }
  if (!terminalSeen) {
    fail("trace lacks a terminal event");
  }
  const expected = config.tenants.reduce(
    (sum, tenant) => sum + tenant.requests.length,
    0,
  );
  if (completedRequests !== expected) {
    fail(`trace completes ${completedRequests} of ${expected} requests`);
  }
  return {
    appliedEvents: trace.length,
    completedRequests,
    finalResidentBytes: residentBytes,
  };
}

// ============================================================
// Validation
// ============================================================

export function validateMultiModelConfig(config: MultiModelConfig): void {
  if (config.tenants.length === 0) {
    throw new MultiModelProtocolError(
      "co-residency requires at least one model",
    );
  }
  assertPositiveSafeInteger(config.deviceMemoryBytes, "deviceMemoryBytes");
  assertPositiveSafeInteger(config.maxBatchTokens, "maxBatchTokens");
  assertPositiveSafeInteger(config.prefillChunkTokens, "prefillChunkTokens");
  if (
    !Number.isFinite(config.loadBandwidthBytesPerSec)
    || config.loadBandwidthBytesPerSec <= 0
  ) {
    throw new MultiModelProtocolError(
      "load bandwidth must be a positive finite rate",
    );
  }
  const ids = new Set<string>();
  for (const tenant of config.tenants) {
    if (tenant.id.length === 0 || ids.has(tenant.id)) {
      throw new MultiModelProtocolError(
        `model id must be non-empty and unique; got ${tenant.id}`,
      );
    }
    ids.add(tenant.id);
    assertPositiveSafeInteger(tenant.weightBytes, `${tenant.id} weightBytes`);
    assertPositiveSafeInteger(tenant.maxKvTokens, `${tenant.id} maxKvTokens`);
    if (!Number.isSafeInteger(tenant.kvBytesPerToken) || tenant.kvBytesPerToken < 0) {
      throw new MultiModelProtocolError(
        `${tenant.id} kvBytesPerToken must be a non-negative safe integer`,
      );
    }
    if (tenant.requests.length === 0) {
      throw new MultiModelProtocolError(`${tenant.id} has no requests`);
    }
    const requestIds = new Set<string>();
    for (const request of tenant.requests) {
      if (request.id.length === 0 || requestIds.has(request.id)) {
        throw new MultiModelProtocolError(
          `${tenant.id} request id must be non-empty and unique`,
        );
      }
      requestIds.add(request.id);
      if (!Number.isSafeInteger(request.arrivalNs) || request.arrivalNs < 0) {
        throw new MultiModelProtocolError(
          `${tenant.id}/${request.id} arrivalNs must be a non-negative safe integer`,
        );
      }
      assertPositiveSafeInteger(
        request.promptTokens,
        `${tenant.id}/${request.id} promptTokens`,
      );
      assertPositiveSafeInteger(
        request.outputTokens,
        `${tenant.id}/${request.id} outputTokens`,
      );
      const peakKv = request.promptTokens + request.outputTokens - 1;
      if (peakKv > tenant.maxKvTokens) {
        throw new MultiModelProtocolError(
          `${tenant.id}/${request.id} needs ${peakKv} KV tokens but its arena holds ${tenant.maxKvTokens}`,
        );
      }
    }
    // A model whose residency footprint exceeds the device can never answer,
    // so the configuration fails closed instead of stalling at run time.
    const footprint = tenantFootprintBytes(tenant);
    if (footprint > config.deviceMemoryBytes) {
      throw new MultiModelProtocolError(
        `${tenant.id} needs ${footprint} resident bytes but the device holds ${config.deviceMemoryBytes}`,
      );
    }
  }
  const pinnedBytes = config.tenants
    .filter((tenant) => tenant.pinned)
    .reduce((sum, tenant) => sum + tenantFootprintBytes(tenant), 0);
  if (pinnedBytes > config.deviceMemoryBytes) {
    throw new MultiModelProtocolError(
      `pinned models need ${pinnedBytes} resident bytes but the device holds ${config.deviceMemoryBytes}`,
    );
  }
  // Evicting every unpinned model is the most room the device can ever offer,
  // so the largest unpinned model must still fit beside the pinned set.
  const largestUnpinned = config.tenants
    .filter((tenant) => !tenant.pinned)
    .reduce((max, tenant) => Math.max(max, tenantFootprintBytes(tenant)), 0);
  if (pinnedBytes + largestUnpinned > config.deviceMemoryBytes) {
    throw new MultiModelProtocolError(
      `pinned models leave ${config.deviceMemoryBytes - pinnedBytes} bytes, too little for the largest unpinned model at ${largestUnpinned} bytes`,
    );
  }
}

// ============================================================
// Helpers
// ============================================================

function requestKey(tenantId: string, requestId: string): string {
  return `${tenantId}\u0000${requestId}`;
}

function loadDurationNs(bytes: number, bytesPerSecond: number): number {
  return Math.max(1, Math.ceil(bytes / bytesPerSecond * 1e9));
}

function remainingKvNeed(request: MutableRequest): number {
  const peak = request.spec.promptTokens + request.spec.outputTokens - 1;
  return peak - request.kvTokens;
}

function requireTenant(
  tenants: ReadonlyMap<string, MutableTenant>,
  id: string,
): MutableTenant {
  const tenant = tenants.get(id);
  if (!tenant) {
    throw new MultiModelProtocolError(`unknown model ${id}`);
  }
  return tenant;
}

function assertPositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new MultiModelProtocolError(
      `${label} must be a positive safe integer`,
    );
  }
}

function average(values: readonly number[]): number {
  return values.length === 0
    ? 0
    : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values: readonly number[], fraction: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(fraction * sorted.length) - 1),
  );
  return sorted[index]!;
}
