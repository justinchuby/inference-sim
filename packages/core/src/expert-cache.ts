import { DiscreteEventSimulator } from "./event-loop.js";

export const EXPERT_CACHE_CONTRACT_REVISION = 1;

export type ExpertCacheTier = "hot" | "warm" | "cold";
export type ExpertLoadTarget = "hot" | "warm";
export type ExpertLoadKind = "demand" | "prefetch";

export interface ExpertSpec {
  readonly id: string;
  readonly bytes: number;
  readonly routingWeight?: number;
}

export interface ExpertCacheConfig {
  readonly experts: readonly ExpertSpec[];
  readonly hotCapacityBytes: number;
  readonly warmCapacityBytes: number;
  readonly warmToHotLatencyNs: number;
  readonly coldToHotLatencyNs: number;
  readonly coldToWarmLatencyNs: number;
  readonly routingSeed: number;
  readonly initialHotExpertIds?: readonly string[];
  readonly initialWarmExpertIds?: readonly string[];
  readonly sourceId?: string;
}

export interface ExpertRouteRequest {
  readonly tokenIndex: number;
  readonly topK: number;
  readonly atNs: number;
}

export interface ExpertRouteResult {
  readonly requestId: string;
  readonly expertIds: readonly string[];
  readonly requestedAtNs: number;
  readonly readyAtNs: number;
  readonly stallNs: number;
  readonly sourceTiers: readonly ExpertCacheTier[];
}

export interface ExpertCacheMetrics {
  readonly routes: number;
  readonly routedExperts: number;
  readonly hotHits: number;
  readonly warmMisses: number;
  readonly coldMisses: number;
  readonly demandLoads: number;
  readonly prefetchLoads: number;
  readonly evictions: number;
  readonly bytesMoved: number;
  readonly stallNs: number;
  readonly hotHitRate: number;
  readonly highWaterHotBytes: number;
  readonly highWaterWarmBytes: number;
}

export interface ExpertPendingLoadSnapshot {
  readonly loadId: string;
  readonly expertId: string;
  readonly sourceTier: "warm" | "cold";
  readonly targetTier: ExpertLoadTarget;
  readonly kind: ExpertLoadKind;
  readonly startedAtNs: number;
  readonly completesAtNs: number;
  readonly bytes: number;
}

export interface ExpertCacheSnapshot {
  readonly currentTimeNs: number;
  readonly hotCapacityBytes: number;
  readonly warmCapacityBytes: number;
  readonly hotResidentBytes: number;
  readonly warmResidentBytes: number;
  readonly hotReservedBytes: number;
  readonly warmReservedBytes: number;
  readonly hotExpertIds: readonly string[];
  readonly warmExpertIds: readonly string[];
  readonly pendingLoads: readonly ExpertPendingLoadSnapshot[];
  readonly metrics: ExpertCacheMetrics;
}

interface ExpertCacheTraceEnvelope {
  readonly contractRevision: typeof EXPERT_CACHE_CONTRACT_REVISION;
  readonly sourceId: string;
  readonly sourceSequence: number;
}

export type ExpertCacheTraceEvent = ExpertCacheTraceEnvelope & (
  | {
      readonly kind: "initialize";
      readonly config: ExpertCacheConfig;
    }
  | {
      readonly kind: "route";
      readonly requestId: string;
      readonly tokenIndex: number;
      readonly topK: number;
      readonly atNs: number;
      readonly expertIds: readonly string[];
    }
  | {
      readonly kind: "prefetch";
      readonly atNs: number;
      readonly targetTier: ExpertLoadTarget;
      readonly expertIds: readonly string[];
    }
  | {
      readonly kind: "evict";
      readonly atNs: number;
      readonly tier: ExpertLoadTarget;
      readonly expertId: string;
      readonly bytes: number;
    }
  | {
      readonly kind: "load_start";
      readonly load: ExpertPendingLoadSnapshot;
    }
  | {
      readonly kind: "load_complete";
      readonly loadId: string;
      readonly expertId: string;
      readonly targetTier: ExpertLoadTarget;
      readonly completedAtNs: number;
    }
  | {
      readonly kind: "access";
      readonly requestId: string;
      readonly requestedAtNs: number;
      readonly readyAtNs: number;
      readonly expertIds: readonly string[];
      readonly sourceTiers: readonly ExpertCacheTier[];
      readonly stallNs: number;
  }
);

type WithoutTraceEnvelope<T> = T extends ExpertCacheTraceEnvelope
  ? Omit<T, keyof ExpertCacheTraceEnvelope>
  : never;

type ExpertCacheTracePayload = WithoutTraceEnvelope<ExpertCacheTraceEvent>;

export interface ExpertCacheReplayResult {
  readonly appliedEvents: number;
  readonly snapshot: ExpertCacheSnapshot;
}

export interface ExpertCacheWorkloadConfig {
  readonly cache: ExpertCacheConfig;
  readonly tokenCount: number;
  readonly topK: number;
  readonly startAtNs?: number;
  readonly tokenIntervalNs: number;
  readonly initialPrefetch?: {
    readonly expertIds: readonly string[];
    readonly targetTier: ExpertLoadTarget;
    readonly leadTimeNs: number;
  };
}

export interface ExpertCacheWorkloadResult {
  readonly routes: readonly ExpertRouteResult[];
  readonly snapshot: ExpertCacheSnapshot;
  readonly trace: readonly ExpertCacheTraceEvent[];
}

interface MutableMetrics {
  routes: number;
  routedExperts: number;
  hotHits: number;
  warmMisses: number;
  coldMisses: number;
  demandLoads: number;
  prefetchLoads: number;
  evictions: number;
  bytesMoved: number;
  stallNs: number;
  highWaterHotBytes: number;
  highWaterWarmBytes: number;
}

interface CompletionEvent {
  readonly loadId: string;
}

export class ExpertCacheProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ExpertCacheProtocolError";
  }
}

export class ExpertCacheReplayError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ExpertCacheReplayError";
  }
}

export class ExpertCacheSimulator {
  private readonly config: ExpertCacheConfig;
  private readonly sourceId: string;
  private readonly experts: Map<string, ExpertSpec>;
  private readonly hot = new Map<string, number>();
  private readonly warm = new Map<string, number>();
  private readonly pending = new Map<string, ExpertPendingLoadSnapshot>();
  private readonly pendingByTarget = new Map<string, string>();
  private readonly events: ExpertCacheTraceEvent[] = [];
  private readonly eventLoop = new DiscreteEventSimulator<CompletionEvent>();
  private readonly rng: DeterministicRng;
  private readonly metrics: MutableMetrics = emptyMetrics();
  private currentTimeNs = 0;
  private hotReservedBytes = 0;
  private warmReservedBytes = 0;
  private accessClock = 0;
  private nextLoadId = 1;
  private nextRequestId = 1;
  private nextSourceSequence = 0;

  constructor(config: ExpertCacheConfig) {
    validateConfig(config);
    this.config = cloneConfig(config);
    this.sourceId = config.sourceId ?? "expert-cache";
    this.experts = new Map(config.experts.map((expert) => [expert.id, expert]));
    this.rng = new DeterministicRng(config.routingSeed);

    for (const id of config.initialWarmExpertIds ?? []) {
      this.installInitial(id, "warm");
    }
    for (const id of config.initialHotExpertIds ?? []) {
      this.installInitial(id, "hot");
    }
    this.updateHighWater();
    this.commitEvent({ kind: "initialize", config: cloneConfig(config) });
  }

  processToken(request: ExpertRouteRequest): ExpertRouteResult {
    assertNonNegativeSafeInteger(request.tokenIndex, "token index");
    assertPositiveSafeInteger(request.topK, "topK");
    this.advanceTo(request.atNs);
    if (request.topK > this.experts.size) {
      throw new ExpertCacheProtocolError(
        `topK ${request.topK} exceeds expert count ${this.experts.size}`,
      );
    }

    const rngCheckpoint = this.rng.snapshot();
    const selected = selectWithoutReplacement(
      [...this.experts.values()],
      request.topK,
      this.rng,
    );
    const selectedBytes = selected.reduce((sum, expert) => (
      checkedAdd(sum, expert.bytes, "routed expert bytes")
    ), 0);
    if (selectedBytes > this.config.hotCapacityBytes) {
      this.rng.restore(rngCheckpoint);
      throw new ExpertCacheProtocolError(
        `routed working set requires ${selectedBytes} hot bytes but capacity is ${this.config.hotCapacityBytes}`,
      );
    }

    const expertIds = selected.map((expert) => expert.id);
    const sourceTiers = expertIds.map((id) => this.tierFor(id));
    const newDemandBytes = expertIds.reduce((sum, id) => (
      this.hot.has(id) || this.pendingTarget(id, "hot")
        ? sum
        : checkedAdd(sum, this.requireExpert(id).bytes, "new demand bytes")
    ), 0);
    const protectedHotBytes = expertIds.reduce((sum, id) => (
      this.hot.has(id)
        ? checkedAdd(sum, this.requireExpert(id).bytes, "protected hot bytes")
        : sum
    ), 0);
    if (
      protectedHotBytes + this.hotReservedBytes + newDemandBytes
      > this.config.hotCapacityBytes
    ) {
      this.rng.restore(rngCheckpoint);
      throw new ExpertCacheProtocolError(
        "pending hot reservations leave insufficient capacity for the routed working set",
      );
    }
    const requestId = `expert-request-${this.nextRequestId++}`;
    this.metrics.routes++;
    this.metrics.routedExperts += expertIds.length;
    this.commitEvent({
      kind: "route",
      requestId,
      tokenIndex: request.tokenIndex,
      topK: request.topK,
      atNs: request.atNs,
      expertIds,
    });

    const protectedIds = new Set(expertIds);
    let readyAtNs = request.atNs;
    for (let index = 0; index < expertIds.length; index++) {
      const id = expertIds[index];
      const sourceTier = sourceTiers[index];
      if (sourceTier === "hot") {
        this.metrics.hotHits++;
        continue;
      }
      if (sourceTier === "warm") {
        this.metrics.warmMisses++;
      } else {
        this.metrics.coldMisses++;
      }
      const existing = this.pendingTarget(id, "hot");
      const load = existing
        ?? this.startLoad(id, "hot", "demand", request.atNs, protectedIds);
      readyAtNs = Math.max(readyAtNs, load.completesAtNs);
    }

    this.advanceTo(readyAtNs);
    for (const id of expertIds) {
      if (!this.hot.has(id)) {
        throw new ExpertCacheProtocolError(
          `expert ${id} was not hot when request ${requestId} became ready`,
        );
      }
      this.touch(this.hot, id);
    }
    const stallNs = readyAtNs - request.atNs;
    this.metrics.stallNs = checkedAdd(
      this.metrics.stallNs,
      stallNs,
      "expert cache stall",
    );
    this.commitEvent({
      kind: "access",
      requestId,
      requestedAtNs: request.atNs,
      readyAtNs,
      expertIds,
      sourceTiers,
      stallNs,
    });
    return {
      requestId,
      expertIds,
      requestedAtNs: request.atNs,
      readyAtNs,
      stallNs,
      sourceTiers,
    };
  }

  prefetch(
    expertIds: readonly string[],
    targetTier: ExpertLoadTarget,
    atNs: number,
  ): readonly string[] {
    this.advanceTo(atNs);
    const uniqueIds = unique(expertIds);
    for (const id of uniqueIds) {
      this.requireExpert(id);
    }
    const resident = targetTier === "hot" ? this.hot : this.warm;
    const reserved = targetTier === "hot"
      ? this.hotReservedBytes
      : this.warmReservedBytes;
    const capacity = targetTier === "hot"
      ? this.config.hotCapacityBytes
      : this.config.warmCapacityBytes;
    const incomingBytes = uniqueIds.reduce((sum, id) => (
      resident.has(id) || this.pendingTarget(id, targetTier)
        ? sum
        : checkedAdd(sum, this.requireExpert(id).bytes, "prefetch bytes")
    ), 0);
    if (reserved + incomingBytes > capacity) {
      throw new ExpertCacheProtocolError(
        `pending reservations plus prefetch require ${reserved + incomingBytes} ${targetTier} bytes but capacity is ${capacity}`,
      );
    }
    this.commitEvent({
      kind: "prefetch",
      atNs,
      targetTier,
      expertIds: uniqueIds,
    });

    const loadIds: string[] = [];
    for (const id of uniqueIds) {
      const resident = targetTier === "hot" ? this.hot.has(id) : this.warm.has(id);
      if (resident || this.pendingTarget(id, targetTier)) {
        continue;
      }
      loadIds.push(
        this.startLoad(id, targetTier, "prefetch", atNs, new Set()).loadId,
      );
    }
    return loadIds;
  }

  advanceTo(timestampNs: number): void {
    assertNonNegativeSafeInteger(timestampNs, "advance timestamp");
    if (timestampNs < this.currentTimeNs) {
      throw new ExpertCacheProtocolError(
        `cannot move cache time backward from ${this.currentTimeNs}ns to ${timestampNs}ns`,
      );
    }
    this.eventLoop.run((event) => {
      this.completeLoad(event.payload.loadId, event.timestampNs);
    }, { untilNs: timestampNs });
    this.currentTimeNs = timestampNs;
  }

  snapshot(): ExpertCacheSnapshot {
    return buildSnapshot({
      currentTimeNs: this.currentTimeNs,
      config: this.config,
      experts: this.experts,
      hot: this.hot,
      warm: this.warm,
      hotReservedBytes: this.hotReservedBytes,
      warmReservedBytes: this.warmReservedBytes,
      pending: this.pending,
      metrics: this.metrics,
    });
  }

  trace(): readonly ExpertCacheTraceEvent[] {
    return structuredClone(this.events);
  }

  private startLoad(
    expertId: string,
    targetTier: ExpertLoadTarget,
    kind: ExpertLoadKind,
    atNs: number,
    protectedIds: ReadonlySet<string>,
  ): ExpertPendingLoadSnapshot {
    const expert = this.requireExpert(expertId);
    const sourceTier: "warm" | "cold" =
      targetTier === "hot" && this.warm.has(expertId) ? "warm" : "cold";
    const latencyNs = sourceTier === "warm"
      ? this.config.warmToHotLatencyNs
      : targetTier === "hot"
        ? this.config.coldToHotLatencyNs
        : this.config.coldToWarmLatencyNs;
    const completesAtNs = checkedAdd(atNs, latencyNs, "expert load completion");
    const resident = targetTier === "hot" ? this.hot : this.warm;
    const capacity = targetTier === "hot"
      ? this.config.hotCapacityBytes
      : this.config.warmCapacityBytes;
    const reserved = targetTier === "hot"
      ? this.hotReservedBytes
      : this.warmReservedBytes;
    const victims = chooseVictims(
      resident,
      this.experts,
      checkedAdd(residentBytes(resident, this.experts), reserved, "cache use"),
      expert.bytes,
      capacity,
      protectedIds,
    );
    for (const victimId of victims) {
      resident.delete(victimId);
      const victim = this.requireExpert(victimId);
      this.metrics.evictions++;
      this.commitEvent({
        kind: "evict",
        atNs,
        tier: targetTier,
        expertId: victimId,
        bytes: victim.bytes,
      });
    }

    const load: ExpertPendingLoadSnapshot = {
      loadId: `expert-load-${this.nextLoadId++}`,
      expertId,
      sourceTier,
      targetTier,
      kind,
      startedAtNs: atNs,
      completesAtNs,
      bytes: expert.bytes,
    };
    if (targetTier === "hot") {
      this.hotReservedBytes = checkedAdd(
        this.hotReservedBytes,
        expert.bytes,
        "hot reserved bytes",
      );
    } else {
      this.warmReservedBytes = checkedAdd(
        this.warmReservedBytes,
        expert.bytes,
        "warm reserved bytes",
      );
    }
    this.pending.set(load.loadId, load);
    this.pendingByTarget.set(targetKey(expertId, targetTier), load.loadId);
    if (kind === "demand") {
      this.metrics.demandLoads++;
    } else {
      this.metrics.prefetchLoads++;
    }
    this.metrics.bytesMoved = checkedAdd(
      this.metrics.bytesMoved,
      expert.bytes,
      "expert bytes moved",
    );
    this.updateHighWater();
    this.eventLoop.scheduleAt(completesAtNs, { loadId: load.loadId });
    this.commitEvent({ kind: "load_start", load });
    return load;
  }

  private completeLoad(loadId: string, completedAtNs: number): void {
    const load = this.pending.get(loadId);
    if (!load) {
      throw new ExpertCacheProtocolError(`unknown pending load ${loadId}`);
    }
    if (load.completesAtNs !== completedAtNs) {
      throw new ExpertCacheProtocolError(
        `load ${loadId} completed at ${completedAtNs}ns, expected ${load.completesAtNs}ns`,
      );
    }
    const resident = load.targetTier === "hot" ? this.hot : this.warm;
    if (load.targetTier === "hot") {
      this.hotReservedBytes -= load.bytes;
    } else {
      this.warmReservedBytes -= load.bytes;
    }
    this.pending.delete(loadId);
    this.pendingByTarget.delete(targetKey(load.expertId, load.targetTier));
    this.touch(resident, load.expertId);
    this.commitEvent({
      kind: "load_complete",
      loadId,
      expertId: load.expertId,
      targetTier: load.targetTier,
      completedAtNs,
    });
  }

  private installInitial(expertId: string, tier: ExpertLoadTarget): void {
    const expert = this.requireExpert(expertId);
    const resident = tier === "hot" ? this.hot : this.warm;
    const capacity = tier === "hot"
      ? this.config.hotCapacityBytes
      : this.config.warmCapacityBytes;
    if (resident.has(expertId)) {
      throw new ExpertCacheProtocolError(
        `initial ${tier} expert ${expertId} is duplicated`,
      );
    }
    if (residentBytes(resident, this.experts) + expert.bytes > capacity) {
      throw new ExpertCacheProtocolError(
        `initial ${tier} experts exceed ${capacity} bytes`,
      );
    }
    this.touch(resident, expertId);
  }

  private tierFor(expertId: string): ExpertCacheTier {
    if (this.hot.has(expertId)) {
      return "hot";
    }
    if (this.warm.has(expertId)) {
      return "warm";
    }
    return "cold";
  }

  private pendingTarget(
    expertId: string,
    targetTier: ExpertLoadTarget,
  ): ExpertPendingLoadSnapshot | undefined {
    const loadId = this.pendingByTarget.get(targetKey(expertId, targetTier));
    return loadId ? this.pending.get(loadId) : undefined;
  }

  private requireExpert(id: string): ExpertSpec {
    const expert = this.experts.get(id);
    if (!expert) {
      throw new ExpertCacheProtocolError(`unknown expert ${id}`);
    }
    return expert;
  }

  private touch(resident: Map<string, number>, id: string): void {
    resident.set(id, ++this.accessClock);
  }

  private updateHighWater(): void {
    this.metrics.highWaterHotBytes = Math.max(
      this.metrics.highWaterHotBytes,
      residentBytes(this.hot, this.experts) + this.hotReservedBytes,
    );
    this.metrics.highWaterWarmBytes = Math.max(
      this.metrics.highWaterWarmBytes,
      residentBytes(this.warm, this.experts) + this.warmReservedBytes,
    );
  }

  private commitEvent(
    payload: ExpertCacheTracePayload,
  ): void {
    this.events.push({
      ...payload,
      contractRevision: EXPERT_CACHE_CONTRACT_REVISION,
      sourceId: this.sourceId,
      sourceSequence: this.nextSourceSequence++,
    } as ExpertCacheTraceEvent);
  }
}

export function replayExpertCacheTrace(
  trace: readonly ExpertCacheTraceEvent[],
): ExpertCacheReplayResult {
  if (trace.length === 0 || trace[0].kind !== "initialize") {
    throw new ExpertCacheReplayError("first event must initialize expert cache");
  }
  const initial = trace[0];
  try {
    validateConfig(initial.config);
  } catch (error) {
    throw new ExpertCacheReplayError(errorMessage(error));
  }
  const config = cloneConfig(initial.config);
  const experts = new Map(config.experts.map((expert) => [expert.id, expert]));
  const hot = new Map<string, number>();
  const warm = new Map<string, number>();
  const pending = new Map<string, ExpertPendingLoadSnapshot>();
  const pendingByTarget = new Map<string, string>();
  const outstandingRoutes = new Map<string, {
    readonly requestedAtNs: number;
    readonly expertIds: readonly string[];
    readonly sourceTiers: readonly ExpertCacheTier[];
  }>();
  const metrics = emptyMetrics();
  const rng = new DeterministicRng(config.routingSeed);
  let hotReservedBytes = 0;
  let warmReservedBytes = 0;
  let currentTimeNs = 0;
  let accessClock = 0;
  let sourceId = "";

  for (let index = 0; index < trace.length; index++) {
    const event = trace[index];
    try {
      if (event.contractRevision !== EXPERT_CACHE_CONTRACT_REVISION) {
        replayFail(`unsupported contract revision ${event.contractRevision}`);
      }
      if (event.sourceSequence !== index) {
        replayFail(`event ${index} has source sequence ${event.sourceSequence}`);
      }
      if (index === 0) {
        sourceId = event.sourceId;
      } else if (event.sourceId !== sourceId) {
        replayFail(`event ${index} changed source identity`);
      }

      switch (event.kind) {
        case "initialize": {
          if (index !== 0) {
            replayFail("cache initialized more than once");
          }
          for (const id of config.initialWarmExpertIds ?? []) {
            warm.set(id, ++accessClock);
          }
          for (const id of config.initialHotExpertIds ?? []) {
            hot.set(id, ++accessClock);
          }
          metrics.highWaterHotBytes = residentBytes(hot, experts);
          metrics.highWaterWarmBytes = residentBytes(warm, experts);
          break;
        }
        case "route": {
          requireMonotonicTime(event.atNs, currentTimeNs);
          currentTimeNs = event.atNs;
          const expected = selectWithoutReplacement(
            [...experts.values()],
            event.topK,
            rng,
          ).map((expert) => expert.id);
          assertArrayEqual(expected, event.expertIds, "route expert ids");
          if (outstandingRoutes.has(event.requestId)) {
            replayFail(`duplicate request id ${event.requestId}`);
          }
          outstandingRoutes.set(event.requestId, {
            requestedAtNs: event.atNs,
            expertIds: [...event.expertIds],
            sourceTiers: event.expertIds.map((id) => (
              hot.has(id) ? "hot" : warm.has(id) ? "warm" : "cold"
            )),
          });
          metrics.routes++;
          metrics.routedExperts += event.expertIds.length;
          break;
        }
        case "prefetch": {
          requireMonotonicTime(event.atNs, currentTimeNs);
          currentTimeNs = event.atNs;
          if (unique(event.expertIds).length !== event.expertIds.length) {
            replayFail("prefetch expert ids must be unique");
          }
          for (const id of event.expertIds) {
            if (!experts.has(id)) {
              replayFail(`prefetch references unknown expert ${id}`);
            }
          }
          break;
        }
        case "evict": {
          requireMonotonicTime(event.atNs, currentTimeNs);
          currentTimeNs = event.atNs;
          const resident = event.tier === "hot" ? hot : warm;
          const protectedIds = new Set(
            [...outstandingRoutes.values()].flatMap((route) => route.expertIds),
          );
          const expectedVictim = [...resident.entries()]
            .filter(([id]) => !protectedIds.has(id))
            .sort((left, right) => (
              left[1] - right[1] || left[0].localeCompare(right[0])
            ))[0]?.[0];
          if (event.expertId !== expectedVictim) {
            replayFail(
              `eviction chose ${event.expertId}; expected LRU victim ${expectedVictim ?? "none"}`,
            );
          }
          if (!resident.delete(event.expertId)) {
            replayFail(
              `cannot evict non-resident ${event.tier} expert ${event.expertId}`,
            );
          }
          if (experts.get(event.expertId)?.bytes !== event.bytes) {
            replayFail(`eviction bytes mismatch for ${event.expertId}`);
          }
          metrics.evictions++;
          break;
        }
        case "load_start": {
          requireMonotonicTime(event.load.startedAtNs, currentTimeNs);
          currentTimeNs = event.load.startedAtNs;
          if (pending.has(event.load.loadId)) {
            replayFail(`duplicate load id ${event.load.loadId}`);
          }
          const pendingKey = targetKey(
            event.load.expertId,
            event.load.targetTier,
          );
          if (pendingByTarget.has(pendingKey)) {
            replayFail(
              `duplicate pending ${event.load.targetTier} load for ${event.load.expertId}`,
            );
          }
          const expert = experts.get(event.load.expertId);
          if (!expert || expert.bytes !== event.load.bytes) {
            replayFail(`invalid load expert or bytes for ${event.load.loadId}`);
          }
          const expectedLatency = event.load.sourceTier === "warm"
            ? config.warmToHotLatencyNs
            : event.load.targetTier === "hot"
              ? config.coldToHotLatencyNs
              : config.coldToWarmLatencyNs;
          if (
            event.load.completesAtNs
            !== event.load.startedAtNs + expectedLatency
          ) {
            replayFail(`load latency mismatch for ${event.load.loadId}`);
          }
          if (event.load.sourceTier === "warm" && !warm.has(event.load.expertId)) {
            replayFail(`load ${event.load.loadId} lacks warm source`);
          }
          const expectedSource =
            event.load.targetTier === "hot" && warm.has(event.load.expertId)
              ? "warm"
              : "cold";
          if (event.load.sourceTier !== expectedSource) {
            replayFail(`load source mismatch for ${event.load.loadId}`);
          }
          pending.set(event.load.loadId, structuredClone(event.load));
          pendingByTarget.set(pendingKey, event.load.loadId);
          if (event.load.targetTier === "hot") {
            hotReservedBytes += event.load.bytes;
            if (
              residentBytes(hot, experts) + hotReservedBytes
              > config.hotCapacityBytes
            ) {
              replayFail("hot cache capacity exceeded");
            }
            metrics.highWaterHotBytes = Math.max(
              metrics.highWaterHotBytes,
              residentBytes(hot, experts) + hotReservedBytes,
            );
          } else {
            warmReservedBytes += event.load.bytes;
            if (
              residentBytes(warm, experts) + warmReservedBytes
              > config.warmCapacityBytes
            ) {
              replayFail("warm cache capacity exceeded");
            }
            metrics.highWaterWarmBytes = Math.max(
              metrics.highWaterWarmBytes,
              residentBytes(warm, experts) + warmReservedBytes,
            );
          }
          if (event.load.kind === "demand") {
            metrics.demandLoads++;
          } else {
            metrics.prefetchLoads++;
          }
          metrics.bytesMoved += event.load.bytes;
          break;
        }
        case "load_complete": {
          requireMonotonicTime(event.completedAtNs, currentTimeNs);
          currentTimeNs = event.completedAtNs;
          const load = pending.get(event.loadId);
          if (
            !load
            || load.expertId !== event.expertId
            || load.targetTier !== event.targetTier
            || load.completesAtNs !== event.completedAtNs
          ) {
            replayFail(`completion mismatch for ${event.loadId}`);
          }
          pending.delete(event.loadId);
          pendingByTarget.delete(targetKey(load.expertId, load.targetTier));
          if (load.targetTier === "hot") {
            hotReservedBytes -= load.bytes;
            hot.set(load.expertId, ++accessClock);
          } else {
            warmReservedBytes -= load.bytes;
            warm.set(load.expertId, ++accessClock);
          }
          break;
        }
        case "access": {
          requireMonotonicTime(event.readyAtNs, currentTimeNs);
          currentTimeNs = event.readyAtNs;
          const route = outstandingRoutes.get(event.requestId);
          if (
            !route
            || route.requestedAtNs !== event.requestedAtNs
            || event.readyAtNs < event.requestedAtNs
            || event.stallNs !== event.readyAtNs - event.requestedAtNs
            || event.expertIds.length !== event.sourceTiers.length
          ) {
            replayFail(`access timing mismatch for ${event.requestId}`);
          }
          assertArrayEqual(
            route.expertIds,
            event.expertIds,
            `access experts for ${event.requestId}`,
          );
          assertArrayEqual(
            route.sourceTiers,
            event.sourceTiers,
            `access source tiers for ${event.requestId}`,
          );
          for (let i = 0; i < event.expertIds.length; i++) {
            const id = event.expertIds[i];
            if (!hot.has(id)) {
              replayFail(`access ${event.requestId} uses non-hot expert ${id}`);
            }
            hot.set(id, ++accessClock);
            if (event.sourceTiers[i] === "hot") {
              metrics.hotHits++;
            } else if (event.sourceTiers[i] === "warm") {
              metrics.warmMisses++;
            } else {
              metrics.coldMisses++;
            }
          }
          metrics.stallNs += event.stallNs;
          outstandingRoutes.delete(event.requestId);
          break;
        }
      }
    } catch (error) {
      if (error instanceof ExpertCacheReplayError) {
        throw error;
      }
      throw new ExpertCacheReplayError(
        `event ${index}: ${errorMessage(error)}`,
      );
    }
  }

  return {
    appliedEvents: trace.length,
    snapshot: buildSnapshot({
      currentTimeNs,
      config,
      experts,
      hot,
      warm,
      hotReservedBytes,
      warmReservedBytes,
      pending,
      metrics,
    }),
  };
}

export function simulateExpertCacheWorkload(
  config: ExpertCacheWorkloadConfig,
): ExpertCacheWorkloadResult {
  assertNonNegativeSafeInteger(config.tokenCount, "workload token count");
  assertPositiveSafeInteger(config.topK, "workload topK");
  assertNonNegativeSafeInteger(
    config.tokenIntervalNs,
    "workload token interval",
  );
  const startAtNs = config.startAtNs ?? 0;
  assertNonNegativeSafeInteger(startAtNs, "workload start");
  const cache = new ExpertCacheSimulator(config.cache);

  let firstTokenAtNs = startAtNs;
  if (config.initialPrefetch) {
    assertNonNegativeSafeInteger(
      config.initialPrefetch.leadTimeNs,
      "initial prefetch lead time",
    );
    cache.prefetch(
      config.initialPrefetch.expertIds,
      config.initialPrefetch.targetTier,
      startAtNs,
    );
    firstTokenAtNs = checkedAdd(
      startAtNs,
      config.initialPrefetch.leadTimeNs,
      "first token timestamp",
    );
    cache.advanceTo(firstTokenAtNs);
  }

  const routes: ExpertRouteResult[] = [];
  for (let tokenIndex = 0; tokenIndex < config.tokenCount; tokenIndex++) {
    const plannedAtNs = checkedAdd(
      firstTokenAtNs,
      checkedMultiply(
        tokenIndex,
        config.tokenIntervalNs,
        "token schedule offset",
      ),
      "planned token timestamp",
    );
    const dispatchAtNs = Math.max(cache.snapshot().currentTimeNs, plannedAtNs);
    routes.push(cache.processToken({
      tokenIndex,
      topK: config.topK,
      atNs: dispatchAtNs,
    }));
  }

  const trace = cache.trace();
  const snapshot = cache.snapshot();
  const replayed = replayExpertCacheTrace(trace).snapshot;
  if (JSON.stringify(replayed) !== JSON.stringify(snapshot)) {
    throw new ExpertCacheProtocolError(
      "expert cache replay diverged from workload state",
    );
  }
  return { routes, snapshot, trace };
}

function buildSnapshot(input: {
  readonly currentTimeNs: number;
  readonly config: ExpertCacheConfig;
  readonly experts: ReadonlyMap<string, ExpertSpec>;
  readonly hot: ReadonlyMap<string, number>;
  readonly warm: ReadonlyMap<string, number>;
  readonly hotReservedBytes: number;
  readonly warmReservedBytes: number;
  readonly pending: ReadonlyMap<string, ExpertPendingLoadSnapshot>;
  readonly metrics: MutableMetrics;
}): ExpertCacheSnapshot {
  return {
    currentTimeNs: input.currentTimeNs,
    hotCapacityBytes: input.config.hotCapacityBytes,
    warmCapacityBytes: input.config.warmCapacityBytes,
    hotResidentBytes: residentBytes(input.hot, input.experts),
    warmResidentBytes: residentBytes(input.warm, input.experts),
    hotReservedBytes: input.hotReservedBytes,
    warmReservedBytes: input.warmReservedBytes,
    hotExpertIds: sortedResidentIds(input.hot),
    warmExpertIds: sortedResidentIds(input.warm),
    pendingLoads: [...input.pending.values()]
      .sort((left, right) => (
        left.completesAtNs - right.completesAtNs
        || left.loadId.localeCompare(right.loadId)
      ))
      .map((load) => structuredClone(load)),
    metrics: {
      ...input.metrics,
      hotHitRate: input.metrics.routedExperts === 0
        ? 0
        : input.metrics.hotHits / input.metrics.routedExperts,
    },
  };
}

function chooseVictims(
  resident: ReadonlyMap<string, number>,
  experts: ReadonlyMap<string, ExpertSpec>,
  usedAndReservedBytes: number,
  incomingBytes: number,
  capacityBytes: number,
  protectedIds: ReadonlySet<string>,
): string[] {
  if (incomingBytes > capacityBytes) {
    throw new ExpertCacheProtocolError(
      `expert requires ${incomingBytes} bytes but cache capacity is ${capacityBytes}`,
    );
  }
  let required = usedAndReservedBytes + incomingBytes - capacityBytes;
  if (required <= 0) {
    return [];
  }
  const candidates = [...resident.entries()]
    .filter(([id]) => !protectedIds.has(id))
    .sort((left, right) => left[1] - right[1] || left[0].localeCompare(right[0]));
  const victims: string[] = [];
  for (const [id] of candidates) {
    victims.push(id);
    required -= experts.get(id)?.bytes ?? 0;
    if (required <= 0) {
      return victims;
    }
  }
  throw new ExpertCacheProtocolError(
    `cache capacity cannot admit ${incomingBytes} bytes without evicting a protected expert or pending reservation`,
  );
}

function selectWithoutReplacement(
  experts: readonly ExpertSpec[],
  topK: number,
  rng: DeterministicRng,
): ExpertSpec[] {
  const candidates = [...experts].sort((left, right) =>
    left.id.localeCompare(right.id)
  );
  const selected: ExpertSpec[] = [];
  for (let count = 0; count < topK; count++) {
    const total = candidates.reduce(
      (sum, expert) => sum + (expert.routingWeight ?? 1),
      0,
    );
    const draw = rng.nextFloat() * total;
    let cumulative = 0;
    let selectedIndex = candidates.length - 1;
    for (let index = 0; index < candidates.length; index++) {
      cumulative += candidates[index].routingWeight ?? 1;
      if (draw < cumulative) {
        selectedIndex = index;
        break;
      }
    }
    selected.push(candidates.splice(selectedIndex, 1)[0]);
  }
  return selected;
}

function validateConfig(config: ExpertCacheConfig): void {
  assertPositiveSafeInteger(config.hotCapacityBytes, "hot capacity");
  assertNonNegativeSafeInteger(config.warmCapacityBytes, "warm capacity");
  assertNonNegativeSafeInteger(config.warmToHotLatencyNs, "warm-to-hot latency");
  assertNonNegativeSafeInteger(config.coldToHotLatencyNs, "cold-to-hot latency");
  assertNonNegativeSafeInteger(
    config.coldToWarmLatencyNs,
    "cold-to-warm latency",
  );
  assertNonNegativeSafeInteger(config.routingSeed, "routing seed");
  if (config.experts.length === 0) {
    throw new ExpertCacheProtocolError("at least one expert is required");
  }
  const ids = new Set<string>();
  for (const expert of config.experts) {
    if (expert.id.length === 0 || ids.has(expert.id)) {
      throw new ExpertCacheProtocolError(
        `expert id ${JSON.stringify(expert.id)} must be non-empty and unique`,
      );
    }
    ids.add(expert.id);
    assertPositiveSafeInteger(expert.bytes, `expert ${expert.id} bytes`);
    if (
      expert.routingWeight !== undefined
      && (!Number.isFinite(expert.routingWeight) || expert.routingWeight <= 0)
    ) {
      throw new ExpertCacheProtocolError(
        `expert ${expert.id} routing weight must be positive`,
      );
    }
  }
  for (const id of [
    ...(config.initialHotExpertIds ?? []),
    ...(config.initialWarmExpertIds ?? []),
  ]) {
    if (!ids.has(id)) {
      throw new ExpertCacheProtocolError(
        `initial cache references unknown expert ${id}`,
      );
    }
  }
  for (const [tier, initialIds, capacity] of [
    ["hot", config.initialHotExpertIds ?? [], config.hotCapacityBytes],
    ["warm", config.initialWarmExpertIds ?? [], config.warmCapacityBytes],
  ] as const) {
    if (unique(initialIds).length !== initialIds.length) {
      throw new ExpertCacheProtocolError(
        `initial ${tier} expert ids must be unique`,
      );
    }
    const bytes = initialIds.reduce((sum, id) => (
      checkedAdd(
        sum,
        config.experts.find((expert) => expert.id === id)?.bytes ?? 0,
        `initial ${tier} bytes`,
      )
    ), 0);
    if (bytes > capacity) {
      throw new ExpertCacheProtocolError(
        `initial ${tier} experts require ${bytes} bytes but capacity is ${capacity}`,
      );
    }
  }
}

function residentBytes(
  resident: ReadonlyMap<string, number>,
  experts: ReadonlyMap<string, ExpertSpec>,
): number {
  let bytes = 0;
  for (const id of resident.keys()) {
    bytes += experts.get(id)?.bytes ?? 0;
  }
  return bytes;
}

function sortedResidentIds(resident: ReadonlyMap<string, number>): string[] {
  return [...resident.entries()]
    .sort((left, right) => left[1] - right[1] || left[0].localeCompare(right[0]))
    .map(([id]) => id);
}

function cloneConfig(config: ExpertCacheConfig): ExpertCacheConfig {
  return structuredClone(config);
}

function targetKey(expertId: string, tier: ExpertLoadTarget): string {
  return `${tier}\u0000${expertId}`;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function emptyMetrics(): MutableMetrics {
  return {
    routes: 0,
    routedExperts: 0,
    hotHits: 0,
    warmMisses: 0,
    coldMisses: 0,
    demandLoads: 0,
    prefetchLoads: 0,
    evictions: 0,
    bytesMoved: 0,
    stallNs: 0,
    highWaterHotBytes: 0,
    highWaterWarmBytes: 0,
  };
}

function assertArrayEqual(
  expected: readonly string[],
  actual: readonly string[],
  label: string,
): void {
  if (
    expected.length !== actual.length
    || expected.some((value, index) => value !== actual[index])
  ) {
    replayFail(`${label} mismatch`);
  }
}

function requireMonotonicTime(timestampNs: number, currentTimeNs: number): void {
  assertNonNegativeSafeInteger(timestampNs, "trace timestamp");
  if (timestampNs < currentTimeNs) {
    replayFail(`trace time moved backward from ${currentTimeNs} to ${timestampNs}`);
  }
}

function replayFail(message: string): never {
  throw new ExpertCacheReplayError(message);
}

function assertNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new ExpertCacheProtocolError(
      `${label} must be a non-negative safe integer; got ${value}`,
    );
  }
}

function assertPositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new ExpertCacheProtocolError(
      `${label} must be a positive safe integer; got ${value}`,
    );
  }
}

function checkedAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    throw new ExpertCacheProtocolError(
      `${label} exceeds Number.MAX_SAFE_INTEGER`,
    );
  }
  return result;
}

function checkedMultiply(left: number, right: number, label: string): number {
  const result = left * right;
  if (!Number.isSafeInteger(result)) {
    throw new ExpertCacheProtocolError(
      `${label} exceeds Number.MAX_SAFE_INTEGER`,
    );
  }
  return result;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

class DeterministicRng {
  private state: number;

  constructor(seed: number) {
    this.state = (seed >>> 0) || 0x9e3779b9;
  }

  nextFloat(): number {
    let value = this.state;
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    this.state = value >>> 0;
    return this.state / 0x1_0000_0000;
  }

  snapshot(): number {
    return this.state;
  }

  restore(state: number): void {
    this.state = state;
  }
}
