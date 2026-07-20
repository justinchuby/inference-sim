import { DiscreteEventSimulator } from "./event-loop.js";

export const EXPERT_CACHE_CONTRACT_REVISION = 5;

export type ExpertCacheTier = "hot" | "warm" | "cold";
export type ExpertLoadTarget = "hot" | "warm";
export type ExpertLoadKind = "demand" | "prefetch";

export interface ExpertSpec {
  readonly id: string;
  readonly bytes: number;
  readonly routingWeight?: number;
}

export interface AdaptiveExpertPrefetchPolicy {
  readonly targetTier: "warm";
  readonly minObservations: number;
  readonly intervalTokens: number;
  readonly maxExpertsPerDecision: number;
}

export interface ExpertCacheTierPartition {
  readonly id: string;
  readonly expertIds: readonly string[];
  readonly capacityBytes: number;
}

export interface ExpertCacheConfig {
  readonly experts: readonly ExpertSpec[];
  readonly hotCapacityBytes: number;
  readonly warmCapacityBytes: number;
  readonly hotPartitions?: readonly ExpertCacheTierPartition[];
  readonly warmPartitions?: readonly ExpertCacheTierPartition[];
  readonly warmToHotLatencyNs: number;
  readonly coldToHotLatencyNs: number;
  readonly coldToWarmLatencyNs: number;
  readonly routingSeed: number;
  readonly initialHotExpertIds?: readonly string[];
  readonly initialWarmExpertIds?: readonly string[];
  readonly adaptivePrefetch?: AdaptiveExpertPrefetchPolicy;
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

export interface ExpertPendingRoute {
  readonly requestId: string;
  readonly tokenIndex: number;
  readonly expertIds: readonly string[];
  readonly requestedAtNs: number;
  readonly sourceTiers: readonly ExpertCacheTier[];
  readonly requiredLoadIds: readonly string[];
  readonly newDemandLoadIds: readonly string[];
}

export interface ExpertCacheMetrics {
  readonly routes: number;
  readonly routedExperts: number;
  readonly hotHits: number;
  readonly warmMisses: number;
  readonly coldMisses: number;
  readonly demandLoads: number;
  readonly prefetchLoads: number;
  readonly adaptivePrefetchDecisions: number;
  readonly adaptivePrefetchSelections: number;
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
  readonly hotPartitions: readonly ExpertCachePartitionSnapshot[];
  readonly warmPartitions: readonly ExpertCachePartitionSnapshot[];
  readonly pendingLoads: readonly ExpertPendingLoadSnapshot[];
  readonly metrics: ExpertCacheMetrics;
}

export interface ExpertCachePartitionSnapshot {
  readonly id: string;
  readonly capacityBytes: number;
  readonly residentBytes: number;
  readonly reservedBytes: number;
  readonly expertIds: readonly string[];
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
      readonly trigger: "manual" | "adaptive";
    }
  | {
      readonly kind: "prefetch_decision";
      readonly tokenIndex: number;
      readonly atNs: number;
      readonly observedRoutes: number;
      readonly expertIds: readonly string[];
    }
  | {
      readonly kind: "evict";
      readonly atNs: number;
      readonly tier: ExpertLoadTarget;
      readonly partitionId: string;
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
      readonly kind: "load_retime";
      readonly loadId: string;
      readonly atNs: number;
      readonly priorCompletesAtNs: number;
      readonly completesAtNs: number;
      readonly physicalCompletesAtNs?: number;
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
  adaptivePrefetchDecisions: number;
  adaptivePrefetchSelections: number;
  evictions: number;
  bytesMoved: number;
  stallNs: number;
  highWaterHotBytes: number;
  highWaterWarmBytes: number;
}

interface CompletionEvent {
  readonly loadId: string;
}

interface ExpertRouteHistory {
  count: number;
  lastTokenIndex: number;
}

interface ExpertCacheTierLayout {
  readonly partitions: readonly ExpertCacheTierPartition[];
  readonly partitionByExpertId: ReadonlyMap<string, string>;
  readonly capacityByPartitionId: ReadonlyMap<string, number>;
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
  private readonly hotLayout: ExpertCacheTierLayout;
  private readonly warmLayout: ExpertCacheTierLayout;
  private readonly hot = new Map<string, number>();
  private readonly warm = new Map<string, number>();
  private readonly pending = new Map<string, ExpertPendingLoadSnapshot>();
  private readonly pendingByTarget = new Map<string, string>();
  private readonly routeHistory = new Map<string, ExpertRouteHistory>();
  private readonly events: ExpertCacheTraceEvent[] = [];
  private readonly eventLoop = new DiscreteEventSimulator<CompletionEvent>();
  private readonly completionEventByLoad = new Map<string, number>();
  private readonly rng: DeterministicRng;
  private readonly metrics: MutableMetrics = emptyMetrics();
  private currentTimeNs = 0;
  private hotReservedBytes = 0;
  private warmReservedBytes = 0;
  private accessClock = 0;
  private nextLoadId = 1;
  private nextRequestId = 1;
  private nextSourceSequence = 0;
  private observedRoutes = 0;
  private activeRoute?: ExpertPendingRoute;

  constructor(config: ExpertCacheConfig) {
    validateConfig(config);
    this.config = cloneConfig(config);
    this.sourceId = config.sourceId ?? "expert-cache";
    this.experts = new Map(config.experts.map((expert) => [expert.id, expert]));
    this.hotLayout = buildTierLayout(this.config, "hot");
    this.warmLayout = buildTierLayout(this.config, "warm");
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
    const route = this.beginTokenRoute(request);
    return this.completeTokenRoute(route.requestId);
  }

  beginTokenRoute(request: ExpertRouteRequest): ExpertPendingRoute {
    if (this.activeRoute !== undefined) {
      throw new ExpertCacheProtocolError(
        `route ${this.activeRoute.requestId} must complete before another route begins`,
      );
    }
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
    const expertIds = selected.map((expert) => expert.id);
    const sourceTiers = expertIds.map((id) => this.tierFor(id));
    for (const partitionId of unique(expertIds.map((id) => (
      requirePartitionId(this.hotLayout, id, "hot")
    )))) {
      const selectedInPartition = expertIds.filter((id) => (
        requirePartitionId(this.hotLayout, id, "hot") === partitionId
      ));
      const newDemandBytes = selectedInPartition.reduce((sum, id) => (
        this.hot.has(id) || this.pendingTarget(id, "hot")
          ? sum
          : checkedAdd(sum, this.requireExpert(id).bytes, "new demand bytes")
      ), 0);
      const protectedHotBytes = selectedInPartition.reduce((sum, id) => (
        this.hot.has(id)
          ? checkedAdd(sum, this.requireExpert(id).bytes, "protected hot bytes")
          : sum
      ), 0);
      const capacity = requirePartitionCapacity(
        this.hotLayout,
        partitionId,
        "hot",
      );
      const reserved = reservedBytesInPartition(
        this.pending,
        this.hotLayout,
        "hot",
        partitionId,
      );
      if (protectedHotBytes + reserved + newDemandBytes > capacity) {
        this.rng.restore(rngCheckpoint);
        throw new ExpertCacheProtocolError(
          `pending hot reservations leave insufficient capacity in partition ${partitionId} for the routed working set`,
        );
      }
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
    const requiredLoadIds: string[] = [];
    const newDemandLoadIds: string[] = [];
    for (let index = 0; index < expertIds.length; index++) {
      const id = expertIds[index];
      const sourceTier = sourceTiers[index];
      if (sourceTier === "hot") {
        continue;
      }
      const existing = this.pendingTarget(id, "hot");
      const load = existing
        ?? this.startLoad(id, "hot", "demand", request.atNs, protectedIds);
      requiredLoadIds.push(load.loadId);
      if (existing === undefined) {
        newDemandLoadIds.push(load.loadId);
      }
    }

    const route: ExpertPendingRoute = {
      requestId,
      tokenIndex: request.tokenIndex,
      expertIds,
      requestedAtNs: request.atNs,
      sourceTiers,
      requiredLoadIds,
      newDemandLoadIds,
    };
    this.activeRoute = route;
    return structuredClone(route);
  }

  completeTokenRoute(requestId: string): ExpertRouteResult {
    const route = this.activeRoute;
    if (route === undefined || route.requestId !== requestId) {
      throw new ExpertCacheProtocolError(
        `cannot complete inactive route ${requestId}`,
      );
    }
    let readyAtNs = Math.max(route.requestedAtNs, this.currentTimeNs);
    for (const loadId of route.requiredLoadIds) {
      const load = this.pending.get(loadId);
      if (load !== undefined) {
        readyAtNs = Math.max(readyAtNs, load.completesAtNs);
      }
    }
    this.advanceTo(readyAtNs);
    for (const id of route.expertIds) {
      if (!this.hot.has(id)) {
        throw new ExpertCacheProtocolError(
          `expert ${id} was not hot when request ${route.requestId} became ready`,
        );
      }
      this.touch(this.hot, id);
    }
    for (const sourceTier of route.sourceTiers) {
      if (sourceTier === "hot") {
        this.metrics.hotHits++;
      } else if (sourceTier === "warm") {
        this.metrics.warmMisses++;
      } else {
        this.metrics.coldMisses++;
      }
    }
    const stallNs = readyAtNs - route.requestedAtNs;
    this.metrics.stallNs = checkedAdd(
      this.metrics.stallNs,
      stallNs,
      "expert cache stall",
    );
    this.commitEvent({
      kind: "access",
      requestId: route.requestId,
      requestedAtNs: route.requestedAtNs,
      readyAtNs,
      expertIds: route.expertIds,
      sourceTiers: route.sourceTiers,
      stallNs,
    });
    this.observeRoute(route.tokenIndex, route.expertIds);
    this.activeRoute = undefined;
    this.runAdaptivePrefetch(route.tokenIndex, readyAtNs);
    return {
      requestId: route.requestId,
      expertIds: route.expertIds,
      requestedAtNs: route.requestedAtNs,
      readyAtNs,
      stallNs,
      sourceTiers: route.sourceTiers,
    };
  }

  prefetch(
    expertIds: readonly string[],
    targetTier: ExpertLoadTarget,
    atNs: number,
  ): readonly string[] {
    if (this.activeRoute !== undefined) {
      throw new ExpertCacheProtocolError(
        `route ${this.activeRoute.requestId} must complete before prefetch`,
      );
    }
    return this.prefetchWithTrigger(expertIds, targetTier, atNs, "manual");
  }

  private prefetchWithTrigger(
    expertIds: readonly string[],
    targetTier: ExpertLoadTarget,
    atNs: number,
    trigger: "manual" | "adaptive",
  ): readonly string[] {
    this.advanceTo(atNs);
    const uniqueIds = unique(expertIds);
    for (const id of uniqueIds) {
      this.requireExpert(id);
    }
    const resident = targetTier === "hot" ? this.hot : this.warm;
    const layout = targetTier === "hot" ? this.hotLayout : this.warmLayout;
    for (const partitionId of unique(uniqueIds.map((id) => (
      requirePartitionId(layout, id, targetTier)
    )))) {
      const incomingBytes = uniqueIds.reduce((sum, id) => (
        requirePartitionId(layout, id, targetTier) !== partitionId
        || resident.has(id)
        || this.pendingTarget(id, targetTier)
          ? sum
          : checkedAdd(sum, this.requireExpert(id).bytes, "prefetch bytes")
      ), 0);
      const reserved = reservedBytesInPartition(
        this.pending,
        layout,
        targetTier,
        partitionId,
      );
      const capacity = requirePartitionCapacity(
        layout,
        partitionId,
        targetTier,
      );
      if (reserved + incomingBytes > capacity) {
        throw new ExpertCacheProtocolError(
          `pending reservations plus prefetch require ${reserved + incomingBytes} ${targetTier} bytes in partition ${partitionId} but capacity is ${capacity}`,
        );
      }
    }
    this.commitEvent({
      kind: "prefetch",
      atNs,
      targetTier,
      expertIds: uniqueIds,
      trigger,
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

  private observeRoute(
    tokenIndex: number,
    expertIds: readonly string[],
  ): void {
    this.observedRoutes++;
    for (const id of expertIds) {
      const history = this.routeHistory.get(id) ?? {
        count: 0,
        lastTokenIndex: -1,
      };
      history.count++;
      history.lastTokenIndex = tokenIndex;
      this.routeHistory.set(id, history);
    }
  }

  private runAdaptivePrefetch(tokenIndex: number, atNs: number): void {
    const policy = this.config.adaptivePrefetch;
    if (
      policy === undefined
      || (tokenIndex + 1) % policy.intervalTokens !== 0
    ) {
      return;
    }
    const expertIds = selectAdaptivePrefetchExperts({
      policy,
      experts: this.experts,
      history: this.routeHistory,
      warm: this.warm,
      pendingByTarget: this.pendingByTarget,
      pending: this.pending,
      warmLayout: this.warmLayout,
    });
    this.metrics.adaptivePrefetchDecisions++;
    this.metrics.adaptivePrefetchSelections += expertIds.length;
    this.commitEvent({
      kind: "prefetch_decision",
      tokenIndex,
      atNs,
      observedRoutes: this.observedRoutes,
      expertIds,
    });
    if (expertIds.length > 0) {
      this.prefetchWithTrigger(expertIds, "warm", atNs, "adaptive");
    }
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
      hotLayout: this.hotLayout,
      warmLayout: this.warmLayout,
      pending: this.pending,
      metrics: this.metrics,
    });
  }

  trace(): readonly ExpertCacheTraceEvent[] {
    return structuredClone(this.events);
  }

  traceLength(): number {
    return this.events.length;
  }

  traceFrom(sourceSequence: number): readonly ExpertCacheTraceEvent[] {
    assertNonNegativeSafeInteger(sourceSequence, "trace source sequence");
    if (sourceSequence > this.events.length) {
      throw new ExpertCacheProtocolError(
        `trace source sequence ${sourceSequence} exceeds ${this.events.length}`,
      );
    }
    return structuredClone(this.events.slice(sourceSequence));
  }

  retimePendingPrefetch(
    loadId: string,
    completesAtNs: number,
    physicalCompletesAtNs?: number,
  ): ExpertPendingLoadSnapshot {
    const load = this.pending.get(loadId);
    if (load?.kind !== "prefetch") {
      throw new ExpertCacheProtocolError(
        `cannot retime non-prefetch load ${loadId}`,
      );
    }
    return this.retimePendingLoad(
      loadId,
      completesAtNs,
      physicalCompletesAtNs,
    );
  }

  retimePendingLoad(
    loadId: string,
    completesAtNs: number,
    physicalCompletesAtNs?: number,
  ): ExpertPendingLoadSnapshot {
    assertNonNegativeSafeInteger(
      completesAtNs,
      "retimed load completion",
    );
    const load = this.pending.get(loadId);
    if (load === undefined) {
      throw new ExpertCacheProtocolError(
        `cannot retime unknown pending load ${loadId}`,
      );
    }
    if (
      completesAtNs < this.currentTimeNs
      || completesAtNs < load.startedAtNs
    ) {
      throw new ExpertCacheProtocolError(
        `retimed load ${loadId} completion ${completesAtNs}ns precedes current/load time`,
      );
    }
    if (
      physicalCompletesAtNs !== undefined
      && (
        !Number.isSafeInteger(physicalCompletesAtNs)
        || physicalCompletesAtNs < load.startedAtNs
        || physicalCompletesAtNs > completesAtNs
      )
    ) {
      throw new ExpertCacheProtocolError(
        `retimed load ${loadId} has invalid physical completion ${physicalCompletesAtNs}`,
      );
    }
    if (
      completesAtNs === load.completesAtNs
      && physicalCompletesAtNs === undefined
    ) {
      return structuredClone(load);
    }
    const updated = { ...load, completesAtNs };
    if (completesAtNs !== load.completesAtNs) {
      const eventId = this.completionEventByLoad.get(loadId);
      if (eventId === undefined || !this.eventLoop.cancel(eventId)) {
        throw new ExpertCacheProtocolError(
          `pending completion event disappeared for ${loadId}`,
        );
      }
      this.pending.set(loadId, updated);
      this.completionEventByLoad.set(
        loadId,
        this.eventLoop.scheduleAt(completesAtNs, { loadId }),
      );
    }
    this.commitEvent({
      kind: "load_retime",
      loadId,
      atNs: this.currentTimeNs,
      priorCompletesAtNs: load.completesAtNs,
      completesAtNs,
      ...(physicalCompletesAtNs === undefined
        ? {}
        : { physicalCompletesAtNs }),
    });
    return structuredClone(updated);
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
    const layout = targetTier === "hot" ? this.hotLayout : this.warmLayout;
    const partitionId = requirePartitionId(
      layout,
      expertId,
      targetTier,
    );
    const capacity = requirePartitionCapacity(
      layout,
      partitionId,
      targetTier,
    );
    const partitionResident = new Map(
      [...resident.entries()].filter(([id]) => (
        requirePartitionId(layout, id, targetTier) === partitionId
      )),
    );
    const reserved = reservedBytesInPartition(
      this.pending,
      layout,
      targetTier,
      partitionId,
    );
    const victims = chooseVictims(
      partitionResident,
      this.experts,
      checkedAdd(
        residentBytes(partitionResident, this.experts),
        reserved,
        "cache partition use",
      ),
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
        partitionId,
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
    this.completionEventByLoad.set(
      load.loadId,
      this.eventLoop.scheduleAt(completesAtNs, { loadId: load.loadId }),
    );
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
    this.completionEventByLoad.delete(loadId);
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
    const layout = tier === "hot" ? this.hotLayout : this.warmLayout;
    const partitionId = requirePartitionId(layout, expertId, tier);
    const capacity = requirePartitionCapacity(layout, partitionId, tier);
    if (resident.has(expertId)) {
      throw new ExpertCacheProtocolError(
        `initial ${tier} expert ${expertId} is duplicated`,
      );
    }
    const partitionResidentBytes = residentBytesInPartition(
      resident,
      this.experts,
      layout,
      partitionId,
      tier,
    );
    if (partitionResidentBytes + expert.bytes > capacity) {
      throw new ExpertCacheProtocolError(
        `initial ${tier} experts exceed ${capacity} bytes in partition ${partitionId}`,
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
  const hotLayout = buildTierLayout(config, "hot");
  const warmLayout = buildTierLayout(config, "warm");
  const hot = new Map<string, number>();
  const warm = new Map<string, number>();
  const pending = new Map<string, ExpertPendingLoadSnapshot>();
  const pendingByTarget = new Map<string, string>();
  const outstandingRoutes = new Map<string, {
    readonly tokenIndex: number;
    readonly requestedAtNs: number;
    readonly expertIds: readonly string[];
    readonly sourceTiers: readonly ExpertCacheTier[];
  }>();
  const metrics = emptyMetrics();
  const rng = new DeterministicRng(config.routingSeed);
  const routeHistory = new Map<string, ExpertRouteHistory>();
  let hotReservedBytes = 0;
  let warmReservedBytes = 0;
  let currentTimeNs = 0;
  let accessClock = 0;
  let sourceId = "";
  let observedRoutes = 0;
  let expectedDecision: {
    readonly tokenIndex: number;
    readonly atNs: number;
    readonly observedRoutes: number;
    readonly expertIds: readonly string[];
  } | undefined;
  let expectedAdaptivePrefetch: readonly string[] | undefined;

  for (let index = 0; index < trace.length; index++) {
    const event = trace[index];
    try {
      if (expectedDecision !== undefined && event.kind !== "prefetch_decision") {
        replayFail(
          `adaptive prefetch decision after token ${expectedDecision.tokenIndex} was omitted`,
        );
      }
      if (
        expectedAdaptivePrefetch !== undefined
        && event.kind !== "prefetch"
      ) {
        replayFail("adaptive prefetch request was omitted");
      }
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
          if (outstandingRoutes.size > 0) {
            replayFail("a route began before the prior route completed");
          }
          if (outstandingRoutes.has(event.requestId)) {
            replayFail(`duplicate request id ${event.requestId}`);
          }
          outstandingRoutes.set(event.requestId, {
            tokenIndex: event.tokenIndex,
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
          if (event.trigger !== "manual" && event.trigger !== "adaptive") {
            replayFail(`unsupported prefetch trigger ${String(event.trigger)}`);
          }
          if (unique(event.expertIds).length !== event.expertIds.length) {
            replayFail("prefetch expert ids must be unique");
          }
          for (const id of event.expertIds) {
            if (!experts.has(id)) {
              replayFail(`prefetch references unknown expert ${id}`);
            }
          }
          if (event.trigger === "adaptive") {
            if (expectedAdaptivePrefetch === undefined) {
              replayFail("unexpected adaptive prefetch request");
            }
            assertArrayEqual(
              expectedAdaptivePrefetch,
              event.expertIds,
              "adaptive prefetch experts",
            );
            expectedAdaptivePrefetch = undefined;
          } else if (expectedAdaptivePrefetch !== undefined) {
            replayFail("adaptive prefetch request was replaced by manual work");
          }
          break;
        }
        case "prefetch_decision": {
          if (expectedDecision === undefined) {
            replayFail("unexpected adaptive prefetch decision");
          }
          if (
            event.tokenIndex !== expectedDecision.tokenIndex
            || event.atNs !== expectedDecision.atNs
            || event.observedRoutes !== expectedDecision.observedRoutes
          ) {
            replayFail(
              `adaptive prefetch decision metadata mismatch for token ${event.tokenIndex}`,
            );
          }
          assertArrayEqual(
            expectedDecision.expertIds,
            event.expertIds,
            "adaptive prefetch decision experts",
          );
          metrics.adaptivePrefetchDecisions++;
          metrics.adaptivePrefetchSelections += event.expertIds.length;
          expectedDecision = undefined;
          if (event.expertIds.length > 0) {
            expectedAdaptivePrefetch = [...event.expertIds];
          }
          break;
        }
        case "evict": {
          requireMonotonicTime(event.atNs, currentTimeNs);
          currentTimeNs = event.atNs;
          const resident = event.tier === "hot" ? hot : warm;
          const layout = event.tier === "hot" ? hotLayout : warmLayout;
          const expertPartitionId = requirePartitionId(
            layout,
            event.expertId,
            event.tier,
          );
          if (event.partitionId !== expertPartitionId) {
            replayFail(
              `eviction expert ${event.expertId} belongs to ${expertPartitionId}, not ${event.partitionId}`,
            );
          }
          const protectedIds = new Set(
            [...outstandingRoutes.values()].flatMap((route) => route.expertIds),
          );
          const expectedVictim = [...resident.entries()]
            .filter(([id]) => (
              !protectedIds.has(id)
              && requirePartitionId(layout, id, event.tier)
                === event.partitionId
            ))
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
            const partitionId = requirePartitionId(
              hotLayout,
              event.load.expertId,
              "hot",
            );
            if (
              residentBytesInPartition(
                hot,
                experts,
                hotLayout,
                partitionId,
                "hot",
              ) + reservedBytesInPartition(
                pending,
                hotLayout,
                "hot",
                partitionId,
              ) > requirePartitionCapacity(hotLayout, partitionId, "hot")
            ) {
              replayFail(`hot cache partition ${partitionId} capacity exceeded`);
            }
            metrics.highWaterHotBytes = Math.max(
              metrics.highWaterHotBytes,
              residentBytes(hot, experts) + hotReservedBytes,
            );
          } else {
            warmReservedBytes += event.load.bytes;
            const partitionId = requirePartitionId(
              warmLayout,
              event.load.expertId,
              "warm",
            );
            if (
              residentBytesInPartition(
                warm,
                experts,
                warmLayout,
                partitionId,
                "warm",
              ) + reservedBytesInPartition(
                pending,
                warmLayout,
                "warm",
                partitionId,
              ) > requirePartitionCapacity(warmLayout, partitionId, "warm")
            ) {
              replayFail(`warm cache partition ${partitionId} capacity exceeded`);
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
        case "load_retime": {
          requireMonotonicTime(event.atNs, currentTimeNs);
          currentTimeNs = event.atNs;
          const load = pending.get(event.loadId);
          if (
            load === undefined
            || load.completesAtNs !== event.priorCompletesAtNs
            || event.completesAtNs < event.atNs
            || event.completesAtNs < load.startedAtNs
            || (
              event.physicalCompletesAtNs !== undefined
              && (
                !Number.isSafeInteger(event.physicalCompletesAtNs)
                || event.physicalCompletesAtNs < load.startedAtNs
                || event.physicalCompletesAtNs > event.completesAtNs
              )
            )
          ) {
            replayFail(`invalid load retime for ${event.loadId}`);
          }
          pending.set(event.loadId, {
            ...load,
            completesAtNs: event.completesAtNs,
          });
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
          observedRoutes++;
          for (const id of event.expertIds) {
            const history = routeHistory.get(id) ?? {
              count: 0,
              lastTokenIndex: -1,
            };
            history.count++;
            history.lastTokenIndex = route.tokenIndex;
            routeHistory.set(id, history);
          }
          const policy = config.adaptivePrefetch;
          if (
            policy !== undefined
            && (route.tokenIndex + 1) % policy.intervalTokens === 0
          ) {
            expectedDecision = {
              tokenIndex: route.tokenIndex,
              atNs: event.readyAtNs,
              observedRoutes,
              expertIds: selectAdaptivePrefetchExperts({
                policy,
                experts,
                history: routeHistory,
                warm,
                pendingByTarget,
                pending,
                warmLayout,
              }),
            };
          }
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

  if (expectedDecision !== undefined) {
    replayFail(
      `adaptive prefetch decision after token ${expectedDecision.tokenIndex} was omitted`,
    );
  }
  if (expectedAdaptivePrefetch !== undefined) {
    replayFail("adaptive prefetch request was omitted");
  }
  if (outstandingRoutes.size > 0) {
    replayFail(
      `route ${[...outstandingRoutes.keys()][0]} did not complete`,
    );
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
      hotLayout,
      warmLayout,
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
  readonly hotLayout: ExpertCacheTierLayout;
  readonly warmLayout: ExpertCacheTierLayout;
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
    hotPartitions: partitionSnapshots(
      input.hot,
      input.pending,
      input.experts,
      input.hotLayout,
      "hot",
    ),
    warmPartitions: partitionSnapshots(
      input.warm,
      input.pending,
      input.experts,
      input.warmLayout,
      "warm",
    ),
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

function selectAdaptivePrefetchExperts(input: {
  readonly policy: AdaptiveExpertPrefetchPolicy;
  readonly experts: ReadonlyMap<string, ExpertSpec>;
  readonly history: ReadonlyMap<string, ExpertRouteHistory>;
  readonly warm: ReadonlyMap<string, number>;
  readonly pendingByTarget: ReadonlyMap<string, string>;
  readonly pending: ReadonlyMap<string, ExpertPendingLoadSnapshot>;
  readonly warmLayout: ExpertCacheTierLayout;
}): string[] {
  const ranked = [...input.history.entries()]
    .filter(([id, history]) => (
      history.count >= input.policy.minObservations
      && !input.warm.has(id)
      && !input.pendingByTarget.has(targetKey(id, "warm"))
    ))
    .sort((left, right) => (
      right[1].count - left[1].count
      || right[1].lastTokenIndex - left[1].lastTokenIndex
      || left[0].localeCompare(right[0])
    ));
  const selected: string[] = [];
  const incomingBytesByPartition = new Map<string, number>();
  for (const [id] of ranked) {
    if (selected.length >= input.policy.maxExpertsPerDecision) {
      break;
    }
    const expertBytes = input.experts.get(id)?.bytes;
    const partitionId = requirePartitionId(input.warmLayout, id, "warm");
    const incomingBytes = incomingBytesByPartition.get(partitionId) ?? 0;
    const availableBytes = requirePartitionCapacity(
      input.warmLayout,
      partitionId,
      "warm",
    ) - reservedBytesInPartition(
      input.pending,
      input.warmLayout,
      "warm",
      partitionId,
    ) - incomingBytes;
    if (expertBytes === undefined || expertBytes > availableBytes) {
      continue;
    }
    selected.push(id);
    incomingBytesByPartition.set(partitionId, incomingBytes + expertBytes);
  }
  return selected;
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
  if (config.adaptivePrefetch !== undefined) {
    if (config.adaptivePrefetch.targetTier !== "warm") {
      throw new ExpertCacheProtocolError(
        "adaptive prefetch target tier must be warm",
      );
    }
    assertPositiveSafeInteger(
      config.adaptivePrefetch.minObservations,
      "adaptive prefetch minimum observations",
    );
    assertPositiveSafeInteger(
      config.adaptivePrefetch.intervalTokens,
      "adaptive prefetch interval tokens",
    );
    assertPositiveSafeInteger(
      config.adaptivePrefetch.maxExpertsPerDecision,
      "adaptive prefetch maximum experts per decision",
    );
    if (config.warmCapacityBytes === 0) {
      throw new ExpertCacheProtocolError(
        "adaptive prefetch requires positive warm capacity",
      );
    }
  }
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
  validateTierPartitions(config, "hot", ids);
  validateTierPartitions(config, "warm", ids);
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
  for (const [tier, initialIds] of [
    ["hot", config.initialHotExpertIds ?? []],
    ["warm", config.initialWarmExpertIds ?? []],
  ] as const) {
    if (unique(initialIds).length !== initialIds.length) {
      throw new ExpertCacheProtocolError(
        `initial ${tier} expert ids must be unique`,
      );
    }
    const layout = buildTierLayout(config, tier);
    for (const partition of layout.partitions) {
      const bytes = initialIds.reduce((sum, id) => (
        requirePartitionId(layout, id, tier) !== partition.id
          ? sum
          : checkedAdd(
              sum,
              config.experts.find((expert) => expert.id === id)?.bytes ?? 0,
              `initial ${tier} bytes`,
            )
      ), 0);
      if (bytes > partition.capacityBytes) {
        throw new ExpertCacheProtocolError(
          `initial ${tier} experts require ${bytes} bytes in partition ${partition.id} but capacity is ${partition.capacityBytes}`,
        );
      }
    }
  }
}

function validateTierPartitions(
  config: ExpertCacheConfig,
  tier: ExpertLoadTarget,
  expertIds: ReadonlySet<string>,
): void {
  const partitions = tier === "hot"
    ? config.hotPartitions
    : config.warmPartitions;
  if (partitions === undefined) {
    return;
  }
  if (partitions.length === 0) {
    throw new ExpertCacheProtocolError(
      `${tier} partitions must not be empty`,
    );
  }
  const partitionIds = new Set<string>();
  const assigned = new Set<string>();
  let capacityBytes = 0;
  for (const partition of partitions) {
    if (partition.id.length === 0 || partitionIds.has(partition.id)) {
      throw new ExpertCacheProtocolError(
        `${tier} partition id ${JSON.stringify(partition.id)} must be non-empty and unique`,
      );
    }
    partitionIds.add(partition.id);
    assertNonNegativeSafeInteger(
      partition.capacityBytes,
      `${tier} partition ${partition.id} capacity`,
    );
    capacityBytes = checkedAdd(
      capacityBytes,
      partition.capacityBytes,
      `${tier} partition capacity`,
    );
    if (partition.expertIds.length === 0) {
      throw new ExpertCacheProtocolError(
        `${tier} partition ${partition.id} must own at least one expert`,
      );
    }
    for (const expertId of partition.expertIds) {
      if (!expertIds.has(expertId)) {
        throw new ExpertCacheProtocolError(
          `${tier} partition ${partition.id} references unknown expert ${expertId}`,
        );
      }
      if (assigned.has(expertId)) {
        throw new ExpertCacheProtocolError(
          `${tier} expert ${expertId} is assigned to more than one partition`,
        );
      }
      assigned.add(expertId);
    }
  }
  const missing = [...expertIds].filter((expertId) => !assigned.has(expertId));
  if (missing.length > 0) {
    throw new ExpertCacheProtocolError(
      `${tier} partitions do not assign experts ${missing.join(", ")}`,
    );
  }
  const declaredCapacity = tier === "hot"
    ? config.hotCapacityBytes
    : config.warmCapacityBytes;
  if (capacityBytes !== declaredCapacity) {
    throw new ExpertCacheProtocolError(
      `${tier} partition capacity ${capacityBytes} does not equal aggregate ${declaredCapacity}`,
    );
  }
}

function buildTierLayout(
  config: ExpertCacheConfig,
  tier: ExpertLoadTarget,
): ExpertCacheTierLayout {
  const configured = tier === "hot"
    ? config.hotPartitions
    : config.warmPartitions;
  const partitions = configured === undefined
    ? [{
        id: "default",
        expertIds: config.experts.map((expert) => expert.id),
        capacityBytes: tier === "hot"
          ? config.hotCapacityBytes
          : config.warmCapacityBytes,
      }]
    : configured.map((partition) => ({
        ...partition,
        expertIds: [...partition.expertIds],
      }));
  return {
    partitions,
    partitionByExpertId: new Map(partitions.flatMap((partition) => (
      partition.expertIds.map((expertId) => [expertId, partition.id] as const)
    ))),
    capacityByPartitionId: new Map(partitions.map((partition) => (
      [partition.id, partition.capacityBytes] as const
    ))),
  };
}

function requirePartitionId(
  layout: ExpertCacheTierLayout,
  expertId: string,
  tier: ExpertLoadTarget,
): string {
  const partitionId = layout.partitionByExpertId.get(expertId);
  if (partitionId === undefined) {
    throw new ExpertCacheProtocolError(
      `${tier} cache has no partition for expert ${expertId}`,
    );
  }
  return partitionId;
}

function requirePartitionCapacity(
  layout: ExpertCacheTierLayout,
  partitionId: string,
  tier: ExpertLoadTarget,
): number {
  const capacity = layout.capacityByPartitionId.get(partitionId);
  if (capacity === undefined) {
    throw new ExpertCacheProtocolError(
      `${tier} cache has unknown partition ${partitionId}`,
    );
  }
  return capacity;
}

function reservedBytesInPartition(
  pending: ReadonlyMap<string, ExpertPendingLoadSnapshot>,
  layout: ExpertCacheTierLayout,
  tier: ExpertLoadTarget,
  partitionId: string,
): number {
  let bytes = 0;
  for (const load of pending.values()) {
    if (
      load.targetTier === tier
      && requirePartitionId(layout, load.expertId, tier) === partitionId
    ) {
      bytes = checkedAdd(bytes, load.bytes, `${tier} partition reserved bytes`);
    }
  }
  return bytes;
}

function residentBytesInPartition(
  resident: ReadonlyMap<string, number>,
  experts: ReadonlyMap<string, ExpertSpec>,
  layout: ExpertCacheTierLayout,
  partitionId: string,
  tier: ExpertLoadTarget,
): number {
  let bytes = 0;
  for (const id of resident.keys()) {
    if (requirePartitionId(layout, id, tier) === partitionId) {
      bytes = checkedAdd(
        bytes,
        experts.get(id)?.bytes ?? 0,
        `${tier} partition resident bytes`,
      );
    }
  }
  return bytes;
}

function partitionSnapshots(
  resident: ReadonlyMap<string, number>,
  pending: ReadonlyMap<string, ExpertPendingLoadSnapshot>,
  experts: ReadonlyMap<string, ExpertSpec>,
  layout: ExpertCacheTierLayout,
  tier: ExpertLoadTarget,
): ExpertCachePartitionSnapshot[] {
  return layout.partitions.map((partition) => ({
    id: partition.id,
    capacityBytes: partition.capacityBytes,
    residentBytes: residentBytesInPartition(
      resident,
      experts,
      layout,
      partition.id,
      tier,
    ),
    reservedBytes: reservedBytesInPartition(
      pending,
      layout,
      tier,
      partition.id,
    ),
    expertIds: sortedResidentIds(new Map(
      [...resident.entries()].filter(([id]) => (
        requirePartitionId(layout, id, tier) === partition.id
      )),
    )),
  }));
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
    adaptivePrefetchDecisions: 0,
    adaptivePrefetchSelections: 0,
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
