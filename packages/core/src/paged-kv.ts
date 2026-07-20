export const PAGED_KV_CONTRACT_REVISION = 1;

export interface PagedKvConfig {
  readonly sequenceId: string;
  readonly pageSizeTokens: number;
  readonly bytesPerToken: number;
  readonly capacityBytes: number;
  readonly initialTokenLength?: number;
  readonly sourceId?: string;
}

export interface PagedKvCheckpoint {
  readonly checkpointId: string;
  readonly generation: number;
  readonly baseTokenLength: number;
}

export interface PagedKvPageSnapshot {
  readonly physicalPageId: string;
  readonly validTokens: number;
  readonly capacityTokens: number;
  readonly bytes: number;
}

export interface PagedKvSnapshot {
  readonly sequenceId: string;
  readonly logicalTokenLength: number;
  readonly highWaterTokenLength: number;
  readonly generation: number;
  readonly pageSizeTokens: number;
  readonly pageBytes: number;
  readonly capacityBytes: number;
  readonly reservedBytes: number;
  readonly freeBytes: number;
  readonly livePages: readonly PagedKvPageSnapshot[];
  readonly activeCheckpoint?: PagedKvCheckpoint;
}

interface PagedKvTraceEnvelope {
  readonly contractRevision: typeof PAGED_KV_CONTRACT_REVISION;
  readonly sourceId: string;
  readonly sourceSequence: number;
}

export type PagedKvTraceEvent = PagedKvTraceEnvelope & (
  | {
      readonly kind: "initialize";
      readonly sequenceId: string;
      readonly pageSizeTokens: number;
      readonly bytesPerToken: number;
      readonly capacityBytes: number;
      readonly initialTokenLength: number;
      readonly allocatedPageIds: readonly string[];
    }
  | {
      readonly kind: "append";
      readonly baseTokenLength: number;
      readonly appendedTokens: number;
      readonly finalTokenLength: number;
      readonly allocatedPageIds: readonly string[];
    }
  | {
      readonly kind: "checkpoint";
      readonly checkpoint: PagedKvCheckpoint;
    }
  | {
      readonly kind: "restore";
      readonly checkpointId: string;
      readonly acceptedTokens: number;
      readonly finalTokenLength: number;
      readonly releasedPageIds: readonly string[];
      readonly generation: number;
    }
  | {
      readonly kind: "commit";
      readonly checkpointId: string;
      readonly generation: number;
    }
);

type WithoutTraceEnvelope<T> = T extends PagedKvTraceEnvelope
  ? Omit<T, keyof PagedKvTraceEnvelope>
  : never;

type PagedKvTracePayload = WithoutTraceEnvelope<PagedKvTraceEvent>;

export interface PagedKvReplayResult {
  readonly appliedEvents: number;
  readonly snapshot: PagedKvSnapshot;
}

interface MutableCheckpoint extends PagedKvCheckpoint {
  active: boolean;
}

export class PagedKvProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PagedKvProtocolError";
  }
}

export class PagedKvReplayError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PagedKvReplayError";
  }
}

export class PagedKvCacheSimulator {
  private readonly sequenceId: string;
  private readonly sourceId: string;
  private readonly pageSizeTokens: number;
  private readonly bytesPerToken: number;
  private readonly pageBytes: number;
  private readonly capacityBytes: number;
  private readonly maxPages: number;
  private readonly livePageIds: string[] = [];
  private readonly events: PagedKvTraceEvent[] = [];
  private logicalTokenLength: number;
  private highWaterTokenLength: number;
  private generation = 0;
  private nextPageId = 1;
  private nextCheckpointId = 1;
  private nextSourceSequence = 0;
  private activeCheckpoint?: MutableCheckpoint;

  constructor(config: PagedKvConfig) {
    validateConfig(config);
    this.sequenceId = config.sequenceId;
    this.sourceId = config.sourceId ?? `paged-kv:${config.sequenceId}`;
    this.pageSizeTokens = config.pageSizeTokens;
    this.bytesPerToken = config.bytesPerToken;
    this.pageBytes = checkedMultiply(
      config.pageSizeTokens,
      config.bytesPerToken,
      "page bytes",
    );
    this.capacityBytes = config.capacityBytes;
    this.maxPages = Math.floor(config.capacityBytes / this.pageBytes);
    this.logicalTokenLength = config.initialTokenLength ?? 0;
    this.highWaterTokenLength = this.logicalTokenLength;

    const initialPageCount = pagesForTokens(
      this.logicalTokenLength,
      this.pageSizeTokens,
    );
    if (initialPageCount > this.maxPages) {
      throw new PagedKvProtocolError(
        `initial KV requires ${initialPageCount} pages but capacity holds ${this.maxPages}`,
      );
    }
    const allocatedPageIds = this.allocatePages(initialPageCount);
    this.commitEvent({
      kind: "initialize",
      sequenceId: this.sequenceId,
      pageSizeTokens: this.pageSizeTokens,
      bytesPerToken: this.bytesPerToken,
      capacityBytes: this.capacityBytes,
      initialTokenLength: this.logicalTokenLength,
      allocatedPageIds,
    });
  }

  append(tokens: number): void {
    assertNonNegativeSafeInteger(tokens, "append tokens");
    const finalTokenLength = checkedAdd(
      this.logicalTokenLength,
      tokens,
      "logical token length",
    );
    const requiredPages = pagesForTokens(
      finalTokenLength,
      this.pageSizeTokens,
    );
    const pagesToAllocate = requiredPages - this.livePageIds.length;
    if (requiredPages > this.maxPages) {
      throw new PagedKvProtocolError(
        `append requires ${requiredPages} pages but capacity holds ${this.maxPages}`,
      );
    }

    const baseTokenLength = this.logicalTokenLength;
    const allocatedPageIds = this.allocatePages(pagesToAllocate);
    this.logicalTokenLength = finalTokenLength;
    this.highWaterTokenLength = Math.max(
      this.highWaterTokenLength,
      finalTokenLength,
    );
    this.commitEvent({
      kind: "append",
      baseTokenLength,
      appendedTokens: tokens,
      finalTokenLength,
      allocatedPageIds,
    });
  }

  checkpoint(): PagedKvCheckpoint {
    if (this.activeCheckpoint?.active) {
      throw new PagedKvProtocolError(
        `checkpoint ${this.activeCheckpoint.checkpointId} is still active`,
      );
    }
    const checkpoint: MutableCheckpoint = {
      checkpointId: `kv-checkpoint-${this.nextCheckpointId++}`,
      generation: this.generation,
      baseTokenLength: this.logicalTokenLength,
      active: true,
    };
    this.activeCheckpoint = checkpoint;
    const publicCheckpoint = checkpointSnapshot(checkpoint);
    this.commitEvent({ kind: "checkpoint", checkpoint: publicCheckpoint });
    return publicCheckpoint;
  }

  restore(checkpoint: PagedKvCheckpoint, acceptedTokens: number): void {
    assertNonNegativeSafeInteger(acceptedTokens, "accepted tokens");
    const active = this.requireActiveCheckpoint(checkpoint);
    const finalTokenLength = checkedAdd(
      active.baseTokenLength,
      acceptedTokens,
      "restored token length",
    );
    if (finalTokenLength > this.logicalTokenLength) {
      throw new PagedKvProtocolError(
        `restore target ${finalTokenLength} exceeds current length ${this.logicalTokenLength}`,
      );
    }
    const requiredPages = pagesForTokens(
      finalTokenLength,
      this.pageSizeTokens,
    );
    const releasedPageIds = this.livePageIds.splice(requiredPages);
    this.logicalTokenLength = finalTokenLength;
    active.active = false;
    this.generation++;
    delete this.activeCheckpoint;
    this.commitEvent({
      kind: "restore",
      checkpointId: checkpoint.checkpointId,
      acceptedTokens,
      finalTokenLength,
      releasedPageIds,
      generation: this.generation,
    });
  }

  commit(checkpoint: PagedKvCheckpoint): void {
    const active = this.requireActiveCheckpoint(checkpoint);
    active.active = false;
    this.generation++;
    delete this.activeCheckpoint;
    this.commitEvent({
      kind: "commit",
      checkpointId: checkpoint.checkpointId,
      generation: this.generation,
    });
  }

  snapshot(): PagedKvSnapshot {
    return buildSnapshot({
      sequenceId: this.sequenceId,
      logicalTokenLength: this.logicalTokenLength,
      highWaterTokenLength: this.highWaterTokenLength,
      generation: this.generation,
      pageSizeTokens: this.pageSizeTokens,
      pageBytes: this.pageBytes,
      capacityBytes: this.capacityBytes,
      livePageIds: this.livePageIds,
      activeCheckpoint: this.activeCheckpoint?.active
        ? checkpointSnapshot(this.activeCheckpoint)
        : undefined,
    });
  }

  trace(): readonly PagedKvTraceEvent[] {
    return this.events.map(cloneEvent);
  }

  assertInvariants(): void {
    const expectedPages = pagesForTokens(
      this.logicalTokenLength,
      this.pageSizeTokens,
    );
    if (expectedPages !== this.livePageIds.length) {
      throw new PagedKvProtocolError(
        `logical length requires ${expectedPages} pages, found ${this.livePageIds.length}`,
      );
    }
    if (
      this.livePageIds.length > this.maxPages
      || new Set(this.livePageIds).size !== this.livePageIds.length
    ) {
      throw new PagedKvProtocolError("live page capacity/identity invariant failed");
    }
    if (this.logicalTokenLength > this.highWaterTokenLength) {
      throw new PagedKvProtocolError(
        "logical token length exceeds high-water length",
      );
    }
    const reservedBytes = checkedMultiply(
      this.livePageIds.length,
      this.pageBytes,
      "reserved KV bytes",
    );
    if (reservedBytes > this.capacityBytes) {
      throw new PagedKvProtocolError("KV byte ledger exceeds capacity");
    }
    if (
      this.activeCheckpoint?.active
      && this.activeCheckpoint.generation !== this.generation
    ) {
      throw new PagedKvProtocolError("active checkpoint has stale generation");
    }
  }

  private allocatePages(count: number): string[] {
    const result: string[] = [];
    for (let index = 0; index < count; index++) {
      const pageId = `${this.sequenceId}:kv-page-${this.nextPageId++}`;
      this.livePageIds.push(pageId);
      result.push(pageId);
    }
    return result;
  }

  private requireActiveCheckpoint(
    checkpoint: PagedKvCheckpoint,
  ): MutableCheckpoint {
    const active = this.activeCheckpoint;
    if (
      !active?.active
      || checkpoint.checkpointId !== active.checkpointId
      || checkpoint.generation !== active.generation
      || checkpoint.baseTokenLength !== active.baseTokenLength
      || active.generation !== this.generation
    ) {
      throw new PagedKvProtocolError(
        `checkpoint ${checkpoint.checkpointId} is unknown, stale, or already consumed`,
      );
    }
    return active;
  }

  private commitEvent(
    payload: PagedKvTracePayload,
  ): void {
    this.assertInvariants();
    this.events.push({
      contractRevision: PAGED_KV_CONTRACT_REVISION,
      sourceId: this.sourceId,
      sourceSequence: this.nextSourceSequence++,
      ...payload,
    } as PagedKvTraceEvent);
  }
}

export function replayPagedKvTrace(
  config: PagedKvConfig,
  events: readonly PagedKvTraceEvent[],
): PagedKvReplayResult {
  validateConfig(config);
  const sourceId = config.sourceId ?? `paged-kv:${config.sequenceId}`;
  const pageBytes = checkedReplayMultiply(
    config.pageSizeTokens,
    config.bytesPerToken,
    "page bytes",
  );
  const maxPages = Math.floor(config.capacityBytes / pageBytes);
  const state = {
    logicalTokenLength: 0,
    highWaterTokenLength: 0,
    generation: 0,
    livePageIds: [] as string[],
    seenPageIds: new Set<string>(),
    seenCheckpointIds: new Set<string>(),
    activeCheckpoint: undefined as PagedKvCheckpoint | undefined,
  };

  for (let index = 0; index < events.length; index++) {
    const event = events[index];
    try {
      if (
        event.contractRevision !== PAGED_KV_CONTRACT_REVISION
        || event.sourceId !== sourceId
        || event.sourceSequence !== index
      ) {
        replayFail("trace envelope mismatch");
      }
      if (index === 0 && event.kind !== "initialize") {
        replayFail("first event must initialize the cache");
      }
      if (index > 0 && event.kind === "initialize") {
        replayFail("cache initialized more than once");
      }

      switch (event.kind) {
        case "initialize": {
          if (
            event.sequenceId !== config.sequenceId
            || event.pageSizeTokens !== config.pageSizeTokens
            || event.bytesPerToken !== config.bytesPerToken
            || event.capacityBytes !== config.capacityBytes
            || event.initialTokenLength !== (config.initialTokenLength ?? 0)
          ) {
            replayFail("initialize payload does not match config");
          }
          const expectedPages = pagesForTokens(
            event.initialTokenLength,
            config.pageSizeTokens,
          );
          acceptAllocatedPages(
            state,
            event.allocatedPageIds,
            expectedPages,
            maxPages,
          );
          state.logicalTokenLength = event.initialTokenLength;
          state.highWaterTokenLength = event.initialTokenLength;
          break;
        }
        case "append": {
          if (
            event.baseTokenLength !== state.logicalTokenLength
            || event.finalTokenLength
              !== checkedReplayAdd(
                state.logicalTokenLength,
                event.appendedTokens,
                "append length",
              )
          ) {
            replayFail("append length transition mismatch");
          }
          assertReplayNonNegative(event.appendedTokens, "append tokens");
          const expectedPages = pagesForTokens(
            event.finalTokenLength,
            config.pageSizeTokens,
          );
          acceptAllocatedPages(
            state,
            event.allocatedPageIds,
            expectedPages - state.livePageIds.length,
            maxPages,
          );
          state.logicalTokenLength = event.finalTokenLength;
          state.highWaterTokenLength = Math.max(
            state.highWaterTokenLength,
            event.finalTokenLength,
          );
          break;
        }
        case "checkpoint": {
          if (state.activeCheckpoint) {
            replayFail("overlapping checkpoints are not allowed");
          }
          const checkpoint = event.checkpoint;
          if (
            checkpoint.checkpointId.length === 0
            || state.seenCheckpointIds.has(checkpoint.checkpointId)
            || checkpoint.generation !== state.generation
            || checkpoint.baseTokenLength !== state.logicalTokenLength
          ) {
            replayFail("checkpoint identity/state mismatch");
          }
          state.seenCheckpointIds.add(checkpoint.checkpointId);
          state.activeCheckpoint = { ...checkpoint };
          break;
        }
        case "restore": {
          const checkpoint = state.activeCheckpoint;
          if (
            !checkpoint
            || event.checkpointId !== checkpoint.checkpointId
            || event.generation !== state.generation + 1
          ) {
            replayFail("restore checkpoint/generation mismatch");
          }
          assertReplayNonNegative(event.acceptedTokens, "accepted tokens");
          const finalTokenLength = checkedReplayAdd(
            checkpoint.baseTokenLength,
            event.acceptedTokens,
            "restore length",
          );
          if (
            finalTokenLength !== event.finalTokenLength
            || finalTokenLength > state.logicalTokenLength
          ) {
            replayFail("restore target mismatch");
          }
          const requiredPages = pagesForTokens(
            finalTokenLength,
            config.pageSizeTokens,
          );
          const expectedReleased = state.livePageIds.slice(requiredPages);
          if (!arraysEqual(expectedReleased, event.releasedPageIds)) {
            replayFail("restore released-page list mismatch");
          }
          state.livePageIds.splice(requiredPages);
          state.logicalTokenLength = finalTokenLength;
          state.generation = event.generation;
          state.activeCheckpoint = undefined;
          break;
        }
        case "commit": {
          const checkpoint = state.activeCheckpoint;
          if (
            !checkpoint
            || event.checkpointId !== checkpoint.checkpointId
            || event.generation !== state.generation + 1
          ) {
            replayFail("commit checkpoint/generation mismatch");
          }
          state.generation = event.generation;
          state.activeCheckpoint = undefined;
          break;
        }
      }
      assertReplayState(state, config, pageBytes, maxPages);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new PagedKvReplayError(`event ${index}: ${message}`);
    }
  }
  if (events.length === 0) {
    throw new PagedKvReplayError("trace is empty");
  }

  return {
    appliedEvents: events.length,
    snapshot: buildSnapshot({
      sequenceId: config.sequenceId,
      logicalTokenLength: state.logicalTokenLength,
      highWaterTokenLength: state.highWaterTokenLength,
      generation: state.generation,
      pageSizeTokens: config.pageSizeTokens,
      pageBytes,
      capacityBytes: config.capacityBytes,
      livePageIds: state.livePageIds,
      activeCheckpoint: state.activeCheckpoint,
    }),
  };
}

function buildSnapshot(input: {
  sequenceId: string;
  logicalTokenLength: number;
  highWaterTokenLength: number;
  generation: number;
  pageSizeTokens: number;
  pageBytes: number;
  capacityBytes: number;
  livePageIds: readonly string[];
  activeCheckpoint?: PagedKvCheckpoint;
}): PagedKvSnapshot {
  const livePages = input.livePageIds.map((physicalPageId, index) => {
    const pageStart = index * input.pageSizeTokens;
    const validTokens = Math.max(
      0,
      Math.min(input.pageSizeTokens, input.logicalTokenLength - pageStart),
    );
    return {
      physicalPageId,
      validTokens,
      capacityTokens: input.pageSizeTokens,
      bytes: input.pageBytes,
    };
  });
  const reservedBytes = input.livePageIds.length * input.pageBytes;
  return {
    sequenceId: input.sequenceId,
    logicalTokenLength: input.logicalTokenLength,
    highWaterTokenLength: input.highWaterTokenLength,
    generation: input.generation,
    pageSizeTokens: input.pageSizeTokens,
    pageBytes: input.pageBytes,
    capacityBytes: input.capacityBytes,
    reservedBytes,
    freeBytes: input.capacityBytes - reservedBytes,
    livePages,
    ...(input.activeCheckpoint
      ? { activeCheckpoint: { ...input.activeCheckpoint } }
      : {}),
  };
}

function acceptAllocatedPages(
  state: {
    livePageIds: string[];
    seenPageIds: Set<string>;
  },
  pageIds: readonly string[],
  expectedCount: number,
  maxPages: number,
): void {
  if (pageIds.length !== expectedCount) {
    replayFail(
      `allocated ${pageIds.length} pages, expected ${expectedCount}`,
    );
  }
  if (state.livePageIds.length + pageIds.length > maxPages) {
    replayFail("allocated pages exceed capacity");
  }
  for (const pageId of pageIds) {
    if (pageId.length === 0 || state.seenPageIds.has(pageId)) {
      replayFail(`page id ${pageId} is empty or reused`);
    }
    state.seenPageIds.add(pageId);
    state.livePageIds.push(pageId);
  }
}

function assertReplayState(
  state: {
    logicalTokenLength: number;
    highWaterTokenLength: number;
    generation: number;
    livePageIds: string[];
    activeCheckpoint?: PagedKvCheckpoint;
  },
  config: PagedKvConfig,
  pageBytes: number,
  maxPages: number,
): void {
  if (
    pagesForTokens(state.logicalTokenLength, config.pageSizeTokens)
      !== state.livePageIds.length
    || state.livePageIds.length > maxPages
    || state.logicalTokenLength > state.highWaterTokenLength
    || state.livePageIds.length * pageBytes > config.capacityBytes
    || state.activeCheckpoint?.generation !== undefined
      && state.activeCheckpoint.generation !== state.generation
  ) {
    replayFail("paged KV invariant failed");
  }
}

function validateConfig(config: PagedKvConfig): void {
  if (config.sequenceId.length === 0) {
    throw new PagedKvProtocolError("sequenceId must not be empty");
  }
  if ((config.sourceId ?? "x").length === 0) {
    throw new PagedKvProtocolError("sourceId must not be empty");
  }
  assertPositiveSafeInteger(config.pageSizeTokens, "pageSizeTokens");
  assertPositiveSafeInteger(config.bytesPerToken, "bytesPerToken");
  assertPositiveSafeInteger(config.capacityBytes, "capacityBytes");
  assertNonNegativeSafeInteger(
    config.initialTokenLength ?? 0,
    "initialTokenLength",
  );
}

function checkpointSnapshot(
  checkpoint: PagedKvCheckpoint,
): PagedKvCheckpoint {
  return {
    checkpointId: checkpoint.checkpointId,
    generation: checkpoint.generation,
    baseTokenLength: checkpoint.baseTokenLength,
  };
}

function cloneEvent(event: PagedKvTraceEvent): PagedKvTraceEvent {
  switch (event.kind) {
    case "initialize":
    case "append":
      return { ...event, allocatedPageIds: [...event.allocatedPageIds] };
    case "checkpoint":
      return { ...event, checkpoint: { ...event.checkpoint } };
    case "restore":
      return { ...event, releasedPageIds: [...event.releasedPageIds] };
    case "commit":
      return { ...event };
  }
}

function pagesForTokens(tokens: number, pageSizeTokens: number): number {
  return tokens === 0 ? 0 : Math.ceil(tokens / pageSizeTokens);
}

function checkedAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    throw new PagedKvProtocolError(`${label} exceeds Number.MAX_SAFE_INTEGER`);
  }
  return result;
}

function checkedMultiply(left: number, right: number, label: string): number {
  const result = left * right;
  if (!Number.isSafeInteger(result)) {
    throw new PagedKvProtocolError(`${label} exceeds Number.MAX_SAFE_INTEGER`);
  }
  return result;
}

function checkedReplayAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    replayFail(`${label} exceeds Number.MAX_SAFE_INTEGER`);
  }
  return result;
}

function checkedReplayMultiply(
  left: number,
  right: number,
  label: string,
): number {
  const result = left * right;
  if (!Number.isSafeInteger(result)) {
    throw new PagedKvReplayError(
      `${label} exceeds Number.MAX_SAFE_INTEGER`,
    );
  }
  return result;
}

function assertPositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new PagedKvProtocolError(
      `${label} must be a positive safe integer; got ${value}`,
    );
  }
}

function assertNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new PagedKvProtocolError(
      `${label} must be a non-negative safe integer; got ${value}`,
    );
  }
}

function assertReplayNonNegative(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    replayFail(`${label} must be a non-negative safe integer`);
  }
}

function arraysEqual(
  left: readonly string[],
  right: readonly string[],
): boolean {
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function replayFail(message: string): never {
  throw new Error(message);
}
