import { describe, expect, it } from "vitest";
import {
  PagedKvCacheSimulator,
  PagedKvProtocolError,
  PagedKvReplayError,
  replayPagedKvTrace,
  type PagedKvConfig,
  type PagedKvTraceEvent,
} from "../src/index.js";

const config: PagedKvConfig = {
  sequenceId: "sequence-1",
  pageSizeTokens: 4,
  bytesPerToken: 8,
  capacityBytes: 3 * 4 * 8,
  initialTokenLength: 3,
};

describe("PagedKvCacheSimulator", () => {
  it("restores by checkpoint plus accepted offset and masks the partial tail", () => {
    const cache = new PagedKvCacheSimulator(config);
    const checkpoint = cache.checkpoint();
    cache.append(7);
    expect(cache.snapshot().livePages).toHaveLength(3);

    cache.restore(checkpoint, 2);
    cache.append(1);
    const snapshot = cache.snapshot();

    expect(snapshot.logicalTokenLength).toBe(6);
    expect(snapshot.highWaterTokenLength).toBe(10);
    expect(snapshot.livePages.map((page) => page.validTokens)).toEqual([4, 2]);
    expect(snapshot.reservedBytes).toBe(64);
    expect(snapshot.freeBytes).toBe(32);
    expect(replayPagedKvTrace(config, cache.trace()).snapshot).toEqual(snapshot);
  });

  it("preflights capacity atomically", () => {
    const cache = new PagedKvCacheSimulator(config);
    const before = cache.snapshot();

    expect(() => cache.append(10)).toThrowError(PagedKvProtocolError);
    expect(cache.snapshot()).toEqual(before);
    expect(cache.trace()).toHaveLength(1);
  });

  it("never reuses a released physical page id", () => {
    const cache = new PagedKvCacheSimulator(config);
    const checkpoint = cache.checkpoint();
    cache.append(7);
    const released = cache.snapshot().livePages[2].physicalPageId;
    cache.restore(checkpoint, 0);
    cache.append(5);
    const pageIds = cache.snapshot().livePages.map((page) => page.physicalPageId);

    expect(pageIds).not.toContain(released);
    expect(new Set(pageIds).size).toBe(pageIds.length);
  });

  it("consumes checkpoints exactly once on restore or commit", () => {
    const cache = new PagedKvCacheSimulator(config);
    const restored = cache.checkpoint();
    cache.append(1);
    cache.restore(restored, 0);
    expect(() => cache.restore(restored, 0)).toThrowError(PagedKvProtocolError);

    const committed = cache.checkpoint();
    cache.append(1);
    cache.commit(committed);
    expect(() => cache.commit(committed)).toThrowError(PagedKvProtocolError);
  });
});

describe("replayPagedKvTrace", () => {
  it("rejects page identity reuse at the shortest bad event", () => {
    const cache = new PagedKvCacheSimulator(config);
    cache.append(5);
    const trace: PagedKvTraceEvent[] = cache.trace().map((event) => (
      event.kind === "append"
        ? {
            ...event,
            allocatedPageIds: [
              cache.trace()[0].kind === "initialize"
                ? cache.trace()[0].allocatedPageIds[0]
                : "impossible",
              ...event.allocatedPageIds.slice(1),
            ],
          }
        : { ...event }
    ));

    expect(() => replayPagedKvTrace(config, trace)).toThrowError(
      PagedKvReplayError,
    );
    expect(() => replayPagedKvTrace(config, trace)).toThrowError(/^event 1:/);
  });

  it("rejects a corrupted released-page list", () => {
    const cache = new PagedKvCacheSimulator(config);
    const checkpoint = cache.checkpoint();
    cache.append(7);
    cache.restore(checkpoint, 0);
    const trace = cache.trace().map((event) => (
      event.kind === "restore"
        ? { ...event, releasedPageIds: [] }
        : { ...event }
    ));

    expect(() => replayPagedKvTrace(config, trace)).toThrowError(
      "event 3: restore released-page list mismatch",
    );
  });
});
