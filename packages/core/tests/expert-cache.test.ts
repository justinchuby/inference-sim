import { describe, expect, it } from "vitest";
import {
  ExpertCacheProtocolError,
  ExpertCacheReplayError,
  ExpertCacheSimulator,
  replayExpertCacheTrace,
  simulateExpertCacheWorkload,
  type ExpertCacheConfig,
  type ExpertCacheTraceEvent,
} from "../src/index.js";

const config: ExpertCacheConfig = {
  experts: [
    { id: "e0", bytes: 40, routingWeight: 1 },
    { id: "e1", bytes: 60, routingWeight: 2 },
    { id: "e2", bytes: 80, routingWeight: 3 },
    { id: "e3", bytes: 100, routingWeight: 4 },
  ],
  hotCapacityBytes: 180,
  warmCapacityBytes: 180,
  warmToHotLatencyNs: 5,
  coldToHotLatencyNs: 20,
  coldToWarmLatencyNs: 12,
  routingSeed: 7,
  initialHotExpertIds: ["e0"],
  initialWarmExpertIds: ["e1", "e2"],
};

describe("ExpertCacheSimulator", () => {
  it("routes without replacement and accounts exact byte capacity", () => {
    const cache = new ExpertCacheSimulator(config);
    const result = cache.processToken({ tokenIndex: 0, topK: 2, atNs: 0 });
    const snapshot = cache.snapshot();

    expect(new Set(result.expertIds).size).toBe(2);
    expect(snapshot.hotResidentBytes + snapshot.hotReservedBytes)
      .toBeLessThanOrEqual(config.hotCapacityBytes);
    expect(snapshot.warmResidentBytes + snapshot.warmReservedBytes)
      .toBeLessThanOrEqual(config.warmCapacityBytes);
    expect(snapshot.metrics.routedExperts).toBe(2);
    expect(snapshot.metrics.hotHits + snapshot.metrics.warmMisses
      + snapshot.metrics.coldMisses).toBe(2);
  });

  it("prefetches asynchronously and turns a completed copy into a hit", () => {
    const cache = new ExpertCacheSimulator(config);
    const [loadId] = cache.prefetch(["e3"], "hot", 10);

    expect(loadId).toBeDefined();
    expect(cache.snapshot().pendingLoads).toHaveLength(1);
    expect(cache.snapshot().hotReservedBytes).toBe(100);

    cache.advanceTo(29);
    expect(cache.snapshot().hotExpertIds).not.toContain("e3");
    cache.advanceTo(30);
    expect(cache.snapshot().hotExpertIds).toContain("e3");
    expect(cache.snapshot().hotReservedBytes).toBe(0);
  });

  it("evicts least-recently-used bytes deterministically", () => {
    const cache = new ExpertCacheSimulator({
      ...config,
      hotCapacityBytes: 140,
      initialHotExpertIds: ["e0", "e1"],
      routingSeed: 1,
    });
    cache.prefetch(["e2"], "hot", 0);
    cache.advanceTo(5);

    expect(cache.snapshot().hotExpertIds).toEqual(["e1", "e2"]);
    expect(cache.snapshot().metrics.evictions).toBe(1);
  });

  it("independently replays the full cache trace", () => {
    const cache = new ExpertCacheSimulator(config);
    cache.prefetch(["e3"], "warm", 0);
    cache.advanceTo(12);
    cache.processToken({ tokenIndex: 4, topK: 2, atNs: 12 });

    expect(replayExpertCacheTrace(cache.trace()).snapshot)
      .toEqual(cache.snapshot());
  });

  it("rejects a routed working set larger than hot capacity", () => {
    const cache = new ExpertCacheSimulator({
      ...config,
      hotCapacityBytes: 100,
    });
    expect(() => cache.processToken({ tokenIndex: 0, topK: 4, atNs: 0 }))
      .toThrowError(ExpertCacheProtocolError);
  });

  it("rejects a mutated route at the offending trace prefix", () => {
    const cache = new ExpertCacheSimulator(config);
    cache.processToken({ tokenIndex: 0, topK: 1, atNs: 0 });
    const trace: ExpertCacheTraceEvent[] = structuredClone(cache.trace());
    const route = trace.find((event) => event.kind === "route");
    if (!route || route.kind !== "route") {
      throw new Error("missing route event");
    }
    route.expertIds = [route.expertIds[0] === "e0" ? "e1" : "e0"];

    expect(() => replayExpertCacheTrace(trace))
      .toThrowError(ExpertCacheReplayError);
  });

  it("rejects a non-LRU eviction even when final bytes would fit", () => {
    const cache = new ExpertCacheSimulator({
      ...config,
      hotCapacityBytes: 140,
      initialHotExpertIds: ["e0", "e1"],
      routingSeed: 1,
    });
    cache.prefetch(["e2"], "hot", 0);
    const trace: ExpertCacheTraceEvent[] = structuredClone(cache.trace());
    const eviction = trace.find((event) => event.kind === "evict");
    if (!eviction || eviction.kind !== "evict") {
      throw new Error("missing eviction event");
    }
    eviction.expertId = "e1";
    eviction.bytes = 60;

    expect(() => replayExpertCacheTrace(trace))
      .toThrowError(ExpertCacheReplayError);
  });

  it("rejects duplicate or unknown initial expert identities", () => {
    expect(() => new ExpertCacheSimulator({
      ...config,
      experts: [...config.experts, { id: "e0", bytes: 1 }],
    })).toThrowError(ExpertCacheProtocolError);
    expect(() => new ExpertCacheSimulator({
      ...config,
      initialHotExpertIds: ["missing"],
    })).toThrowError(ExpertCacheProtocolError);
    expect(() => new ExpertCacheSimulator({
      ...config,
      initialHotExpertIds: ["e0", "e0"],
    })).toThrowError(ExpertCacheProtocolError);
  });

  it("runs a deterministic workload with an initial prefetch window", () => {
    const first = simulateExpertCacheWorkload({
      cache: config,
      tokenCount: 4,
      topK: 2,
      tokenIntervalNs: 10,
      initialPrefetch: {
        expertIds: ["e3"],
        targetTier: "warm",
        leadTimeNs: 12,
      },
    });
    const second = simulateExpertCacheWorkload({
      cache: config,
      tokenCount: 4,
      topK: 2,
      tokenIntervalNs: 10,
      initialPrefetch: {
        expertIds: ["e3"],
        targetTier: "warm",
        leadTimeNs: 12,
      },
    });

    expect(first).toEqual(second);
    expect(first.routes).toHaveLength(4);
    expect(first.snapshot.metrics.routes).toBe(4);
  });
});
