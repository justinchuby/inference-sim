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

  it("holds route access until externally retimed demand loads complete", () => {
    const cache = new ExpertCacheSimulator({
      ...config,
      initialHotExpertIds: [],
      initialWarmExpertIds: [],
    });
    const pendingRoute = cache.beginTokenRoute({
      tokenIndex: 0,
      topK: 1,
      atNs: 0,
    });
    const [loadId] = pendingRoute.newDemandLoadIds;

    expect(loadId).toBeDefined();
    expect(pendingRoute.requiredLoadIds).toEqual([loadId]);
    expect(cache.trace().some((event) => event.kind === "access")).toBe(false);
    expect(() => cache.beginTokenRoute({
      tokenIndex: 1,
      topK: 1,
      atNs: 0,
    })).toThrow("must complete before another route begins");
    expect(() => cache.prefetch(["e0"], "warm", 0))
      .toThrow("must complete before prefetch");
    expect(() => cache.retimePendingPrefetch(loadId, 30))
      .toThrow("cannot retime non-prefetch");

    cache.retimePendingLoad(loadId, 37, 37);
    const completed = cache.completeTokenRoute(pendingRoute.requestId);

    expect(completed.readyAtNs).toBe(37);
    expect(completed.stallNs).toBe(37);
    expect(cache.snapshot().pendingLoads).toHaveLength(0);
    expect(replayExpertCacheTrace(cache.trace()).snapshot)
      .toEqual(cache.snapshot());

    const omittedAccess = cache.trace().filter(
      (event) => event.kind !== "access",
    );
    expect(() => replayExpertCacheTrace(omittedAccess))
      .toThrow("did not complete");
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

  it("retimes pending prefetch completion with replayable reservation state", () => {
    const cache = new ExpertCacheSimulator(config);
    const [loadId] = cache.prefetch(["e3"], "warm", 0);
    const updated = cache.retimePendingPrefetch(loadId, 30);

    expect(updated.completesAtNs).toBe(30);
    cache.advanceTo(12);
    expect(cache.snapshot().pendingLoads[0].completesAtNs).toBe(30);
    expect(cache.snapshot().warmExpertIds).not.toContain("e3");
    cache.advanceTo(30);
    expect(cache.snapshot().warmExpertIds).toContain("e3");
    expect(replayExpertCacheTrace(cache.trace()).snapshot)
      .toEqual(cache.snapshot());
  });

  it("rejects invalid or mutated prefetch retiming", () => {
    const cache = new ExpertCacheSimulator(config);
    const [loadId] = cache.prefetch(["e3"], "warm", 10);

    expect(() => cache.retimePendingPrefetch(loadId, 9))
      .toThrow("precedes current/load time");
    cache.retimePendingPrefetch(loadId, 40);
    const trace: ExpertCacheTraceEvent[] = structuredClone(cache.trace());
    const retime = trace.find((event) => event.kind === "load_retime");
    if (retime?.kind !== "load_retime") {
      throw new Error("missing load retime");
    }
    retime.priorCompletesAtNs++;
    expect(() => replayExpertCacheTrace(trace))
      .toThrowError(ExpertCacheReplayError);

    const invalidPhysical: ExpertCacheTraceEvent[] = structuredClone(
      cache.trace(),
    );
    const physicalRetime = invalidPhysical.find(
      (event) => event.kind === "load_retime",
    );
    if (physicalRetime?.kind !== "load_retime") {
      throw new Error("missing load retime");
    }
    physicalRetime.physicalCompletesAtNs =
      physicalRetime.completesAtNs + 1;
    expect(() => replayExpertCacheTrace(invalidPhysical))
      .toThrowError(ExpertCacheReplayError);
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

  it("isolates capacity and LRU eviction across explicit tier partitions", () => {
    const partitioned: ExpertCacheConfig = {
      experts: Array.from({ length: 4 }, (_, index) => ({
        id: `e${index}`,
        bytes: 40,
      })),
      hotCapacityBytes: 80,
      warmCapacityBytes: 0,
      hotPartitions: [
        { id: "owner-0", expertIds: ["e0", "e1"], capacityBytes: 40 },
        { id: "owner-1", expertIds: ["e2", "e3"], capacityBytes: 40 },
      ],
      warmPartitions: [
        { id: "node-0", expertIds: ["e0", "e1"], capacityBytes: 0 },
        { id: "node-1", expertIds: ["e2", "e3"], capacityBytes: 0 },
      ],
      warmToHotLatencyNs: 5,
      coldToHotLatencyNs: 20,
      coldToWarmLatencyNs: 12,
      routingSeed: 7,
      initialHotExpertIds: ["e0", "e2"],
    };
    const cache = new ExpertCacheSimulator(partitioned);

    cache.prefetch(["e1"], "hot", 0);
    cache.advanceTo(20);
    expect(cache.snapshot().hotExpertIds).toEqual(["e2", "e1"]);
    expect(cache.snapshot().hotPartitions).toEqual([
      {
        id: "owner-0",
        capacityBytes: 40,
        residentBytes: 40,
        reservedBytes: 0,
        expertIds: ["e1"],
      },
      {
        id: "owner-1",
        capacityBytes: 40,
        residentBytes: 40,
        reservedBytes: 0,
        expertIds: ["e2"],
      },
    ]);

    cache.prefetch(["e3"], "hot", 20);
    cache.advanceTo(40);
    expect(cache.snapshot().hotExpertIds).toEqual(["e1", "e3"]);
    expect(replayExpertCacheTrace(cache.trace()).snapshot)
      .toEqual(cache.snapshot());

    const mutated: ExpertCacheTraceEvent[] = structuredClone(cache.trace());
    const firstEviction = mutated.find((event) => event.kind === "evict");
    if (firstEviction?.kind !== "evict") {
      throw new Error("missing partition eviction");
    }
    firstEviction.expertId = "e2";
    expect(() => replayExpertCacheTrace(mutated))
      .toThrow("belongs to owner-1, not owner-0");
  });

  it("fails closed for incomplete or inconsistent tier partitions", () => {
    const base: ExpertCacheConfig = {
      ...config,
      initialHotExpertIds: [],
      initialWarmExpertIds: [],
      hotPartitions: [
        { id: "owner-0", expertIds: ["e0", "e1"], capacityBytes: 180 },
      ],
    };
    expect(() => new ExpertCacheSimulator(base))
      .toThrow("hot partitions do not assign experts e2, e3");
    expect(() => new ExpertCacheSimulator({
      ...base,
      hotPartitions: [
        {
          id: "owner-0",
          expertIds: ["e0", "e1", "e2", "e3"],
          capacityBytes: 179,
        },
      ],
    })).toThrow("does not equal aggregate 180");
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

  it("adapts warm prefetch only from observed route history", () => {
    const result = simulateExpertCacheWorkload({
      cache: {
        ...config,
        initialWarmExpertIds: [],
        adaptivePrefetch: {
          targetTier: "warm",
          minObservations: 1,
          intervalTokens: 1,
          maxExpertsPerDecision: 2,
        },
      },
      tokenCount: 4,
      topK: 2,
      tokenIntervalNs: 10,
    });
    const observed = new Set<string>();
    for (const event of result.trace) {
      if (event.kind === "route") {
        event.expertIds.forEach((id) => observed.add(id));
      }
      if (event.kind === "prefetch_decision") {
        expect(event.expertIds.every((id) => observed.has(id))).toBe(true);
      }
    }
    expect(result.snapshot.metrics.adaptivePrefetchDecisions).toBe(4);
    expect(result.snapshot.metrics.adaptivePrefetchSelections)
      .toBeGreaterThan(0);
    expect(replayExpertCacheTrace(result.trace).snapshot)
      .toEqual(result.snapshot);
  });

  it("rejects a mutated or omitted adaptive prefetch decision", () => {
    const result = simulateExpertCacheWorkload({
      cache: {
        ...config,
        initialWarmExpertIds: [],
        adaptivePrefetch: {
          targetTier: "warm",
          minObservations: 1,
          intervalTokens: 1,
          maxExpertsPerDecision: 1,
        },
      },
      tokenCount: 2,
      topK: 1,
      tokenIntervalNs: 10,
    });
    const mutated: ExpertCacheTraceEvent[] = structuredClone(result.trace);
    const decision = mutated.find(
      (event) => event.kind === "prefetch_decision",
    );
    if (!decision || decision.kind !== "prefetch_decision") {
      throw new Error("missing adaptive prefetch decision");
    }
    decision.expertIds = [
      decision.expertIds[0] === "e0" ? "e1" : "e0",
    ];
    expect(() => replayExpertCacheTrace(mutated))
      .toThrowError(ExpertCacheReplayError);

    const omitted = result.trace.filter(
      (event) => event.kind !== "prefetch_decision",
    );
    expect(() => replayExpertCacheTrace(omitted))
      .toThrowError(ExpertCacheReplayError);

    const relabeled: ExpertCacheTraceEvent[] = structuredClone(result.trace);
    const adaptive = relabeled.find((event) => (
      event.kind === "prefetch" && event.trigger === "adaptive"
    ));
    if (!adaptive || adaptive.kind !== "prefetch") {
      throw new Error("missing adaptive prefetch request");
    }
    adaptive.trigger = "manual";
    expect(() => replayExpertCacheTrace(relabeled))
      .toThrowError(ExpertCacheReplayError);
  });
});
