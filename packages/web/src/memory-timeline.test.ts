import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "./App.js";
import { simulateDashboard } from "./dashboard-simulation.js";
import { memoryTimeline } from "./memory-timeline.js";
import type { DashboardResult, DashboardRunConfig } from "./types.js";

const batched: DashboardRunConfig = {
  ...DEFAULT_CONFIG,
  serving: {
    ...DEFAULT_CONFIG.serving,
    requestCount: 6,
    maxBatchSize: 3,
    outputTokens: 12,
    arrivalGapUs: 20_000,
  },
};

describe("memory timeline", () => {
  it("shows KV rising and being released rather than one flat figure", () => {
    const timeline = memoryTimeline(simulateDashboard(batched))!;

    expect(timeline.samples.length).toBeGreaterThan(4);
    const kv = timeline.samples.map((sample) => sample.bytes.kv!);
    // It must actually vary: a static ledger already reports the high water,
    // and a chart that only redraws that number would be worth nothing.
    expect(Math.max(...kv)).toBeGreaterThan(Math.min(...kv));
    // And it must come back down, or nothing was ever released.
    const peakAt = kv.indexOf(Math.max(...kv));
    expect(Math.min(...kv.slice(peakAt))).toBeLessThan(Math.max(...kv));
  });

  it("never charges more than the domain can hold", () => {
    const timeline = memoryTimeline(simulateDashboard(batched))!;

    for (const sample of timeline.samples) {
      expect(sample.total).toBeLessThanOrEqual(timeline.capacityBytes);
      expect(sample.bytes.kv!).toBeGreaterThanOrEqual(0);
      // The bands must add up to the line the reader sees at the top.
      expect(
        Object.values(sample.bytes).reduce((sum, value) => sum + value, 0),
      ).toBeCloseTo(sample.total, 6);
    }
  });

  it("agrees with the ledger the run reports", () => {
    const result = simulateDashboard(batched);
    const timeline = memoryTimeline(result)!;
    const entry = result.scenario.memoryLedger.find(
      (candidate) => candidate.domainId === timeline.domainId,
    )!;

    // The peak of the series cannot exceed the reservation the ledger made,
    // or the two views of the same run would be telling different stories.
    expect(timeline.peakTotalBytes).toBeLessThanOrEqual(entry.reservedBytes);
    expect(timeline.residentBytes).toBeCloseTo(
      entry.reservedBytes - (entry.reservedByPurpose.kv ?? 0),
      6,
    );
    // Every purpose the ledger reports for this domain is charted, and KV is
    // last so it stacks on top of what does not move.
    expect(timeline.purposes).toContain("weights");
    expect(timeline.purposes.at(-1)).toBe("kv");
  });

  it("counts every request that is holding an arena", () => {
    const timeline = memoryTimeline(simulateDashboard(batched))!;

    const busiest = Math.max(
      ...timeline.samples.map((sample) => sample.liveRequests),
    );
    expect(busiest).toBeGreaterThan(1);
    expect(busiest).toBeLessThanOrEqual(batched.serving.requestCount);
    // KV must be zero exactly when nothing is live, and positive when
    // something is, or the two series contradict each other.
    for (const sample of timeline.samples) {
      expect(sample.bytes.kv! > 0).toBe(sample.liveRequests > 0);
    }
  });

  it("offers nothing for a run with no per-request timeline", () => {
    // The expert-cache study schedules no requests, so there is no arena to
    // follow and the chart must be absent rather than empty.
    expect(memoryTimeline(simulateDashboard({
      ...DEFAULT_CONFIG,
      mode: "expert-cache",
    }))).toBeUndefined();
  });

  it("attributes each request its own prompt rather than the run's mean", () => {
    const result = simulateDashboard(batched);
    const timeline = memoryTimeline(result)!;
    const requests = result.serving!.requests;

    // Carried per request, so a run with uneven prompts bends the shape
    // correctly instead of only getting the peak right.
    for (const request of requests) {
      expect(request.promptTokens).toBeGreaterThan(0);
    }
    // With every request holding its whole prompt and all but one of its
    // generated tokens, the peak must equal the largest concurrent sum.
    const bytesPerToken = timeline.samples
      .map((sample) => sample.bytes.kv!)
      .filter((kv) => kv > 0)[0]! / requests[0]!.promptTokens;
    const peakKv = Math.max(...timeline.samples.map((s) => s.bytes.kv!));
    const busiest = Math.max(
      ...timeline.samples.map((sample) => sample.liveRequests),
    );
    expect(peakKv / bytesPerToken).toBeGreaterThanOrEqual(
      busiest * requests[0]!.promptTokens - 1e-6,
    );
  });

  it("stays linear in the number of token events", () => {
    // Evaluating every request at every instant is quadratic in a product that
    // reaches tens of millions at the maximum request and output counts, which
    // blocked the main thread for over a second at a quarter of that and for
    // minutes at the top. Doubling the work must roughly double the time, not
    // square it.
    const run = (outputTokens: number) => {
      const result = simulateDashboard({
        ...DEFAULT_CONFIG,
        serving: {
          ...DEFAULT_CONFIG.serving,
          requestCount: 16,
          maxBatchSize: 1,
          outputTokens,
          maxBatchTokens: 512,
        },
      });
      const startedAt = performance.now();
      const timeline = memoryTimeline(result)!;
      return {
        samples: timeline.samples.length,
        elapsedMs: performance.now() - startedAt,
      };
    };

    const small = run(256);
    const large = run(1024);
    expect(large.samples / small.samples).toBeGreaterThan(3);
    // Generous, because a timing assertion must not be flaky. A quadratic
    // implementation grows about sixteen-fold here and would blow through it.
    expect(large.elapsedMs).toBeLessThan(Math.max(small.elapsedMs * 8, 250));
  });
});

/**
 * The straightforward reading of the same definition: at each instant, ask
 * every request whether it is live and how many tokens it has emitted. This is
 * what the sweep replaced, kept as an oracle because the sweep's deltas are
 * easy to get subtly wrong at boundaries and no property assertion would
 * notice a one-instant shift.
 */
function referenceTimeline(
  result: Pick<DashboardResult, "serving" | "scenario">,
): readonly {
  readonly atNs: number;
  readonly kv: number;
  readonly liveRequests: number;
}[] {
  const serving = result.serving!;
  const entry = result.scenario.memoryLedger
    .filter((candidate) => candidate.enabled)
    .find((candidate) => (
      (candidate.reservedByPurpose.weights ?? 0) > 0
      && (candidate.reservedByPurpose.kv ?? 0) > 0
    ))!;
  const bytesPerToken = serving.kvBudgetTokens > 0
    ? (entry.reservedByPurpose.kv ?? 0) / serving.kvBudgetTokens
    : 0;
  const instants = new Set<number>();
  for (const request of serving.requests) {
    instants.add(request.firstTokenNs);
    instants.add(request.completedAtNs);
    for (const at of request.tokenTimestampsNs) {
      instants.add(at);
    }
  }
  return [...instants].sort((left, right) => left - right).map((atNs) => {
    let tokens = 0;
    let liveRequests = 0;
    for (const request of serving.requests) {
      if (atNs < request.firstTokenNs || atNs > request.completedAtNs) {
        continue;
      }
      liveRequests++;
      let generated = 0;
      for (const at of request.tokenTimestampsNs) {
        if (at <= atNs) {
          generated++;
        }
      }
      tokens += request.promptTokens + Math.max(0, generated - 1);
    }
    return { atNs, kv: tokens * bytesPerToken, liveRequests };
  });
}

describe("memory timeline sweep", () => {
  it("matches an instant-by-instant reading of the same definition", () => {
    // Batching regimes chosen to exercise the boundaries the sweep could get
    // wrong: several arenas released at once, one request per batch so every
    // token lands on its own instant, arrivals close enough that a completion
    // coincides with another request's first token, and a long single stream.
    for (const [requestCount, maxBatchSize, outputTokens, arrivalGapUs] of [
      [6, 3, 12, 20_000],
      [4, 1, 9, 500],
      [8, 4, 7, 100],
      [3, 2, 20, 5_000],
      [1, 1, 16, 250],
    ] as const) {
      const label = `r${requestCount} b${maxBatchSize} o${outputTokens}`;
      const result = simulateDashboard({
        ...DEFAULT_CONFIG,
        serving: {
          ...DEFAULT_CONFIG.serving,
          requestCount,
          maxBatchSize,
          outputTokens,
          arrivalGapUs,
        },
      });
      const swept = memoryTimeline(result)!;
      const expected = referenceTimeline(result);

      expect(swept.samples.length, label).toBe(expected.length);
      for (const [index, want] of expected.entries()) {
        const got = swept.samples[index]!;
        expect(got.atNs, `${label}@${index}`).toBe(want.atNs);
        expect(got.bytes.kv!, `${label}@${index}`).toBeCloseTo(want.kv, 3);
        expect(got.liveRequests, `${label}@${index}`).toBe(want.liveRequests);
      }
    }
  });
});
