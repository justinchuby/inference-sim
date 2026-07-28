import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "./App.js";
import { simulateDashboard } from "./dashboard-simulation.js";
import { memoryTimeline } from "./memory-timeline.js";
import type { DashboardRunConfig } from "./types.js";

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
});
