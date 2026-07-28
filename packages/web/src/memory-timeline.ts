import type { DashboardResult } from "./types.js";

/** One instant of the run, and the bytes each purpose held at it. */
export interface MemoryTimelineSample {
  readonly atNs: number;
  /** Bytes per allocation purpose, including the varying KV term. */
  readonly bytes: Readonly<Record<string, number>>;
  readonly total: number;
  /** Requests holding a KV arena at this instant. */
  readonly liveRequests: number;
}

export interface MemoryTimeline {
  readonly domainId: string;
  readonly capacityBytes: number;
  readonly residentBytes: number;
  readonly peakTotalBytes: number;
  /** Purposes present in this domain, KV last so it stacks on top. */
  readonly purposes: readonly string[];
  readonly samples: readonly MemoryTimelineSample[];
  /**
   * What the series cannot show. A request's KV is charged from the instant
   * its first token exists, because that is the last event the trace records
   * before the arena is certainly held; the arena is really allocated as
   * prefill begins, so the rise is drawn slightly late.
   */
  readonly caveat: string;
}

/**
 * Reconstruct how much memory the run held over time.
 *
 * Weights and caches do not move once loaded, so the only term that varies is
 * KV: it grows as a request generates and is released when the request
 * retires. The static ledger reports one high-water figure and cannot show
 * that shape, which is what makes a run look permanently full when it was
 * only briefly so.
 *
 * Returns undefined when the run has no per-request timeline to derive from.
 */
export function memoryTimeline(
  result: Pick<DashboardResult, "serving" | "scenario">,
): MemoryTimeline | undefined {
  const serving = result.serving;
  if (serving === undefined || serving.requests.length === 0) {
    return undefined;
  }
  // The domain that holds the weights is the one worth charting: it is where
  // the model and its KV contend.
  const entry = result.scenario.memoryLedger
    .filter((candidate) => candidate.enabled)
    .find((candidate) => (candidate.reservedByPurpose.weights ?? 0) > 0);
  if (entry === undefined) {
    return undefined;
  }
  const kvReservedBytes = entry.reservedByPurpose.kv ?? 0;
  const residentBytes = entry.reservedBytes - kvReservedBytes;
  // Every purpose except KV holds a fixed extent for the whole run. Keeping
  // them apart rather than summing them answers what the memory is for, which
  // one undifferentiated band underneath cannot.
  const constants = Object.entries(entry.reservedByPurpose)
    .filter(([purpose, bytes]) => purpose !== "kv" && bytes > 0)
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0));
  const bytesPerToken = serving.kvBudgetTokens > 0
    ? kvReservedBytes / serving.kvBudgetTokens
    : 0;
  // Prompt length is uniform across a dashboard run, so the prefilled extent
  // each request holds is its share of the total prefill.
  const promptTokens = serving.metrics.requests > 0
    ? serving.metrics.prefillTokens / serving.metrics.requests
    : 0;

  // One sample at every instant something changes, so the series is exact
  // rather than resampled: a coarse grid would smooth away a brief peak, and
  // the peak is the number that decides whether a run fits.
  const instants = new Set<number>([0]);
  for (const request of serving.requests) {
    instants.add(request.firstTokenNs);
    instants.add(request.completedAtNs);
    for (const at of request.tokenTimestampsNs) {
      instants.add(at);
    }
  }
  const ordered = [...instants].sort((left, right) => left - right);

  const samples = ordered.map((atNs) => {
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
      tokens += promptTokens + Math.max(0, generated - 1);
    }
    const kv = tokens * bytesPerToken;
    return {
      atNs,
      bytes: {
        ...Object.fromEntries(constants),
        kv,
      },
      total: residentBytes + kv,
      liveRequests,
    };
  });

  return {
    domainId: entry.domainId,
    capacityBytes: entry.capacityBytes,
    residentBytes,
    peakTotalBytes: samples.reduce(
      (peak, sample) => Math.max(peak, sample.total),
      residentBytes,
    ),
    purposes: [...constants.map(([purpose]) => purpose), "kv"],
    samples,
    caveat: "KV is charged from a request's first token, so the rise is drawn"
      + " slightly later than the arena is really allocated.",
  };
}
