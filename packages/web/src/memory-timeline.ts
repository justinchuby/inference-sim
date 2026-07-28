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
  /**
   * Domains in this topology holding weights, of which this is one. They are
   * not always shards of the same weights: a topology may split attention and
   * FFN across different devices, which is a functional partition rather than
   * a replica, so the count is reported without claiming which.
   */
  readonly weightDomainCount: number;
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
  // the model and its KV contend. A sharded topology has several, and this
  // shows one of them, which the caller is told so it can say so.
  const weightDomains = result.scenario.memoryLedger
    .filter((candidate) => (
      candidate.enabled && (candidate.reservedByPurpose.weights ?? 0) > 0
    ));
  // Only a domain that holds both is chartable. Where a scenario places KV
  // away from the weights, charting the weights domain would draw a flat line
  // and quietly hide the growth happening elsewhere.
  const entry = weightDomains.find(
    (candidate) => (candidate.reservedByPurpose.kv ?? 0) > 0,
  );
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


  // One sample at every instant something changes, so the series is exact
  // rather than resampled: a coarse grid would smooth away a brief peak, and
  // the peak is the number that decides whether a run fits.
  const instants = new Set<number>();
  for (const request of serving.requests) {
    instants.add(request.firstTokenNs);
    instants.add(request.completedAtNs);
    for (const at of request.tokenTimestampsNs) {
      instants.add(at);
    }
  }
  const ordered = [...instants].sort((left, right) => left - right);
  const indexOf = new Map(ordered.map((at, index) => [at, index] as const));

  // Swept rather than evaluated per instant. Scanning every request's tokens
  // at every instant is quadratic in a product that reaches tens of millions
  // at the maximum request and output counts, which is seconds to minutes of
  // blocked main thread. Deltas make it one pass.
  const tokenDelta = new Float64Array(ordered.length + 1);
  const liveDelta = new Float64Array(ordered.length + 1);
  for (const request of serving.requests) {
    // Per request, not the run's mean: dividing total prefill by the request
    // count is right in aggregate but attributes the wrong extent to each
    // request the moment prompts differ, which bends the shape while leaving
    // the peak correct and so would be invisible.
    const promptTokens = request.promptTokens;
    const start = indexOf.get(request.firstTokenNs)!;
    // The arena appears holding the whole prompt: the first token occupies the
    // last prompt position rather than adding one.
    tokenDelta[start] += promptTokens;
    liveDelta[start] += 1;
    for (let index = 1; index < request.tokenTimestampsNs.length; index++) {
      tokenDelta[indexOf.get(request.tokenTimestampsNs[index]!)!] += 1;
    }
    // Held through its completion instant, released after it.
    const end = indexOf.get(request.completedAtNs)! + 1;
    const held = promptTokens
      + Math.max(0, request.tokenTimestampsNs.length - 1);
    tokenDelta[end] -= held;
    liveDelta[end] -= 1;
  }

  let tokens = 0;
  let live = 0;
  const samples = ordered.map((atNs, index) => {
    tokens += tokenDelta[index]!;
    live += liveDelta[index]!;
    // Deltas are integers summed in a float, so they stay exact well past the
    // sample counts reachable here. Clamped anyway so a future non-integer
    // term cannot turn rounding into negative memory.
    const kv = Math.max(0, tokens) * bytesPerToken;
    return {
      atNs,
      bytes: {
        ...Object.fromEntries(constants),
        kv,
      },
      total: residentBytes + kv,
      liveRequests: Math.max(0, Math.round(live)),
    };
  });

  return {
    domainId: entry.domainId,
    weightDomainCount: weightDomains.length,
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
