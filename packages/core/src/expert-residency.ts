import type { ExpertDistribution, ModelProfile } from "./types.js";

/**
 * How a sparse model's routed experts divide between memory and storage when
 * they do not all fit, and how often a token's routed reads are served from
 * each.
 */
export interface ExpertResidencyPlan {
  readonly totalExpertsPerLayer: number;
  readonly residentExpertsPerLayer: number;
  /** Routed-expert bytes held in memory across every layer. */
  readonly residentExpertBytes: number;
  /** Routed-expert bytes left on storage across every layer. */
  readonly streamedExpertBytes: number;
  /**
   * Share of a token's routed-expert reads served from memory. Derived from
   * the model's declared routing distribution, because a cache that keeps the
   * most-requested experts resident hits far more often than the plain ratio
   * of resident experts to total experts under a skewed distribution.
   */
  readonly residentHitFraction: number;
  /** Routed bytes one token reads that miss residency. */
  readonly streamedBytesPerToken: number;
}

/**
 * Probability mass of the `count` most-requested experts out of `total`.
 *
 * A cache holding the most popular experts is the limit an LRU converges to,
 * so this is the hit rate a well-behaved runtime approaches rather than an
 * optimistic bound invented here.
 */
export function topExpertMass(
  distribution: ExpertDistribution,
  count: number,
  total: number,
): number {
  if (total <= 0 || count <= 0) {
    return 0;
  }
  if (count >= total) {
    return 1;
  }
  switch (distribution.kind) {
    case "uniform":
      return count / total;
    case "zipf": {
      // Rank i carries weight 1/i^s, so the top `count` carry the ratio of the
      // truncated harmonic sums.
      let head = 0;
      let all = 0;
      for (let rank = 1; rank <= total; rank++) {
        const weight = 1 / rank ** distribution.s;
        all += weight;
        if (rank <= count) {
          head += weight;
        }
      }
      return all === 0 ? 0 : head / all;
    }
    case "clustered": {
      const hot = Math.min(distribution.hotExperts, total);
      const cold = total - hot;
      if (count <= hot) {
        return hot === 0 ? 0 : distribution.hotFrequency * (count / hot);
      }
      const coldMass = 1 - distribution.hotFrequency;
      return cold === 0
        ? 1
        : distribution.hotFrequency + coldMass * ((count - hot) / cold);
    }
    case "empirical": {
      const sorted = [...distribution.frequencies].sort((a, b) => b - a);
      const all = sorted.reduce((sum, value) => sum + value, 0);
      if (all === 0) {
        return 0;
      }
      const head = sorted.slice(0, count).reduce((sum, value) => sum + value, 0);
      return head / all;
    }
  }
}

/**
 * Divide a model's routed experts between memory and storage given the bytes
 * left for them after everything that must be resident.
 *
 * Returns undefined for a model with no routed experts: a dense model has
 * nothing to stream, and pretending otherwise would let it appear to run on a
 * machine that cannot hold it.
 */
export function planExpertResidency(
  model: ModelProfile,
  availableExpertBytes: number,
): ExpertResidencyPlan | undefined {
  const moe = model.moe;
  if (moe === undefined || moe.numExperts <= 0) {
    return undefined;
  }
  const layers = model.architecture.numLayers;
  const bytesPerExpert = moe.expertBytesPerLayer * layers;
  const totalRoutedBytes = bytesPerExpert * moe.numExperts;
  if (bytesPerExpert <= 0) {
    return undefined;
  }
  // Experts are held whole. A partially resident expert would still have to be
  // read from storage, so rounding down is the only honest direction.
  const residentExperts = Math.max(
    0,
    Math.min(
      moe.numExperts,
      Math.floor(Math.max(0, availableExpertBytes) / bytesPerExpert),
    ),
  );
  const residentHitFraction = topExpertMass(
    moe.activationDistribution,
    residentExperts,
    moe.numExperts,
  );
  const activeRoutedBytesPerToken =
    moe.activeExpertsPerToken * moe.expertBytesPerLayer * layers;
  return {
    totalExpertsPerLayer: moe.numExperts,
    residentExpertsPerLayer: residentExperts,
    residentExpertBytes: residentExperts * bytesPerExpert,
    streamedExpertBytes: totalRoutedBytes - residentExperts * bytesPerExpert,
    residentHitFraction,
    streamedBytesPerToken:
      activeRoutedBytesPerToken * (1 - residentHitFraction),
  };
}
