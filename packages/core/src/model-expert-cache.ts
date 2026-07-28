import type { ExpertCacheConfig, ExpertSpec } from "./expert-cache.js";
import { expertRoutingWeights } from "./expert-residency.js";
import type { SimulationScenario } from "./scenario-types.js";
import type { ModelProfile } from "./types.js";
import { compareIds } from "./ordering.js";

/**
 * The three places a routed expert can be read from, and what each costs to
 * reach, taken from the scenario rather than assumed.
 */
export interface ExpertCacheTierDomains {
  readonly hotDomainId: string;
  readonly warmDomainId: string;
  readonly coldDomainId: string;
}

export interface ModelBoundExpertCache {
  readonly config: ExpertCacheConfig;
  /** Experts a token routes to, taken from the checkpoint. */
  readonly topK: number;
  readonly bytesPerExpert: number;
  readonly hotExperts: number;
  readonly warmExperts: number;
}

/** Time to move `bytes` out of a domain, at its declared latency and rate. */
function transferNs(
  domain: SimulationScenario["memoryDomains"][number],
  bytes: number,
): number {
  return Math.max(
    1,
    Math.round(domain.latencyNs + (bytes / domain.bandwidthBytesPerSec) * 1e9),
  );
}

/**
 * Locate the domains backing each expert tier. Hot is where the compute
 * device's own expert cache lives, cold is that node's storage, and warm is
 * whatever the scenario declares between them.
 */
export function expertCacheTierDomains(
  scenario: SimulationScenario,
): ExpertCacheTierDomains | undefined {
  const allocations = scenario.placements.flatMap(
    (placement) => placement.allocations,
  );
  const find = (prefix: string): string | undefined => allocations
    .filter((allocation) => allocation.physicalAllocationId.startsWith(prefix))
    .map((allocation) => allocation.domainId)
    .sort(compareIds)[0];
  const hotDomainId = find("expert-hot-cache:");
  const coldDomainId = find("expert-backing:");
  if (hotDomainId === undefined || coldDomainId === undefined) {
    return undefined;
  }
  // A machine with one memory pool declares no separate warm tier; falling
  // back to hot keeps the two-tier case exact rather than inventing a level.
  return {
    hotDomainId,
    warmDomainId: find("expert-warm-cache:") ?? hotDomainId,
    coldDomainId,
  };
}

/**
 * Build an expert cache from a checkpoint and the machine it runs on.
 *
 * Every quantity that the standalone mechanism study invents is taken from
 * declared data here: the expert count and per-expert extent from the model's
 * MoE geometry, the routed width from its active experts per token, the
 * routing skew from its declared activation distribution, and each tier
 * promotion cost from the bandwidth and latency of the domain the bytes
 * actually come from. Returns undefined for a dense model, which has no
 * routed expert to cache.
 */
export function expertCacheFromModel(
  model: ModelProfile,
  scenario: SimulationScenario,
  options: {
    readonly hotCapacityBytes: number;
    readonly warmCapacityBytes: number;
    readonly routingSeed: number;
    readonly adaptivePrefetch?: ExpertCacheConfig["adaptivePrefetch"];
  },
): ModelBoundExpertCache | undefined {
  const moe = model.moe;
  const tiers = expertCacheTierDomains(scenario);
  if (moe === undefined || moe.numExperts <= 0 || tiers === undefined) {
    return undefined;
  }
  const bytesPerExpert = Math.round(
    moe.expertBytesPerLayer * model.architecture.numLayers,
  );
  if (bytesPerExpert <= 0) {
    return undefined;
  }
  const domain = (id: string) => {
    const found = scenario.memoryDomains.find(
      (candidate) => candidate.id === id,
    );
    if (found === undefined) {
      throw new Error(`expert cache tier references unknown domain ${id}`);
    }
    return found;
  };
  const weights = expertRoutingWeights(
    moe.activationDistribution,
    moe.numExperts,
  );
  // Padded so ordering by id matches ordering by rank, which is what makes
  // "the first N experts" mean "the N most requested" for the initial tiers.
  const width = String(moe.numExperts - 1).length;
  const experts: readonly ExpertSpec[] = weights.map((weight, index) => ({
    id: `expert-${String(index).padStart(width, "0")}`,
    bytes: bytesPerExpert,
    routingWeight: weight,
  }));

  // Capacity is expressed in whole experts: a tier holding part of an expert
  // still has to fetch it, so partial slots would overstate the cache.
  const hotExperts = Math.max(
    1,
    Math.min(
      moe.numExperts,
      Math.floor(options.hotCapacityBytes / bytesPerExpert),
    ),
  );
  const warmExperts = Math.max(
    0,
    Math.min(
      moe.numExperts - hotExperts,
      Math.floor(options.warmCapacityBytes / bytesPerExpert),
    ),
  );
  const topK = Math.max(
    1,
    Math.min(moe.activeExpertsPerToken, moe.numExperts),
  );
  return {
    topK,
    bytesPerExpert,
    hotExperts,
    warmExperts,
    config: {
      experts,
      hotCapacityBytes: hotExperts * bytesPerExpert,
      warmCapacityBytes: warmExperts * bytesPerExpert,
      // One expert's extent out of the tier that holds it. A fixed constant
      // cannot be right across both a 60 MiB expert and an 850 MiB one, nor
      // across a 7 GB/s SSD and a 1 TB/s device memory.
      warmToHotLatencyNs: transferNs(
        domain(tiers.warmDomainId),
        bytesPerExpert,
      ),
      coldToHotLatencyNs: transferNs(
        domain(tiers.coldDomainId),
        bytesPerExpert,
      ),
      coldToWarmLatencyNs: transferNs(
        domain(tiers.coldDomainId),
        bytesPerExpert,
      ),
      routingSeed: options.routingSeed,
      initialHotExpertIds: experts
        .slice(0, hotExperts)
        .map((expert) => expert.id),
      initialWarmExpertIds: experts
        .slice(hotExperts, hotExperts + warmExperts)
        .map((expert) => expert.id),
      ...(options.adaptivePrefetch === undefined
        ? {}
        : { adaptivePrefetch: options.adaptivePrefetch }),
    },
  };
}
