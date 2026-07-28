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

/**
 * Time to move `bytes` from one domain to another.
 *
 * Bounded by the slower of the source's own rate and the link between them,
 * because a promotion crosses both. Host DRAM reads at 100 GB/s but arrives
 * over PCIe at 32, so charging the source's rate alone would understate every
 * promotion on a machine whose tiers are genuinely separate. Where a scenario
 * declares no link between the two, the source's rate is all there is.
 */
function transferNs(
  scenario: SimulationScenario,
  sourceId: string,
  targetId: string,
  bytes: number,
): number {
  const source = scenario.memoryDomains.find(
    (candidate) => candidate.id === sourceId,
  );
  if (source === undefined) {
    throw new Error(`expert cache tier references unknown domain ${sourceId}`);
  }
  const link = sourceId === targetId
    ? undefined
    : scenario.links.find((candidate) => (
      candidate.sourceDomainId === sourceId
      && candidate.targetDomainId === targetId
    ));
  const bandwidth = Math.min(
    source.bandwidthBytesPerSec,
    link?.bandwidthBytesPerSec ?? Number.POSITIVE_INFINITY,
  );
  const latency = source.latencyNs + (link?.latencyNs ?? 0);
  return Math.max(1, Math.round(latency + (bytes / bandwidth) * 1e9));
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
 * Known limitation, and why this is not yet wired into serving.
 *
 * An expert here is the whole stack of its per-layer weights, so the unit that
 * moves is `expertBytesPerLayer * numLayers`. Real runtimes work a layer at a
 * time: a batch needs every expert it routes to, but only for the layer it is
 * currently executing, so its working set is one layer wide. For a 235B model
 * that is every expert at 8.8 MiB each, about 1.1 GiB, which fits anywhere.
 * The full-depth unit is 827 MiB each and about 105.8 GiB together, which does
 * not, so a cache modelled at this granularity thrashes where real hardware
 * streams smoothly, and a prefill chunk appears to move several times the
 * model's entire expert set.
 *
 * Note that the hit rate itself is unaffected: per-layer capacity and demand
 * scale by the same layer count, so the resident share is identical. Only the
 * reload volume is wrong, and only when the experts do not all fit.
 *
 * Fixing it means caching expert-layer units rather than experts. That cannot
 * be done by routing once per batch instead of once per token: the same routed
 * assignments also place each token's activations on the device owning each
 * expert, and activations genuinely are per token.
 */

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
        scenario,
        tiers.warmDomainId,
        tiers.hotDomainId,
        bytesPerExpert,
      ),
      coldToHotLatencyNs: transferNs(
        scenario,
        tiers.coldDomainId,
        tiers.hotDomainId,
        bytesPerExpert,
      ),
      coldToWarmLatencyNs: transferNs(
        scenario,
        tiers.coldDomainId,
        tiers.warmDomainId,
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
