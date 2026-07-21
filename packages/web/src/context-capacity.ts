import type { SimulationScenario } from "@inference-sim/core";
import type { DashboardRunConfig } from "./types.js";

export type ContextCapacityEstimate =
  | {
      readonly status: "unavailable";
      readonly reason: string;
    }
  | {
      readonly status: "available";
      readonly kvCacheBytesPerToken: number;
      readonly kvCacheEvidence: "architecture_derived" | "metadata_declared";
      readonly concurrentRequests: number;
      readonly totalKvTokenSlots: number;
      readonly maxContextTokensPerRequest: number;
      readonly maxSingleSequenceTokens: number;
      readonly configuredContextTokensPerRequest: number;
      readonly fitsConfiguredContext: boolean;
      readonly bottleneckDomainId: string;
      readonly bottleneckKvBudgetBytes: number;
    };

export function estimateContextCapacity(
  config: DashboardRunConfig,
  scenario: SimulationScenario,
): ContextCapacityEstimate {
  const profile = config.modelBinding?.executionProfile;
  const kvCacheBytesPerToken = profile?.kvCacheBytesPerToken;
  const kvCacheEvidence = profile?.kvCacheEvidence;
  if (
    kvCacheBytesPerToken === undefined
    || kvCacheBytesPerToken <= 0
    || kvCacheEvidence === undefined
  ) {
    return {
      status: "unavailable",
      reason: "KV bytes per token are not available for this model",
    };
  }

  const allocations = uniquePhysicalAllocations(scenario);
  const weightAllocations = allocations.filter((allocation) => (
    allocation.purpose === "weights"
  ));
  const weightBytes = distributedWeightBytes(
    weightAllocations,
    config.modelBinding?.weightBytes ?? 0,
  );
  const kvAllocations = allocations.filter((allocation) => (
    allocation.purpose === "kv"
  ));
  const kvDomainIds = [...new Set(kvAllocations.map(
    (allocation) => allocation.domainId,
  ))];
  if (kvDomainIds.length === 0) {
    return {
      status: "unavailable",
      reason: "The topology has no explicit KV placement",
    };
  }

  const reservedExcludingKv = new Map<string, number>();
  for (const allocation of allocations) {
    if (allocation.purpose === "kv") continue;
    const optionalCache = allocation.purpose === "cache"
      || allocation.purpose === "backing";
    const bytes = optionalCache && !config.serving.useExpertCache
      ? 0
      : allocation.purpose === "weights"
        ? weightBytes.get(allocation.physicalAllocationId) ?? allocation.bytes
        : allocation.bytes;
    reservedExcludingKv.set(
      allocation.domainId,
      (reservedExcludingKv.get(allocation.domainId) ?? 0) + bytes,
    );
  }

  const declaredKvBytesByDomain = new Map<string, number>();
  for (const allocation of kvAllocations) {
    declaredKvBytesByDomain.set(
      allocation.domainId,
      (declaredKvBytesByDomain.get(allocation.domainId) ?? 0)
        + allocation.bytes,
    );
  }
  const declaredKvBytes = [...declaredKvBytesByDomain.values()].reduce(
    (sum, bytes) => sum + bytes,
    0,
  );
  const equalShare = 1 / kvDomainIds.length;
  const capacities = kvDomainIds.map((domainId) => {
    const domain = scenario.memoryDomains.find((candidate) => (
      candidate.id === domainId
    ))!;
    const budgetBytes = Math.max(
      0,
      domain.resourceLimitBytes - (reservedExcludingKv.get(domainId) ?? 0),
    );
    const shardShare = declaredKvBytes > 0
      ? (declaredKvBytesByDomain.get(domainId) ?? 0) / declaredKvBytes
      : equalShare;
    return {
      domainId,
      budgetBytes,
      tokenSlots: shardShare <= 0
        ? Infinity
        : Math.floor(budgetBytes / (kvCacheBytesPerToken * shardShare)),
    };
  }).sort((left, right) => (
    left.tokenSlots - right.tokenSlots
    || left.domainId.localeCompare(right.domainId)
  ));
  const bottleneck = capacities[0]!;
  const totalKvTokenSlots = Math.max(0, bottleneck.tokenSlots);
  const concurrentRequests = Math.max(1, Math.round(
    config.serving.requestCount,
  ));
  const maxContextTokensPerRequest = Math.floor(
    totalKvTokenSlots / concurrentRequests,
  );
  const configuredContextTokensPerRequest = Math.max(
    1,
    Math.round(
      config.serving.promptTokens + config.serving.outputTokens - 1,
    ),
  );
  return {
    status: "available",
    kvCacheBytesPerToken,
    kvCacheEvidence,
    concurrentRequests,
    totalKvTokenSlots,
    maxContextTokensPerRequest,
    maxSingleSequenceTokens: totalKvTokenSlots,
    configuredContextTokensPerRequest,
    fitsConfiguredContext:
      configuredContextTokensPerRequest <= maxContextTokensPerRequest,
    bottleneckDomainId: bottleneck.domainId,
    bottleneckKvBudgetBytes: bottleneck.budgetBytes,
  };
}

function uniquePhysicalAllocations(scenario: SimulationScenario) {
  const allocations = new Map<string, (
    typeof scenario.placements[number]["allocations"][number]
  )>();
  for (const placement of scenario.placements) {
    for (const allocation of placement.allocations) {
      allocations.set(allocation.physicalAllocationId, allocation);
    }
  }
  return [...allocations.values()];
}

function distributedWeightBytes(
  allocations: ReturnType<typeof uniquePhysicalAllocations>,
  totalBytes: number,
): ReadonlyMap<string, number> {
  const result = new Map<string, number>();
  const declaredTotal = allocations.reduce(
    (sum, allocation) => sum + allocation.bytes,
    0,
  );
  let remaining = totalBytes;
  allocations.forEach((allocation, index) => {
    const bytes = index === allocations.length - 1
      ? remaining
      : Math.floor(totalBytes * (allocation.bytes / declaredTotal));
    result.set(allocation.physicalAllocationId, bytes);
    remaining -= bytes;
  });
  return result;
}
