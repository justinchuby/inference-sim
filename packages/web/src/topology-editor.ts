import {
  configureSmallLanNetwork,
  parseSimulationScenario,
  type ComputeCapability,
  type NetworkTransportMode,
  type SimulationScenario,
} from "@inference-sim/core";

export const COMPUTE_CAPABILITIES: readonly ComputeCapability[] = [
  "attention",
  "ffn",
  "collective",
  "copy",
  "sampling",
  "draft",
  "lookup",
];

export const LINK_KINDS = [
  "on-chip",
  "pcie",
  "nvlink",
  "ethernet",
  "infiniband",
  "thunderbolt",
  "storage",
] as const;

export const NETWORK_TRANSPORTS: readonly NetworkTransportMode[] = [
  "tcp",
  "rdma_host",
  "gpudirect_rdma",
];

export function materializeNetworkResources(
  scenario: SimulationScenario,
): SimulationScenario {
  if ((scenario.networkResources?.length ?? 0) > 0) {
    return scenario;
  }
  const networkLink = scenario.links.find((link) => (
    link.transport !== undefined
    && (link.kind === "ethernet" || link.kind === "infiniband")
  ));
  if (networkLink === undefined) {
    return scenario;
  }
  return configureSmallLanNetwork(scenario, {
    advanced: true,
    linkKind: networkLink.kind as "ethernet" | "infiniband",
    transport: networkLink.transport,
    bandwidthBytesPerSec: networkLink.bandwidthBytesPerSec,
    latencyNs: networkLink.latencyNs,
    linkConcurrencyLanes: networkLink.concurrencyLanes,
  });
}

export function finalizeEditedTopology(
  input: SimulationScenario,
): SimulationScenario {
  const provenance = {
    confidence: "heuristic" as const,
    source: "user-edited in inference-sim web topology editor",
  };
  const suffix = input.id.endsWith("-custom") ? "" : "-custom";
  return parseSimulationScenario({
    ...input,
    id: `${input.id}${suffix}`,
    family: "custom",
    memoryDomains: input.memoryDomains.map((domain) => ({
      ...domain,
      provenance,
    })),
    devices: input.devices.map((device) => ({
      ...device,
      provenance,
    })),
    ...(input.networkResources === undefined
      ? {}
      : {
          networkResources: input.networkResources.map((resource) => ({
            ...resource,
            provenance,
          })),
        }),
    links: input.links.map((link) => ({
      ...link,
      provenance,
    })),
    execution: {
      ...input.execution,
      topologyEpoch: input.execution.topologyEpoch + 1,
    },
  });
}

export function gibibytesToBytes(value: number): number {
  return scaledSafeInteger(value, 1024 ** 3, "memory capacity");
}

export function gigabytesPerSecondToBytes(value: number): number {
  return scaledSafeInteger(value, 1_000_000_000, "bandwidth");
}

export function bytesToGibibytes(value: number): number {
  return value / 1024 ** 3;
}

export function bytesToGigabytesPerSecond(value: number): number {
  return value / 1_000_000_000;
}

function scaledSafeInteger(
  value: number,
  scale: number,
  label: string,
): number {
  const result = value * scale;
  if (
    !Number.isFinite(value)
    || value <= 0
    || !Number.isSafeInteger(result)
  ) {
    throw new Error(`${label} must resolve to a positive safe integer`);
  }
  return result;
}
