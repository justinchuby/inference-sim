import {
  parseSimulationScenario,
  type ComputeCapability,
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
