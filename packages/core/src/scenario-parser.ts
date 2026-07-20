import {
  SCENARIO_SCHEMA_VERSION,
  type SimulationScenario,
} from "./scenario-types.js";
import { assertValidScenario } from "./scenario.js";

export class ScenarioParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ScenarioParseError";
  }
}

export function parseSimulationScenario(input: unknown): SimulationScenario {
  return parseSimulationScenarioBoundary(input, "scenario", true);
}

export function parseSimulationScenarioBoundary(
  input: unknown,
  label: string,
  validateSemantics: boolean,
): SimulationScenario {
  const scenario = requireRecord(input, label);
  assertKeys(scenario, [
    "schemaVersion",
    "id",
    "family",
    "memoryDomains",
    "devices",
    "links",
    "placements",
    "transfers",
    "groups",
    "workload",
    "execution",
    "calibration",
  ], ["networkResources"], label);
  requireExact(
    scenario.schemaVersion,
    SCENARIO_SCHEMA_VERSION,
    `${label} schema version`,
  );
  requireStrings(scenario, ["id", "family"], label);
  parseMemoryDomains(scenario.memoryDomains, label);
  parseDevices(scenario.devices, label);
  if (scenario.networkResources !== undefined) {
    parseNetworkResources(scenario.networkResources, label);
  }
  parseLinks(scenario.links, label);
  parsePlacements(scenario.placements, label);
  parseTransfers(scenario.transfers, label);
  parseGroups(scenario.groups, label);
  parseWorkload(scenario.workload, label);
  parseExecution(scenario.execution, label);
  parseCalibration(scenario.calibration, label);

  const parsed = scenario as unknown as SimulationScenario;
  if (validateSemantics) {
    assertValidScenario(parsed);
  }
  return parsed;
}

function parseMemoryDomains(value: unknown, scenarioLabel: string): void {
  requireRecordArray(value, `${scenarioLabel} memoryDomains`)
    .forEach((domain, index) => {
      const label = `${scenarioLabel} memoryDomains[${index}]`;
      assertKeys(domain, [
        "id",
        "nodeId",
        "kind",
        "capacityBytes",
        "resourceLimitBytes",
        "bandwidthBytesPerSec",
        "latencyNs",
        "coherent",
        "allocationClasses",
        "accessibleBy",
        "governor",
        "provenance",
      ], [], label);
      requireStrings(domain, ["id", "nodeId", "kind"], label);
      requireNumbers(
        domain,
        [
          "capacityBytes",
          "resourceLimitBytes",
          "bandwidthBytesPerSec",
          "latencyNs",
        ],
        label,
      );
      requireBoolean(domain.coherent, `${label} coherent`);
      requireStringArray(domain.allocationClasses, `${label} allocationClasses`);
      requireStringArray(domain.accessibleBy, `${label} accessibleBy`);
      parseGovernor(domain.governor, `${label} governor`);
      parseProvenance(domain.provenance, `${label} provenance`);
    });
}

function parseDevices(value: unknown, scenarioLabel: string): void {
  requireRecordArray(value, `${scenarioLabel} devices`)
    .forEach((device, index) => {
      const label = `${scenarioLabel} devices[${index}]`;
      assertKeys(device, [
        "id",
        "nodeId",
        "kind",
        "executionProvider",
        "memoryDomainIds",
        "capabilities",
        "supportedDtypes",
        "maxConcurrentCompute",
        "provenance",
      ], ["computeProfileId"], label);
      requireStrings(
        device,
        ["id", "nodeId", "kind", "executionProvider"],
        label,
      );
      requireStringArray(device.memoryDomainIds, `${label} memoryDomainIds`);
      requireStringArray(device.capabilities, `${label} capabilities`);
      requireStringArray(device.supportedDtypes, `${label} supportedDtypes`);
      if (device.computeProfileId !== undefined) {
        requireNonEmptyString(device.computeProfileId, `${label} computeProfileId`);
      }
      requireFiniteNumber(
        device.maxConcurrentCompute,
        `${label} maxConcurrentCompute`,
      );
      parseProvenance(device.provenance, `${label} provenance`);
    });
}

function parseNetworkResources(value: unknown, scenarioLabel: string): void {
  requireRecordArray(value, `${scenarioLabel} networkResources`)
    .forEach((resource, index) => {
      const label = `${scenarioLabel} networkResources[${index}]`;
      assertKeys(resource, [
        "id",
        "kind",
        "bandwidthBytesPerSec",
        "latencyNs",
        "concurrencyLanes",
        "supportedTransports",
        "directMemoryDomainIds",
        "provenance",
      ], ["nodeId"], label);
      requireStrings(resource, ["id", "kind"], label);
      if (resource.nodeId !== undefined) {
        requireNonEmptyString(resource.nodeId, `${label} nodeId`);
      }
      requireNumbers(
        resource,
        ["bandwidthBytesPerSec", "latencyNs", "concurrencyLanes"],
        label,
      );
      requireStringArray(
        resource.supportedTransports,
        `${label} supportedTransports`,
      );
      requireStringArray(
        resource.directMemoryDomainIds,
        `${label} directMemoryDomainIds`,
      );
      parseProvenance(resource.provenance, `${label} provenance`);
    });
}

function parseLinks(value: unknown, scenarioLabel: string): void {
  requireRecordArray(value, `${scenarioLabel} links`)
    .forEach((link, index) => {
      const label = `${scenarioLabel} links[${index}]`;
      assertKeys(link, [
        "id",
        "sourceDomainId",
        "targetDomainId",
        "kind",
        "bandwidthBytesPerSec",
        "latencyNs",
        "concurrencyLanes",
        "provenance",
      ], ["transport", "networkResourceIds"], label);
      requireStrings(
        link,
        ["id", "sourceDomainId", "targetDomainId", "kind"],
        label,
      );
      requireNumbers(
        link,
        ["bandwidthBytesPerSec", "latencyNs", "concurrencyLanes"],
        label,
      );
      if (link.transport !== undefined) {
        requireNonEmptyString(link.transport, `${label} transport`);
      }
      if (link.networkResourceIds !== undefined) {
        requireStringArray(
          link.networkResourceIds,
          `${label} networkResourceIds`,
        );
      }
      parseProvenance(link.provenance, `${label} provenance`);
    });
}

function parsePlacements(value: unknown, scenarioLabel: string): void {
  requireRecordArray(value, `${scenarioLabel} placements`)
    .forEach((placement, index) => {
      const label = `${scenarioLabel} placements[${index}]`;
      assertKeys(
        placement,
        ["partitionId", "deviceId", "requiredCapabilities", "allocations"],
        ["sharedAllocationIds"],
        label,
      );
      requireStrings(placement, ["partitionId", "deviceId"], label);
      requireStringArray(
        placement.requiredCapabilities,
        `${label} requiredCapabilities`,
      );
      if (placement.sharedAllocationIds !== undefined) {
        requireStringArray(
          placement.sharedAllocationIds,
          `${label} sharedAllocationIds`,
        );
      }
      requireRecordArray(placement.allocations, `${label} allocations`)
        .forEach((allocation, allocationIndex) => {
          const allocationLabel =
            `${label} allocations[${allocationIndex}]`;
          assertKeys(allocation, [
            "physicalAllocationId",
            "domainId",
            "bytes",
            "allocationClass",
            "purpose",
          ], [], allocationLabel);
          requireStrings(allocation, [
            "physicalAllocationId",
            "domainId",
            "allocationClass",
            "purpose",
          ], allocationLabel);
          requireFiniteNumber(allocation.bytes, `${allocationLabel} bytes`);
        });
    });
}

function parseTransfers(value: unknown, scenarioLabel: string): void {
  requireRecordArray(value, `${scenarioLabel} transfers`)
    .forEach((transfer, index) => {
      const label = `${scenarioLabel} transfers[${index}]`;
      assertKeys(transfer, [
        "id",
        "sourceDomainId",
        "targetDomainId",
        "bytes",
        "requiresPinnedStaging",
        "stagingAllocationIds",
      ], [], label);
      requireStrings(
        transfer,
        ["id", "sourceDomainId", "targetDomainId"],
        label,
      );
      requireFiniteNumber(transfer.bytes, `${label} bytes`);
      requireBoolean(
        transfer.requiresPinnedStaging,
        `${label} requiresPinnedStaging`,
      );
      requireStringArray(
        transfer.stagingAllocationIds,
        `${label} stagingAllocationIds`,
      );
    });
}

function parseGroups(value: unknown, scenarioLabel: string): void {
  requireRecordArray(value, `${scenarioLabel} groups`)
    .forEach((group, index) => {
      const label = `${scenarioLabel} groups[${index}]`;
      assertKeys(group, ["id", "orderedRanks"], [], label);
      requireNonEmptyString(group.id, `${label} id`);
      requireRecordArray(group.orderedRanks, `${label} orderedRanks`)
        .forEach((rank, rankIndex) => {
          const rankLabel = `${label} orderedRanks[${rankIndex}]`;
          assertKeys(rank, ["rankId", "deviceId"], [], rankLabel);
          requireStrings(rank, ["rankId", "deviceId"], rankLabel);
        });
    });
}

function parseWorkload(value: unknown, scenarioLabel: string): void {
  const label = `${scenarioLabel} workload`;
  const workload = requireRecord(value, label);
  assertKeys(
    workload,
    ["batchSize", "inputSequenceLength", "outputSequenceLength"],
    ["speculative"],
    label,
  );
  requireNumbers(
    workload,
    ["batchSize", "inputSequenceLength", "outputSequenceLength"],
    label,
  );
  if (workload.speculative !== undefined) {
    const speculativeLabel = `${label} speculative`;
    const speculative = requireRecord(workload.speculative, speculativeLabel);
    assertKeys(
      speculative,
      ["family", "maxAdditionalTokens"],
      [],
      speculativeLabel,
    );
    requireNonEmptyString(speculative.family, `${speculativeLabel} family`);
    requireFiniteNumber(
      speculative.maxAdditionalTokens,
      `${speculativeLabel} maxAdditionalTokens`,
    );
  }
}

function parseExecution(value: unknown, scenarioLabel: string): void {
  const label = `${scenarioLabel} execution`;
  const execution = requireRecord(value, label);
  assertKeys(
    execution,
    ["topologyEpoch", "seed", "maxEvents", "features", "parallelism"],
    [],
    label,
  );
  requireNumbers(execution, ["topologyEpoch", "seed", "maxEvents"], label);
  const featuresLabel = `${label} features`;
  const features = requireRecord(execution.features, featuresLabel);
  assertKeys(features, ["ssdStreaming"], [], featuresLabel);
  requireBoolean(features.ssdStreaming, `${featuresLabel} ssdStreaming`);
  const parallelismLabel = `${label} parallelism`;
  const parallelism = requireRecord(
    execution.parallelism,
    parallelismLabel,
  );
  assertKeys(
    parallelism,
    ["composition", "tensor", "pipeline", "expert", "data"],
    [],
    parallelismLabel,
  );
  requireNonEmptyString(
    parallelism.composition,
    `${parallelismLabel} composition`,
  );
  requireNumbers(
    parallelism,
    ["tensor", "pipeline", "expert", "data"],
    parallelismLabel,
  );
}

function parseCalibration(value: unknown, scenarioLabel: string): void {
  const label = `${scenarioLabel} calibration`;
  const calibration = requireRecord(value, label);
  assertKeys(calibration, ["coefficients"], [], label);
  requireRecordArray(calibration.coefficients, `${label} coefficients`)
    .forEach((coefficient, index) => {
      const coefficientLabel = `${label} coefficients[${index}]`;
      assertKeys(
        coefficient,
        ["id", "value", "unit", "provenance"],
        [],
        coefficientLabel,
      );
      requireStrings(coefficient, ["id", "unit"], coefficientLabel);
      requireFiniteNumber(coefficient.value, `${coefficientLabel} value`);
      parseProvenance(
        coefficient.provenance,
        `${coefficientLabel} provenance`,
      );
    });
}

function parseGovernor(value: unknown, label: string): void {
  const governor = requireRecord(value, label);
  const kind = requireNonEmptyString(governor.kind, `${label} kind`);
  if (kind === "host") {
    assertKeys(governor, ["kind", "nodeId"], [], label);
    requireNonEmptyString(governor.nodeId, `${label} nodeId`);
  } else if (kind === "device") {
    assertKeys(governor, ["kind", "deviceId"], [], label);
    requireNonEmptyString(governor.deviceId, `${label} deviceId`);
  } else if (kind === "none") {
    assertKeys(governor, ["kind"], [], label);
  } else {
    throw new ScenarioParseError(`${label} kind is unsupported`);
  }
}

function parseProvenance(value: unknown, label: string): void {
  const provenance = requireRecord(value, label);
  assertKeys(
    provenance,
    ["confidence", "source"],
    ["measuredAt", "notes"],
    label,
  );
  requireStrings(provenance, ["confidence", "source"], label);
  for (const optional of ["measuredAt", "notes"]) {
    if (provenance[optional] !== undefined) {
      requireNonEmptyString(provenance[optional], `${label} ${optional}`);
    }
  }
}

function requireRecord(
  value: unknown,
  label: string,
): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new ScenarioParseError(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireRecordArray(
  value: unknown,
  label: string,
): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    throw new ScenarioParseError(`${label} must be an array`);
  }
  return value.map((entry, index) => (
    requireRecord(entry, `${label}[${index}]`)
  ));
}

function requireStringArray(value: unknown, label: string): string[] {
  if (
    !Array.isArray(value)
    || value.some((entry) => typeof entry !== "string")
  ) {
    throw new ScenarioParseError(`${label} must be an array of strings`);
  }
  return value;
}

function requireStrings(
  record: Record<string, unknown>,
  fields: readonly string[],
  label: string,
): void {
  for (const field of fields) {
    requireNonEmptyString(record[field], `${label} ${field}`);
  }
}

function requireNumbers(
  record: Record<string, unknown>,
  fields: readonly string[],
  label: string,
): void {
  for (const field of fields) {
    requireFiniteNumber(record[field], `${label} ${field}`);
  }
}

function requireNonEmptyString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new ScenarioParseError(`${label} must be a non-empty string`);
  }
  return value;
}

function requireFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ScenarioParseError(`${label} must be a finite number`);
  }
  return value;
}

function requireBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") {
    throw new ScenarioParseError(`${label} must be boolean`);
  }
  return value;
}

function requireExact(
  value: unknown,
  expected: string | number,
  label: string,
): void {
  if (value !== expected) {
    throw new ScenarioParseError(
      `${label} must be ${String(expected)}, got ${String(value)}`,
    );
  }
}

function assertKeys(
  record: Record<string, unknown>,
  required: readonly string[],
  optional: readonly string[],
  label: string,
): void {
  const allowed = new Set([...required, ...optional]);
  const unknown = Object.keys(record).filter((key) => !allowed.has(key));
  const missing = required.filter((key) => !(key in record));
  if (unknown.length > 0 || missing.length > 0) {
    throw new ScenarioParseError([
      unknown.length > 0 ? `unknown fields ${unknown.sort().join(", ")}` : "",
      missing.length > 0 ? `missing fields ${missing.join(", ")}` : "",
    ].filter(Boolean).map((issue) => `${label} ${issue}`).join("; "));
  }
}
