import {
  PLAN_CONTRACT_REVISION,
  type FrozenPlan,
  type PlanOperation,
  type PlanStep,
} from "./plan-types.js";
import {
  SCENARIO_SCHEMA_VERSION,
  type ComputeCapability,
  type SimulationScenario,
} from "./scenario-types.js";
import { assertValidFrozenPlan } from "./frozen-plan.js";
import {
  canonicalJsonFingerprint,
  canonicalJsonStringify,
} from "./result-artifact.js";

export const FROZEN_PLAN_ARTIFACT_KIND = "inference-sim/frozen-plan";
export const FROZEN_PLAN_ARTIFACT_REVISION = 1;

export interface FrozenPlanArtifact {
  readonly kind: typeof FROZEN_PLAN_ARTIFACT_KIND;
  readonly revision: typeof FROZEN_PLAN_ARTIFACT_REVISION;
  readonly scenarioSchemaVersion: typeof SCENARIO_SCHEMA_VERSION;
  readonly planContractRevision: typeof PLAN_CONTRACT_REVISION;
  readonly scenarioFingerprint: string;
  readonly planFingerprint: string;
  readonly artifactFingerprint: string;
  readonly scenario: SimulationScenario;
  readonly plan: FrozenPlan;
}

export class FrozenPlanArtifactError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FrozenPlanArtifactError";
  }
}

export function createFrozenPlanArtifact(
  scenario: SimulationScenario,
  plan: FrozenPlan,
): FrozenPlanArtifact {
  assertPlanSemantics(scenario, plan);
  const scenarioFingerprint = canonicalJsonFingerprint(scenario);
  const planFingerprint = canonicalJsonFingerprint(plan);
  const unsigned = {
    kind: FROZEN_PLAN_ARTIFACT_KIND,
    revision: FROZEN_PLAN_ARTIFACT_REVISION,
    scenarioSchemaVersion: SCENARIO_SCHEMA_VERSION,
    planContractRevision: PLAN_CONTRACT_REVISION,
    scenarioFingerprint,
    planFingerprint,
    scenario,
    plan,
  } as const;
  return {
    ...unsigned,
    artifactFingerprint: canonicalJsonFingerprint(unsigned),
  };
}

export function parseFrozenPlanArtifact(input: unknown): FrozenPlanArtifact {
  const artifact = requireRecord(input, "FrozenPlan artifact");
  assertExactKeys(artifact, [
    "kind",
    "revision",
    "scenarioSchemaVersion",
    "planContractRevision",
    "scenarioFingerprint",
    "planFingerprint",
    "artifactFingerprint",
    "scenario",
    "plan",
  ], "FrozenPlan artifact");
  requireExactValue(
    artifact.kind,
    FROZEN_PLAN_ARTIFACT_KIND,
    "FrozenPlan artifact kind",
  );
  requireExactValue(
    artifact.revision,
    FROZEN_PLAN_ARTIFACT_REVISION,
    "FrozenPlan artifact revision",
  );
  requireExactValue(
    artifact.scenarioSchemaVersion,
    SCENARIO_SCHEMA_VERSION,
    "FrozenPlan artifact scenario schema version",
  );
  requireExactValue(
    artifact.planContractRevision,
    PLAN_CONTRACT_REVISION,
    "FrozenPlan artifact plan contract revision",
  );

  const scenarioFingerprint = requireFingerprint(
    artifact.scenarioFingerprint,
    "FrozenPlan artifact scenario fingerprint",
  );
  const planFingerprint = requireFingerprint(
    artifact.planFingerprint,
    "FrozenPlan artifact plan fingerprint",
  );
  const artifactFingerprint = requireFingerprint(
    artifact.artifactFingerprint,
    "FrozenPlan artifact fingerprint",
  );
  const scenario = parseScenarioBoundary(artifact.scenario);
  const plan = parsePlan(artifact.plan);

  assertFingerprint("scenario", scenarioFingerprint, scenario);
  assertFingerprint("plan", planFingerprint, plan);
  const unsigned = {
    kind: FROZEN_PLAN_ARTIFACT_KIND,
    revision: FROZEN_PLAN_ARTIFACT_REVISION,
    scenarioSchemaVersion: SCENARIO_SCHEMA_VERSION,
    planContractRevision: PLAN_CONTRACT_REVISION,
    scenarioFingerprint,
    planFingerprint,
    scenario,
    plan,
  } as const;
  assertFingerprint("artifact", artifactFingerprint, unsigned);
  assertPlanSemantics(scenario, plan);

  return {
    ...unsigned,
    artifactFingerprint,
  };
}

export function serializeFrozenPlanArtifact(
  artifact: FrozenPlanArtifact,
  pretty = false,
): string {
  return canonicalJsonStringify(parseFrozenPlanArtifact(artifact), pretty);
}

function parseScenarioBoundary(value: unknown): SimulationScenario {
  const scenario = requireRecord(value, "FrozenPlan artifact scenario");
  assertExactKeys(scenario, [
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
  ], "FrozenPlan artifact scenario");
  requireExactValue(
    scenario.schemaVersion,
    SCENARIO_SCHEMA_VERSION,
    "FrozenPlan artifact embedded scenario schema version",
  );
  requireNonEmptyString(scenario.id, "FrozenPlan artifact scenario id");
  for (const field of [
    "memoryDomains",
    "devices",
    "links",
    "placements",
    "transfers",
    "groups",
  ]) {
    requireArray(scenario[field], `FrozenPlan artifact scenario ${field}`);
  }
  requireRecord(scenario.workload, "FrozenPlan artifact scenario workload");
  requireRecord(scenario.execution, "FrozenPlan artifact scenario execution");
  requireRecord(
    scenario.calibration,
    "FrozenPlan artifact scenario calibration",
  );
  assertScenarioMembers(scenario);
  return scenario as unknown as SimulationScenario;
}

function assertScenarioMembers(scenario: Record<string, unknown>): void {
  requireRecordArray(
    scenario.memoryDomains,
    "FrozenPlan artifact scenario memoryDomains",
  ).forEach((domain, index) => {
    const label = `FrozenPlan artifact scenario memoryDomains[${index}]`;
    assertExactKeys(domain, [
      "id",
      "nodeId",
      "kind",
      "capacityBytes",
      "bandwidthBytesPerSec",
      "latencyNs",
      "coherent",
      "allocationClasses",
      "accessibleBy",
      "governor",
      "provenance",
    ], label);
    requireStrings(domain, ["id", "nodeId", "kind"], label);
    requireNumbers(
      domain,
      ["capacityBytes", "bandwidthBytesPerSec", "latencyNs"],
      label,
    );
    requireBoolean(domain.coherent, `${label} coherent`);
    requireStringArray(domain.allocationClasses, `${label} allocationClasses`);
    requireStringArray(domain.accessibleBy, `${label} accessibleBy`);
    assertGovernor(domain.governor, `${label} governor`);
    assertProvenance(domain.provenance, `${label} provenance`);
  });

  requireRecordArray(
    scenario.devices,
    "FrozenPlan artifact scenario devices",
  ).forEach((device, index) => {
    const label = `FrozenPlan artifact scenario devices[${index}]`;
    assertExactKeys(device, [
      "id",
      "nodeId",
      "kind",
      "executionProvider",
      "memoryDomainIds",
      "capabilities",
      "supportedDtypes",
      "maxConcurrentCompute",
      "provenance",
    ], label);
    requireStrings(
      device,
      ["id", "nodeId", "kind", "executionProvider"],
      label,
    );
    requireStringArray(device.memoryDomainIds, `${label} memoryDomainIds`);
    requireStringArray(device.capabilities, `${label} capabilities`);
    requireStringArray(device.supportedDtypes, `${label} supportedDtypes`);
    requireFiniteNumber(
      device.maxConcurrentCompute,
      `${label} maxConcurrentCompute`,
    );
    assertProvenance(device.provenance, `${label} provenance`);
  });

  requireRecordArray(
    scenario.links,
    "FrozenPlan artifact scenario links",
  ).forEach((link, index) => {
    const label = `FrozenPlan artifact scenario links[${index}]`;
    assertExactKeys(link, [
      "id",
      "sourceDomainId",
      "targetDomainId",
      "kind",
      "bandwidthBytesPerSec",
      "latencyNs",
      "concurrencyLanes",
      "provenance",
    ], label);
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
    assertProvenance(link.provenance, `${label} provenance`);
  });

  requireRecordArray(
    scenario.placements,
    "FrozenPlan artifact scenario placements",
  ).forEach((placement, index) => {
    const label = `FrozenPlan artifact scenario placements[${index}]`;
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
        assertExactKeys(allocation, [
          "physicalAllocationId",
          "domainId",
          "bytes",
          "allocationClass",
          "purpose",
        ], allocationLabel);
        requireStrings(allocation, [
          "physicalAllocationId",
          "domainId",
          "allocationClass",
          "purpose",
        ], allocationLabel);
        requireFiniteNumber(allocation.bytes, `${allocationLabel} bytes`);
      });
  });

  requireRecordArray(
    scenario.transfers,
    "FrozenPlan artifact scenario transfers",
  ).forEach((transfer, index) => {
    const label = `FrozenPlan artifact scenario transfers[${index}]`;
    assertExactKeys(transfer, [
      "id",
      "sourceDomainId",
      "targetDomainId",
      "bytes",
      "requiresPinnedStaging",
      "stagingAllocationIds",
    ], label);
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

  requireRecordArray(
    scenario.groups,
    "FrozenPlan artifact scenario groups",
  ).forEach((group, index) => {
    const label = `FrozenPlan artifact scenario groups[${index}]`;
    assertExactKeys(group, ["id", "orderedRanks"], label);
    requireNonEmptyString(group.id, `${label} id`);
    requireRecordArray(group.orderedRanks, `${label} orderedRanks`)
      .forEach((rank, rankIndex) => {
        const rankLabel = `${label} orderedRanks[${rankIndex}]`;
        assertExactKeys(rank, ["rankId", "deviceId"], rankLabel);
        requireStrings(rank, ["rankId", "deviceId"], rankLabel);
      });
  });

  const workload = requireRecord(
    scenario.workload,
    "FrozenPlan artifact scenario workload",
  );
  assertKeys(
    workload,
    ["batchSize", "inputSequenceLength", "outputSequenceLength"],
    ["speculative"],
    "FrozenPlan artifact scenario workload",
  );
  requireNumbers(
    workload,
    ["batchSize", "inputSequenceLength", "outputSequenceLength"],
    "FrozenPlan artifact scenario workload",
  );
  if (workload.speculative !== undefined) {
    const speculative = requireRecord(
      workload.speculative,
      "FrozenPlan artifact scenario workload speculative",
    );
    assertExactKeys(
      speculative,
      ["family", "maxAdditionalTokens"],
      "FrozenPlan artifact scenario workload speculative",
    );
    requireNonEmptyString(
      speculative.family,
      "FrozenPlan artifact scenario workload speculative family",
    );
    requireFiniteNumber(
      speculative.maxAdditionalTokens,
      "FrozenPlan artifact scenario workload speculative maxAdditionalTokens",
    );
  }

  const execution = requireRecord(
    scenario.execution,
    "FrozenPlan artifact scenario execution",
  );
  assertExactKeys(execution, [
    "topologyEpoch",
    "seed",
    "maxEvents",
    "parallelism",
  ], "FrozenPlan artifact scenario execution");
  requireNumbers(
    execution,
    ["topologyEpoch", "seed", "maxEvents"],
    "FrozenPlan artifact scenario execution",
  );
  const parallelism = requireRecord(
    execution.parallelism,
    "FrozenPlan artifact scenario execution parallelism",
  );
  assertExactKeys(parallelism, [
    "composition",
    "tensor",
    "pipeline",
    "expert",
    "data",
  ], "FrozenPlan artifact scenario execution parallelism");
  requireNonEmptyString(
    parallelism.composition,
    "FrozenPlan artifact scenario execution parallelism composition",
  );
  requireNumbers(
    parallelism,
    ["tensor", "pipeline", "expert", "data"],
    "FrozenPlan artifact scenario execution parallelism",
  );

  const calibration = requireRecord(
    scenario.calibration,
    "FrozenPlan artifact scenario calibration",
  );
  assertExactKeys(
    calibration,
    ["coefficients"],
    "FrozenPlan artifact scenario calibration",
  );
  requireRecordArray(
    calibration.coefficients,
    "FrozenPlan artifact scenario calibration coefficients",
  ).forEach((coefficient, index) => {
    const label =
      `FrozenPlan artifact scenario calibration coefficients[${index}]`;
    assertExactKeys(
      coefficient,
      ["id", "value", "unit", "provenance"],
      label,
    );
    requireStrings(coefficient, ["id", "unit"], label);
    requireFiniteNumber(coefficient.value, `${label} value`);
    assertProvenance(coefficient.provenance, `${label} provenance`);
  });
}

function assertGovernor(value: unknown, label: string): void {
  const governor = requireRecord(value, label);
  const kind = requireNonEmptyString(governor.kind, `${label} kind`);
  if (kind === "host") {
    assertExactKeys(governor, ["kind", "nodeId"], label);
    requireNonEmptyString(governor.nodeId, `${label} nodeId`);
  } else if (kind === "device") {
    assertExactKeys(governor, ["kind", "deviceId"], label);
    requireNonEmptyString(governor.deviceId, `${label} deviceId`);
  } else if (kind === "none") {
    assertExactKeys(governor, ["kind"], label);
  } else {
    throw new FrozenPlanArtifactError(`${label} kind is unsupported`);
  }
}

function assertProvenance(value: unknown, label: string): void {
  const provenance = requireRecord(value, label);
  assertKeys(
    provenance,
    ["confidence", "source"],
    ["measuredAt", "notes"],
    label,
  );
  requireStrings(provenance, ["confidence", "source"], label);
  if (provenance.measuredAt !== undefined) {
    requireNonEmptyString(provenance.measuredAt, `${label} measuredAt`);
  }
  if (provenance.notes !== undefined) {
    requireNonEmptyString(provenance.notes, `${label} notes`);
  }
}

function parsePlan(value: unknown): FrozenPlan {
  const plan = requireRecord(value, "FrozenPlan artifact plan");
  assertExactKeys(plan, [
    "contractRevision",
    "id",
    "executionId",
    "topologyEpoch",
    "steps",
  ], "FrozenPlan artifact plan");
  requireExactValue(
    plan.contractRevision,
    PLAN_CONTRACT_REVISION,
    "FrozenPlan contract revision",
  );
  const id = requireNonEmptyString(plan.id, "FrozenPlan id");
  const executionId = requireNonEmptyString(
    plan.executionId,
    "FrozenPlan execution id",
  );
  const topologyEpoch = requireNonNegativeSafeInteger(
    plan.topologyEpoch,
    "FrozenPlan topology epoch",
  );
  const steps = requireArray(plan.steps, "FrozenPlan steps")
    .map((step, index) => parseStep(step, index));
  return {
    contractRevision: PLAN_CONTRACT_REVISION,
    id,
    executionId,
    topologyEpoch,
    steps,
  };
}

function parseStep(value: unknown, index: number): PlanStep {
  const label = `FrozenPlan steps[${index}]`;
  const step = requireRecord(value, label);
  assertExactKeys(step, [
    "id",
    "participants",
    "dependencies",
    "reads",
    "writes",
    "operation",
  ], label);
  return {
    id: requireNonNegativeSafeInteger(step.id, `${label} id`),
    participants: requireStringArray(step.participants, `${label} participants`),
    dependencies: requireIntegerArray(
      step.dependencies,
      `${label} dependencies`,
    ),
    reads: requireStringArray(step.reads, `${label} reads`),
    writes: requireStringArray(step.writes, `${label} writes`),
    operation: parseOperation(step.operation, `${label} operation`),
  };
}

function parseOperation(value: unknown, label: string): PlanOperation {
  const operation = requireRecord(value, label);
  const kind = operation.kind;
  if (kind === "compute") {
    assertExactKeys(operation, [
      "kind",
      "deviceId",
      "capability",
      "durationNs",
    ], label);
    return {
      kind,
      deviceId: requireNonEmptyString(operation.deviceId, `${label} deviceId`),
      capability: requireComputeCapability(
        operation.capability,
        `${label} capability`,
      ),
      durationNs: requireNonNegativeSafeInteger(
        operation.durationNs,
        `${label} durationNs`,
      ),
    };
  }
  if (kind === "transfer") {
    assertExactKeys(operation, ["kind", "linkId", "durationNs"], label);
    return {
      kind,
      linkId: requireNonEmptyString(operation.linkId, `${label} linkId`),
      durationNs: requireNonNegativeSafeInteger(
        operation.durationNs,
        `${label} durationNs`,
      ),
    };
  }
  if (kind === "collective") {
    assertExactKeys(operation, [
      "kind",
      "groupId",
      "commSequenceId",
      "algorithm",
      "linkIds",
      "durationNs",
    ], label);
    const algorithm = operation.algorithm;
    if (algorithm !== "all_reduce_ring" && algorithm !== "all_to_all_v") {
      throw new FrozenPlanArtifactError(
        `${label} algorithm must be all_reduce_ring or all_to_all_v`,
      );
    }
    return {
      kind,
      groupId: requireNonEmptyString(operation.groupId, `${label} groupId`),
      commSequenceId: requireNonNegativeSafeInteger(
        operation.commSequenceId,
        `${label} commSequenceId`,
      ),
      algorithm,
      linkIds: requireStringArray(operation.linkIds, `${label} linkIds`),
      durationNs: requireNonNegativeSafeInteger(
        operation.durationNs,
        `${label} durationNs`,
      ),
    };
  }
  throw new FrozenPlanArtifactError(
    `${label} kind must be compute, transfer, or collective`,
  );
}

function assertPlanSemantics(
  scenario: SimulationScenario,
  plan: FrozenPlan,
): void {
  try {
    assertValidFrozenPlan(scenario, plan);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new FrozenPlanArtifactError(
      `FrozenPlan artifact semantic validation failed: ${message}`,
    );
  }
}

function assertFingerprint(
  label: string,
  actual: string,
  value: unknown,
): void {
  const expected = canonicalJsonFingerprint(value);
  if (actual !== expected) {
    throw new FrozenPlanArtifactError(
      `FrozenPlan artifact ${label} fingerprint mismatch: expected ${expected}, got ${actual}`,
    );
  }
}

function requireRecord(
  value: unknown,
  label: string,
): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new FrozenPlanArtifactError(`${label} must be an object`);
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new FrozenPlanArtifactError(`${label} must be a plain object`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new FrozenPlanArtifactError(`${label} must be an array`);
  }
  return value;
}

function requireRecordArray(
  value: unknown,
  label: string,
): Record<string, unknown>[] {
  return requireArray(value, label).map((entry, index) => (
    requireRecord(entry, `${label}[${index}]`)
  ));
}

function requireStringArray(value: unknown, label: string): string[] {
  const array = requireArray(value, label);
  return array.map((entry, index) => (
    requireNonEmptyString(entry, `${label}[${index}]`)
  ));
}

function requireIntegerArray(value: unknown, label: string): number[] {
  const array = requireArray(value, label);
  return array.map((entry, index) => (
    requireNonNegativeSafeInteger(entry, `${label}[${index}]`)
  ));
}

function requireNonEmptyString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new FrozenPlanArtifactError(`${label} must be a non-empty string`);
  }
  return value;
}

function requireFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new FrozenPlanArtifactError(`${label} must be a finite number`);
  }
  return value;
}

function requireBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") {
    throw new FrozenPlanArtifactError(`${label} must be a boolean`);
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

function requireNonNegativeSafeInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new FrozenPlanArtifactError(
      `${label} must be a non-negative safe integer`,
    );
  }
  return value as number;
}

function requireComputeCapability(
  value: unknown,
  label: string,
): ComputeCapability {
  const capabilities = [
    "attention",
    "ffn",
    "collective",
    "copy",
    "sampling",
    "draft",
    "lookup",
  ] as const;
  if (!capabilities.includes(value as typeof capabilities[number])) {
    throw new FrozenPlanArtifactError(`${label} is unsupported`);
  }
  return value as typeof capabilities[number];
}

function requireFingerprint(value: unknown, label: string): string {
  if (
    typeof value !== "string"
    || !/^fnv1a32:[0-9a-f]{8}$/.test(value)
  ) {
    throw new FrozenPlanArtifactError(
      `${label} must be an fnv1a32 fingerprint`,
    );
  }
  return value;
}

function requireExactValue(
  value: unknown,
  expected: string | number,
  label: string,
): void {
  if (value !== expected) {
    throw new FrozenPlanArtifactError(
      `${label} must be ${String(expected)}, got ${String(value)}`,
    );
  }
}

function assertExactKeys(
  record: Record<string, unknown>,
  expected: readonly string[],
  label: string,
): void {
  assertKeys(record, expected, [], label);
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
    throw new FrozenPlanArtifactError([
      unknown.length > 0 ? `unknown fields ${unknown.sort().join(", ")}` : "",
      missing.length > 0 ? `missing fields ${missing.join(", ")}` : "",
    ].filter(Boolean).map((issue) => `${label} ${issue}`).join("; "));
  }
}
