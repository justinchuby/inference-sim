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
import { parseSimulationScenarioBoundary } from "./scenario-parser.js";

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
  const scenario = parseSimulationScenarioBoundary(
    artifact.scenario,
    "FrozenPlan artifact scenario",
    false,
  );
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
      ...(operation.componentId === undefined ? [] : ["componentId"]),
      ...(operation.pipelinePhase === undefined ? [] : ["pipelinePhase"]),
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
      ...(operation.componentId === undefined
        ? {}
        : {
            componentId: requireNonEmptyString(
              operation.componentId,
              `${label} componentId`,
            ),
          }),
      ...(operation.pipelinePhase === undefined
        ? {}
        : {
            pipelinePhase: requireNonEmptyString(
              operation.pipelinePhase,
              `${label} pipelinePhase`,
            ),
          }),
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
