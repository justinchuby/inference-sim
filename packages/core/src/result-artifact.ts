export const SIMULATION_RESULT_ARTIFACT_KIND = "inference-sim/result";
export const SIMULATION_RESULT_ARTIFACT_REVISION = 1;
const MAX_ARTIFACT_NESTING_DEPTH = 128;

export interface SimulationResultArtifact<TInput = unknown, TOutput = unknown> {
  readonly kind: typeof SIMULATION_RESULT_ARTIFACT_KIND;
  readonly revision: typeof SIMULATION_RESULT_ARTIFACT_REVISION;
  readonly runKind: string;
  readonly contracts: Readonly<Record<string, number>>;
  readonly inputFingerprint: string;
  readonly outputFingerprint: string;
  readonly artifactFingerprint: string;
  readonly input: TInput;
  readonly output: TOutput;
}

export class SimulationResultArtifactError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SimulationResultArtifactError";
  }
}

export function createSimulationResultArtifact<TInput, TOutput>(
  runKind: string,
  contracts: Readonly<Record<string, number>>,
  input: TInput,
  output: TOutput,
): SimulationResultArtifact<TInput, TOutput> {
  assertNonEmptyString(runKind, "artifact runKind");
  const normalizedContracts = normalizeContracts(contracts);
  const inputFingerprint = fingerprint(input);
  const outputFingerprint = fingerprint(output);
  const unsigned: Omit<
    SimulationResultArtifact<TInput, TOutput>,
    "artifactFingerprint"
  > = {
    kind: SIMULATION_RESULT_ARTIFACT_KIND,
    revision: SIMULATION_RESULT_ARTIFACT_REVISION,
    runKind,
    contracts: normalizedContracts,
    inputFingerprint,
    outputFingerprint,
    input,
    output,
  };
  return {
    ...unsigned,
    artifactFingerprint: fingerprint(unsigned),
  };
}

export function parseSimulationResultArtifact(
  input: unknown,
): SimulationResultArtifact {
  const artifact = requireRecord(input, "simulation result artifact");
  assertExactKeys(artifact, [
    "kind",
    "revision",
    "runKind",
    "contracts",
    "inputFingerprint",
    "outputFingerprint",
    "artifactFingerprint",
    "input",
    "output",
  ]);
  if (artifact.kind !== SIMULATION_RESULT_ARTIFACT_KIND) {
    throw new SimulationResultArtifactError(
      `unsupported simulation result artifact kind ${String(artifact.kind)}`,
    );
  }
  if (artifact.revision !== SIMULATION_RESULT_ARTIFACT_REVISION) {
    throw new SimulationResultArtifactError(
      `unsupported simulation result artifact revision ${String(artifact.revision)}`,
    );
  }
  assertNonEmptyString(artifact.runKind, "artifact runKind");
  const contracts = normalizeContracts(requireRecord(
    artifact.contracts,
    "artifact contracts",
  ) as Readonly<Record<string, number>>);
  const inputFingerprint = requireFingerprint(
    artifact.inputFingerprint,
    "artifact inputFingerprint",
  );
  const outputFingerprint = requireFingerprint(
    artifact.outputFingerprint,
    "artifact outputFingerprint",
  );
  const artifactFingerprint = requireFingerprint(
    artifact.artifactFingerprint,
    "artifact artifactFingerprint",
  );
  const expectedInputFingerprint = fingerprint(artifact.input);
  if (inputFingerprint !== expectedInputFingerprint) {
    throw new SimulationResultArtifactError(
      `artifact input fingerprint mismatch: expected ${expectedInputFingerprint}, got ${inputFingerprint}`,
    );
  }
  const expectedOutputFingerprint = fingerprint(artifact.output);
  if (outputFingerprint !== expectedOutputFingerprint) {
    throw new SimulationResultArtifactError(
      `artifact output fingerprint mismatch: expected ${expectedOutputFingerprint}, got ${outputFingerprint}`,
    );
  }
  const unsigned: Omit<
    SimulationResultArtifact,
    "artifactFingerprint"
  > = {
    kind: SIMULATION_RESULT_ARTIFACT_KIND,
    revision: SIMULATION_RESULT_ARTIFACT_REVISION,
    runKind: artifact.runKind,
    contracts,
    inputFingerprint,
    outputFingerprint,
    input: artifact.input,
    output: artifact.output,
  };
  const expectedArtifactFingerprint = fingerprint(unsigned);
  if (artifactFingerprint !== expectedArtifactFingerprint) {
    throw new SimulationResultArtifactError(
      `artifact fingerprint mismatch: expected ${expectedArtifactFingerprint}, got ${artifactFingerprint}`,
    );
  }
  return {
    ...unsigned,
    artifactFingerprint,
  };
}

export function serializeSimulationResultArtifact(
  artifact: SimulationResultArtifact,
  pretty = false,
): string {
  const validated = parseSimulationResultArtifact(artifact);
  return JSON.stringify(canonicalJsonValue(validated, "$"), null, pretty ? 2 : 0);
}

function normalizeContracts(
  contracts: Readonly<Record<string, number>>,
): Readonly<Record<string, number>> {
  const normalized: Record<string, number> = Object.create(null);
  for (const name of Object.keys(contracts).sort()) {
    assertNonEmptyString(name, "artifact contract name");
    const revision = contracts[name];
    if (!Number.isSafeInteger(revision) || revision <= 0) {
      throw new SimulationResultArtifactError(
        `artifact contract ${name} revision must be a positive safe integer`,
      );
    }
    normalized[name] = revision;
  }
  if (Object.keys(normalized).length === 0) {
    throw new SimulationResultArtifactError(
      "artifact contracts must not be empty",
    );
  }
  return normalized;
}

function fingerprint(value: unknown): string {
  const text = JSON.stringify(canonicalJsonValue(value, "$"));
  let hash = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(text)) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193);
  }
  return `fnv1a32:${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

type JsonValue =
  | null
  | boolean
  | number
  | string
  | readonly JsonValue[]
  | { readonly [key: string]: JsonValue };

function canonicalJsonValue(
  value: unknown,
  path: string,
  ancestors = new WeakSet<object>(),
  depth = 0,
): JsonValue {
  if (depth > MAX_ARTIFACT_NESTING_DEPTH) {
    throw new SimulationResultArtifactError(
      `${path} exceeds the maximum artifact nesting depth`,
    );
  }
  if (
    value === null
    || typeof value === "boolean"
    || typeof value === "string"
  ) {
    return value;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new SimulationResultArtifactError(
        `${path} must contain only finite JSON numbers`,
      );
    }
    return value;
  }
  if (Array.isArray(value)) {
    assertAcyclic(value, path, ancestors);
    try {
      const normalized: JsonValue[] = [];
      for (let index = 0; index < value.length; index++) {
        if (!(index in value)) {
          throw new SimulationResultArtifactError(
            `${path} must not contain sparse array entries`,
          );
        }
        normalized.push(canonicalJsonValue(
          value[index],
          `${path}[${index}]`,
          ancestors,
          depth + 1,
        ));
      }
      return normalized;
    } finally {
      ancestors.delete(value);
    }
  }
  if (typeof value !== "object") {
    throw new SimulationResultArtifactError(
      `${path} contains a non-JSON ${typeof value} value`,
    );
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new SimulationResultArtifactError(
      `${path} must contain only plain JSON objects`,
    );
  }
  assertAcyclic(value, path, ancestors);
  const record = value as Record<string, unknown>;
  const normalized: Record<string, JsonValue> = Object.create(null);
  try {
    for (const key of Object.keys(record).sort()) {
      const entry = record[key];
      if (entry !== undefined) {
        normalized[key] = canonicalJsonValue(
          entry,
          `${path}.${key}`,
          ancestors,
          depth + 1,
        );
      }
    }
  } finally {
    ancestors.delete(value);
  }
  return normalized;
}

function assertAcyclic(
  value: object,
  path: string,
  ancestors: WeakSet<object>,
): void {
  if (ancestors.has(value)) {
    throw new SimulationResultArtifactError(
      `${path} contains a circular reference`,
    );
  }
  ancestors.add(value);
}

function requireRecord(
  value: unknown,
  label: string,
): Record<string, unknown> {
  if (
    value === null
    || typeof value !== "object"
    || Array.isArray(value)
  ) {
    throw new SimulationResultArtifactError(`${label} must be an object`);
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new SimulationResultArtifactError(
      `${label} must be a plain object`,
    );
  }
  return value as Record<string, unknown>;
}

function assertExactKeys(
  record: Record<string, unknown>,
  expected: readonly string[],
): void {
  const expectedSet = new Set(expected);
  const unknown = Object.keys(record).filter((key) => !expectedSet.has(key));
  const missing = expected.filter((key) => !(key in record));
  if (unknown.length > 0 || missing.length > 0) {
    throw new SimulationResultArtifactError(
      [
        unknown.length > 0 ? `unknown fields ${unknown.sort().join(", ")}` : "",
        missing.length > 0 ? `missing fields ${missing.join(", ")}` : "",
      ].filter(Boolean).join("; "),
    );
  }
}

function assertNonEmptyString(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string" || value.length === 0) {
    throw new SimulationResultArtifactError(`${label} must be non-empty`);
  }
}

function requireFingerprint(value: unknown, label: string): string {
  if (
    typeof value !== "string"
    || !/^fnv1a32:[0-9a-f]{8}$/.test(value)
  ) {
    throw new SimulationResultArtifactError(
      `${label} must be an fnv1a32 fingerprint`,
    );
  }
  return value;
}
