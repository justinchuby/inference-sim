import {
  canonicalJsonFingerprint,
  canonicalJsonStringify,
} from "./result-artifact.js";

export const ONNX_MODEL_MANIFEST_KIND = "inference-sim/onnx-model";
export const ONNX_MODEL_MANIFEST_REVISION = 2;

export interface OnnxTensorStorage {
  readonly kind: "inline" | "external";
  readonly byteLength: number;
  readonly location?: string;
  readonly offset?: number;
}

export interface OnnxInitializerManifest {
  readonly name: string;
  readonly dataType: string;
  readonly dimensions: readonly number[];
  readonly elementCount: number;
  readonly logicalByteLength: number;
  readonly storage: OnnxTensorStorage;
}

export interface OnnxExternalDataFileManifest {
  readonly location: string;
  readonly byteLength: number;
  readonly referencedByteLength: number;
  readonly sha256: string;
}

export interface OnnxOperatorCount {
  readonly domain: string;
  readonly opType: string;
  readonly count: number;
}

export interface OnnxArchitectureEvidence {
  readonly source:
    | "onnx_genai_manifest"
    | "genai_config"
    | "inference_metadata"
    | "none";
  readonly modelType?: string;
  readonly hiddenSize?: number;
  readonly intermediateSize?: number;
  readonly numHiddenLayers?: number;
  readonly numAttentionHeads?: number;
  readonly numKeyValueHeads?: number;
  readonly headDimension?: number;
  readonly vocabSize?: number;
  readonly numExperts?: number;
  readonly activeExpertsPerToken?: number;
  readonly expertBytesPerLayer?: number;
  readonly sharedExpertBytesPerLayer?: number;
}

export interface OnnxModelManifestUnsigned {
  readonly kind: typeof ONNX_MODEL_MANIFEST_KIND;
  readonly revision: typeof ONNX_MODEL_MANIFEST_REVISION;
  readonly source: {
    readonly modelFileName: string;
    readonly modelByteLength: number;
    readonly sha256: string;
  };
  readonly model: {
    readonly irVersion: string;
    readonly producerName: string;
    readonly producerVersion: string;
    readonly domain: string;
    readonly modelVersion: string;
  };
  readonly graph: {
    readonly name: string;
    readonly nodeCount: number;
    readonly initializerCount: number;
    readonly inputNames: readonly string[];
    readonly outputNames: readonly string[];
    readonly operators: readonly OnnxOperatorCount[];
  };
  readonly initializers: readonly OnnxInitializerManifest[];
  readonly externalDataFiles: readonly OnnxExternalDataFileManifest[];
  readonly architecture: OnnxArchitectureEvidence;
  readonly totals: {
    readonly initializerElements: number;
    readonly initializerLogicalBytes: number;
    readonly inlineInitializerBytes: number;
    readonly externalInitializerBytes: number;
  };
  readonly profileReadiness: {
    readonly ready: boolean;
    readonly missingFields: readonly string[];
  };
}

export interface OnnxModelManifest extends OnnxModelManifestUnsigned {
  readonly manifestFingerprint: string;
}

export class OnnxModelManifestError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "OnnxModelManifestError";
  }
}

export function createOnnxModelManifest(
  input: OnnxModelManifestUnsigned,
): OnnxModelManifest {
  const unsigned = parseUnsigned(input);
  return {
    ...unsigned,
    manifestFingerprint: canonicalJsonFingerprint(unsigned),
  };
}

export function parseOnnxModelManifest(input: unknown): OnnxModelManifest {
  const manifest = requireRecord(input, "ONNX model manifest");
  assertExactKeys(manifest, [
    "kind",
    "revision",
    "source",
    "model",
    "graph",
    "initializers",
    "externalDataFiles",
    "architecture",
    "totals",
    "profileReadiness",
    "manifestFingerprint",
  ], "ONNX model manifest");
  const unsigned = parseUnsigned(manifest);
  const manifestFingerprint = requireFingerprint(
    manifest.manifestFingerprint,
    "ONNX model manifest fingerprint",
  );
  const expected = canonicalJsonFingerprint(unsigned);
  if (manifestFingerprint !== expected) {
    throw new OnnxModelManifestError(
      `ONNX model manifest fingerprint mismatch: expected ${expected}, got ${manifestFingerprint}`,
    );
  }
  return { ...unsigned, manifestFingerprint };
}

export function serializeOnnxModelManifest(
  manifest: OnnxModelManifest,
  pretty = false,
): string {
  return canonicalJsonStringify(parseOnnxModelManifest(manifest), pretty);
}

function parseUnsigned(input: unknown): OnnxModelManifestUnsigned {
  const manifest = requireRecord(input, "ONNX model manifest");
  const allowed = [
    "kind",
    "revision",
    "source",
    "model",
    "graph",
    "initializers",
    "externalDataFiles",
    "architecture",
    "totals",
    "profileReadiness",
    "manifestFingerprint",
  ];
  assertAllowedKeys(manifest, allowed, "ONNX model manifest");
  requireExact(
    manifest.kind,
    ONNX_MODEL_MANIFEST_KIND,
    "ONNX model manifest kind",
  );
  requireExact(
    manifest.revision,
    ONNX_MODEL_MANIFEST_REVISION,
    "ONNX model manifest revision",
  );

  const source = parseSource(manifest.source);
  const model = parseModel(manifest.model);
  const graph = parseGraph(manifest.graph);
  const initializers = requireArray(
    manifest.initializers,
    "ONNX model manifest initializers",
  ).map(parseInitializer);
  const externalDataFiles = requireArray(
    manifest.externalDataFiles,
    "ONNX model manifest externalDataFiles",
  ).map(parseExternalFile);
  const architecture = parseArchitecture(manifest.architecture);
  const totals = parseTotals(manifest.totals);
  const profileReadiness = parseReadiness(manifest.profileReadiness);

  assertUnique(initializers.map((tensor) => tensor.name), "initializer names");
  assertUnique(
    externalDataFiles.map((file) => file.location),
    "external-data locations",
  );
  const computedTotals = {
    initializerElements: checkedSum(
      initializers.map((tensor) => tensor.elementCount),
      "initializer element total",
    ),
    initializerLogicalBytes: checkedSum(
      initializers.map((tensor) => tensor.logicalByteLength),
      "initializer logical byte total",
    ),
    inlineInitializerBytes: checkedSum(
      initializers.filter((tensor) => tensor.storage.kind === "inline")
        .map((tensor) => tensor.storage.byteLength),
      "inline initializer byte total",
    ),
    externalInitializerBytes: checkedSum(
      initializers.filter((tensor) => tensor.storage.kind === "external")
        .map((tensor) => tensor.storage.byteLength),
      "external initializer byte total",
    ),
  };
  if (JSON.stringify(totals) !== JSON.stringify(computedTotals)) {
    throw new OnnxModelManifestError(
      "ONNX model manifest totals do not match initializer inventory",
    );
  }
  if (graph.initializerCount !== initializers.length) {
    throw new OnnxModelManifestError(
      "ONNX model manifest graph initializer count does not match inventory",
    );
  }
  const expectedMissing = requiredArchitectureFields(architecture)
    .filter((field) => architecture[field] === undefined)
    .sort();
  if (
    profileReadiness.ready !== (expectedMissing.length === 0)
    || JSON.stringify(profileReadiness.missingFields)
      !== JSON.stringify(expectedMissing)
  ) {
    throw new OnnxModelManifestError(
      "ONNX model manifest profile readiness does not match architecture evidence",
    );
  }

  return {
    kind: ONNX_MODEL_MANIFEST_KIND,
    revision: ONNX_MODEL_MANIFEST_REVISION,
    source,
    model,
    graph,
    initializers,
    externalDataFiles,
    architecture,
    totals,
    profileReadiness,
  };
}

function requiredArchitectureFields(
  architecture: OnnxArchitectureEvidence,
): ReadonlyArray<keyof OnnxArchitectureEvidence> {
  const denseFields: ReadonlyArray<keyof OnnxArchitectureEvidence> = [
    "hiddenSize",
    "intermediateSize",
    "numHiddenLayers",
    "numAttentionHeads",
    "numKeyValueHeads",
    "headDimension",
    "vocabSize",
  ];
  return architecture.numExperts === undefined
    ? denseFields
    : [
        ...denseFields,
        "activeExpertsPerToken",
        "expertBytesPerLayer",
        "sharedExpertBytesPerLayer",
      ];
}

function parseSource(value: unknown): OnnxModelManifestUnsigned["source"] {
  const source = requireRecord(value, "ONNX model manifest source");
  assertExactKeys(source, [
    "modelFileName",
    "modelByteLength",
    "sha256",
  ], "ONNX model manifest source");
  return {
    modelFileName: requireString(source.modelFileName, "ONNX model file name"),
    modelByteLength: requireNonNegativeInteger(
      source.modelByteLength,
      "ONNX model byte length",
    ),
    sha256: requireSha256(source.sha256, "ONNX model SHA-256"),
  };
}

function parseModel(value: unknown): OnnxModelManifestUnsigned["model"] {
  const model = requireRecord(value, "ONNX model manifest model");
  assertExactKeys(model, [
    "irVersion",
    "producerName",
    "producerVersion",
    "domain",
    "modelVersion",
  ], "ONNX model manifest model");
  return {
    irVersion: requireUnsignedIntegerString(
      model.irVersion,
      "ONNX IR version",
    ),
    producerName: requireStringValue(model.producerName, "ONNX producer name"),
    producerVersion: requireStringValue(
      model.producerVersion,
      "ONNX producer version",
    ),
    domain: requireStringValue(model.domain, "ONNX model domain"),
    modelVersion: requireUnsignedIntegerString(
      model.modelVersion,
      "ONNX model version",
    ),
  };
}

function parseGraph(value: unknown): OnnxModelManifestUnsigned["graph"] {
  const graph = requireRecord(value, "ONNX model manifest graph");
  assertExactKeys(graph, [
    "name",
    "nodeCount",
    "initializerCount",
    "inputNames",
    "outputNames",
    "operators",
  ], "ONNX model manifest graph");
  const operators = requireArray(
    graph.operators,
    "ONNX model manifest operators",
  ).map((entry, index): OnnxOperatorCount => {
    const operator = requireRecord(entry, `ONNX operator[${index}]`);
    assertExactKeys(
      operator,
      ["domain", "opType", "count"],
      `ONNX operator[${index}]`,
    );
    return {
      domain: requireString(operator.domain, `ONNX operator[${index}] domain`),
      opType: requireString(operator.opType, `ONNX operator[${index}] type`),
      count: requirePositiveInteger(
        operator.count,
        `ONNX operator[${index}] count`,
      ),
    };
  });
  const sorted = [...operators].sort((left, right) => (
    left.domain.localeCompare(right.domain)
    || left.opType.localeCompare(right.opType)
  ));
  if (JSON.stringify(operators) !== JSON.stringify(sorted)) {
    throw new OnnxModelManifestError(
      "ONNX operator inventory must be canonically sorted",
    );
  }
  assertUnique(
    operators.map((operator) => `${operator.domain}\0${operator.opType}`),
    "ONNX operators",
  );
  const nodeCount = requireNonNegativeInteger(
    graph.nodeCount,
    "ONNX graph node count",
  );
  if (
    checkedSum(operators.map((operator) => operator.count), "operator count")
      !== nodeCount
  ) {
    throw new OnnxModelManifestError(
      "ONNX graph node count does not match operator inventory",
    );
  }
  return {
    name: requireStringValue(graph.name, "ONNX graph name"),
    nodeCount,
    initializerCount: requireNonNegativeInteger(
      graph.initializerCount,
      "ONNX graph initializer count",
    ),
    inputNames: requireStringArray(graph.inputNames, "ONNX graph input names"),
    outputNames: requireStringArray(
      graph.outputNames,
      "ONNX graph output names",
    ),
    operators,
  };
}

function parseInitializer(
  value: unknown,
  index: number,
): OnnxInitializerManifest {
  const label = `ONNX initializer[${index}]`;
  const initializer = requireRecord(value, label);
  assertExactKeys(initializer, [
    "name",
    "dataType",
    "dimensions",
    "elementCount",
    "logicalByteLength",
    "storage",
  ], label);
  const dimensions = requireArray(initializer.dimensions, `${label} dimensions`)
    .map((dimension, dimensionIndex) => requireNonNegativeInteger(
      dimension,
      `${label} dimensions[${dimensionIndex}]`,
    ));
  const storageRecord = requireRecord(
    initializer.storage,
    `${label} storage`,
  );
  const kind = storageRecord.kind;
  if (kind === "inline") {
    assertExactKeys(
      storageRecord,
      ["kind", "byteLength"],
      `${label} storage`,
    );
  } else if (kind === "external") {
    assertExactKeys(
      storageRecord,
      ["kind", "byteLength", "location", "offset"],
      `${label} storage`,
    );
  } else {
    throw new OnnxModelManifestError(
      `${label} storage kind must be inline or external`,
    );
  }
  const storage: OnnxTensorStorage = kind === "inline"
    ? {
        kind,
        byteLength: requireNonNegativeInteger(
          storageRecord.byteLength,
          `${label} storage byteLength`,
        ),
      }
    : {
        kind,
        byteLength: requireNonNegativeInteger(
          storageRecord.byteLength,
          `${label} storage byteLength`,
        ),
        location: requireSafeRelativePath(
          storageRecord.location,
          `${label} external location`,
        ),
        offset: requireNonNegativeInteger(
          storageRecord.offset,
          `${label} external offset`,
        ),
      };
  return {
    name: requireString(initializer.name, `${label} name`),
    dataType: requireString(initializer.dataType, `${label} dataType`),
    dimensions,
    elementCount: requireNonNegativeInteger(
      initializer.elementCount,
      `${label} elementCount`,
    ),
    logicalByteLength: requireNonNegativeInteger(
      initializer.logicalByteLength,
      `${label} logicalByteLength`,
    ),
    storage,
  };
}

function parseExternalFile(
  value: unknown,
  index: number,
): OnnxExternalDataFileManifest {
  const label = `ONNX externalDataFiles[${index}]`;
  const file = requireRecord(value, label);
  assertExactKeys(file, [
    "location",
    "byteLength",
    "referencedByteLength",
    "sha256",
  ], label);
  return {
    location: requireSafeRelativePath(file.location, `${label} location`),
    byteLength: requireNonNegativeInteger(
      file.byteLength,
      `${label} byteLength`,
    ),
    referencedByteLength: requireNonNegativeInteger(
      file.referencedByteLength,
      `${label} referencedByteLength`,
    ),
    sha256: requireSha256(file.sha256, `${label} SHA-256`),
  };
}

function parseArchitecture(value: unknown): OnnxArchitectureEvidence {
  const architecture = requireRecord(value, "ONNX architecture evidence");
  assertAllowedKeys(architecture, [
    "source",
    "modelType",
    "hiddenSize",
    "intermediateSize",
    "numHiddenLayers",
    "numAttentionHeads",
    "numKeyValueHeads",
    "headDimension",
    "vocabSize",
    "numExperts",
    "activeExpertsPerToken",
    "expertBytesPerLayer",
    "sharedExpertBytesPerLayer",
  ], "ONNX architecture evidence");
  const source = architecture.source;
  const sources: readonly OnnxArchitectureEvidence["source"][] = [
    "onnx_genai_manifest",
    "genai_config",
    "inference_metadata",
    "none",
  ];
  if (!sources.includes(source as OnnxArchitectureEvidence["source"])) {
    throw new OnnxModelManifestError(
      "ONNX architecture evidence source is unsupported",
    );
  }
  const result: Record<string, unknown> = { source };
  if (architecture.modelType !== undefined) {
    result.modelType = requireString(
      architecture.modelType,
      "ONNX architecture modelType",
    );
  }
  for (const field of [
    "hiddenSize",
    "intermediateSize",
    "numHiddenLayers",
    "numAttentionHeads",
    "numKeyValueHeads",
    "headDimension",
    "vocabSize",
    "numExperts",
    "activeExpertsPerToken",
  ] as const) {
    if (architecture[field] !== undefined) {
      result[field] = requirePositiveInteger(
        architecture[field],
        `ONNX architecture ${field}`,
      );
    }
  }
  if (architecture.expertBytesPerLayer !== undefined) {
    result.expertBytesPerLayer = requirePositiveInteger(
      architecture.expertBytesPerLayer,
      "ONNX architecture expertBytesPerLayer",
    );
  }
  if (architecture.sharedExpertBytesPerLayer !== undefined) {
    result.sharedExpertBytesPerLayer = requireNonNegativeInteger(
      architecture.sharedExpertBytesPerLayer,
      "ONNX architecture sharedExpertBytesPerLayer",
    );
  }
  return result as unknown as OnnxArchitectureEvidence;
}

function parseTotals(value: unknown): OnnxModelManifestUnsigned["totals"] {
  const totals = requireRecord(value, "ONNX model manifest totals");
  assertExactKeys(totals, [
    "initializerElements",
    "initializerLogicalBytes",
    "inlineInitializerBytes",
    "externalInitializerBytes",
  ], "ONNX model manifest totals");
  return {
    initializerElements: requireNonNegativeInteger(
      totals.initializerElements,
      "initializer element total",
    ),
    initializerLogicalBytes: requireNonNegativeInteger(
      totals.initializerLogicalBytes,
      "initializer logical byte total",
    ),
    inlineInitializerBytes: requireNonNegativeInteger(
      totals.inlineInitializerBytes,
      "inline initializer byte total",
    ),
    externalInitializerBytes: requireNonNegativeInteger(
      totals.externalInitializerBytes,
      "external initializer byte total",
    ),
  };
}

function parseReadiness(
  value: unknown,
): OnnxModelManifestUnsigned["profileReadiness"] {
  const readiness = requireRecord(value, "ONNX profile readiness");
  assertExactKeys(
    readiness,
    ["ready", "missingFields"],
    "ONNX profile readiness",
  );
  if (typeof readiness.ready !== "boolean") {
    throw new OnnxModelManifestError("ONNX profile readiness must be boolean");
  }
  const missingFields = requireStringArray(
    readiness.missingFields,
    "ONNX profile missing fields",
  ).sort();
  assertUnique(missingFields, "ONNX profile missing fields");
  return { ready: readiness.ready, missingFields };
}

function checkedSum(values: readonly number[], label: string): number {
  let sum = 0;
  for (const value of values) {
    sum += value;
    if (!Number.isSafeInteger(sum)) {
      throw new OnnxModelManifestError(`${label} exceeds safe integer range`);
    }
  }
  return sum;
}

function requireRecord(
  value: unknown,
  label: string,
): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new OnnxModelManifestError(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new OnnxModelManifestError(`${label} must be an array`);
  }
  return value;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new OnnxModelManifestError(`${label} must be a non-empty string`);
  }
  return value;
}

function requireStringValue(value: unknown, label: string): string {
  if (typeof value !== "string") {
    throw new OnnxModelManifestError(`${label} must be a string`);
  }
  return value;
}

function requireStringArray(value: unknown, label: string): string[] {
  return requireArray(value, label).map((entry, index) => (
    requireString(entry, `${label}[${index}]`)
  ));
}

function requireNonNegativeInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new OnnxModelManifestError(
      `${label} must be a non-negative safe integer`,
    );
  }
  return value as number;
}

function requirePositiveInteger(value: unknown, label: string): number {
  const result = requireNonNegativeInteger(value, label);
  if (result === 0) {
    throw new OnnxModelManifestError(`${label} must be positive`);
  }
  return result;
}

function requireUnsignedIntegerString(value: unknown, label: string): string {
  if (typeof value !== "string" || !/^(0|[1-9]\d*)$/.test(value)) {
    throw new OnnxModelManifestError(
      `${label} must be an unsigned integer string`,
    );
  }
  return value;
}

function requireSha256(value: unknown, label: string): string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) {
    throw new OnnxModelManifestError(`${label} must be a lowercase SHA-256`);
  }
  return value;
}

function requireFingerprint(value: unknown, label: string): string {
  if (
    typeof value !== "string"
    || !/^fnv1a32:[0-9a-f]{8}$/.test(value)
  ) {
    throw new OnnxModelManifestError(
      `${label} must be an fnv1a32 fingerprint`,
    );
  }
  return value;
}

function requireSafeRelativePath(value: unknown, label: string): string {
  const path = requireString(value, label);
  if (
    path.startsWith("/")
    || path.startsWith("\\")
    || /^[A-Za-z]:/.test(path)
    || path.split(/[\\/]/).some((segment) => segment === "..")
  ) {
    throw new OnnxModelManifestError(
      `${label} must remain inside the model package`,
    );
  }
  return path.replaceAll("\\", "/");
}

function requireExact(
  value: unknown,
  expected: string | number,
  label: string,
): void {
  if (value !== expected) {
    throw new OnnxModelManifestError(
      `${label} must be ${String(expected)}, got ${String(value)}`,
    );
  }
}

function assertUnique(values: readonly string[], label: string): void {
  if (new Set(values).size !== values.length) {
    throw new OnnxModelManifestError(`${label} must be unique`);
  }
}

function assertExactKeys(
  record: Record<string, unknown>,
  expected: readonly string[],
  label: string,
): void {
  assertAllowedKeys(record, expected, label);
  const missing = expected.filter((key) => !(key in record));
  if (missing.length > 0) {
    throw new OnnxModelManifestError(
      `${label} missing fields ${missing.join(", ")}`,
    );
  }
}

function assertAllowedKeys(
  record: Record<string, unknown>,
  allowed: readonly string[],
  label: string,
): void {
  const allowedSet = new Set(allowed);
  const unknown = Object.keys(record).filter((key) => !allowedSet.has(key));
  if (unknown.length > 0) {
    throw new OnnxModelManifestError(
      `${label} unknown fields ${unknown.sort().join(", ")}`,
    );
  }
}
