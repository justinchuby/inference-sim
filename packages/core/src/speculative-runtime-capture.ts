import {
  SPECULATIVE_TOKEN_TRACE_REVISION,
  simulateSpeculativeTokenTrace,
  type SpeculativeTokenTrace,
  type SpeculativeTokenTraceResult,
} from "./speculative-token-trace.js";
import type { SpeculativeProposerFamily } from "./speculative-workload.js";

export const SPECULATIVE_RUNTIME_CAPTURE_REVISION = 1;

export interface SpeculativeRuntimeCaptureProvenance {
  readonly source: string;
  readonly runtimeRevision: string;
  readonly modelFingerprint: string;
  readonly tokenizerFingerprint: string;
  readonly generationConfigFingerprint: string;
}

export interface SpeculativeRuntimeCaptureTerminal {
  readonly status: "completed";
  readonly outputTokenCount: number;
  readonly iterationCount: number;
}

interface RuntimeCaptureBase {
  readonly revision: typeof SPECULATIVE_RUNTIME_CAPTURE_REVISION;
  readonly id: string;
  readonly provenance: SpeculativeRuntimeCaptureProvenance;
  readonly completionReason: "max_tokens";
  readonly promptTokenIds: readonly number[];
  readonly outputTokenIds: readonly number[];
  readonly terminal: SpeculativeRuntimeCaptureTerminal;
}

export interface TargetOnlyRuntimeCapture extends RuntimeCaptureBase {
  readonly role: "target_only";
}

export interface SpeculativeRuntimeCaptureIteration {
  readonly id: string;
  readonly outputOffset: number;
  readonly proposalTokenIds: readonly number[];
  readonly targetTokenIds: readonly number[];
  readonly committedTokenIds: readonly number[];
}

export interface SpeculativeRuntimeCapture extends RuntimeCaptureBase {
  readonly role: "speculative";
  readonly family: SpeculativeProposerFamily;
  readonly proposerFingerprint: string;
  readonly maxAdditionalTokens: number;
  readonly iterations: readonly SpeculativeRuntimeCaptureIteration[];
}

export type RuntimeTokenCapture =
  | TargetOnlyRuntimeCapture
  | SpeculativeRuntimeCapture;

export interface BoundSpeculativeRuntimeCapture {
  readonly targetOnlyCaptureId: string;
  readonly speculativeCaptureId: string;
  readonly trace: SpeculativeTokenTrace;
  readonly result: SpeculativeTokenTraceResult;
}

export class SpeculativeRuntimeCaptureError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SpeculativeRuntimeCaptureError";
  }
}

export function parseRuntimeTokenCapture(input: unknown): RuntimeTokenCapture {
  const root = requireRecord(input, "config");
  const value = requireRecord(
    root.runtime_token_capture ?? root,
    "runtime_token_capture",
  );
  if (root.runtime_token_capture !== undefined) {
    assertOnlyKeys(root, ["runtime_token_capture"], "config");
  }
  const role = requireString(value, "role", "runtime_token_capture");
  const commonKeys = [
    "revision",
    "id",
    "role",
    "provenance",
    "completion_reason",
    "prompt_token_ids",
    "output_token_ids",
    "terminal",
  ];
  if (role === "target_only") {
    assertOnlyKeys(value, commonKeys, "runtime_token_capture");
  } else if (role === "speculative") {
    assertOnlyKeys(value, [
      ...commonKeys,
      "family",
      "proposer_fingerprint",
      "max_additional_tokens",
      "iterations",
    ], "runtime_token_capture");
  } else {
    throw new SpeculativeRuntimeCaptureError(
      `unsupported runtime token capture role ${role}`,
    );
  }

  const provenanceValue = requireRecord(
    value.provenance,
    "runtime_token_capture.provenance",
  );
  assertOnlyKeys(provenanceValue, [
    "source",
    "runtime_revision",
    "model_fingerprint",
    "tokenizer_fingerprint",
    "generation_config_fingerprint",
  ], "runtime_token_capture.provenance");
  const terminalValue = requireRecord(
    value.terminal,
    "runtime_token_capture.terminal",
  );
  assertOnlyKeys(terminalValue, [
    "status",
    "output_token_count",
    "iteration_count",
  ], "runtime_token_capture.terminal");

  const base = {
    revision: requireNumber(
      value,
      "revision",
      "runtime_token_capture",
    ) as typeof SPECULATIVE_RUNTIME_CAPTURE_REVISION,
    id: requireString(value, "id", "runtime_token_capture"),
    provenance: {
      source: requireString(
        provenanceValue,
        "source",
        "runtime_token_capture.provenance",
      ),
      runtimeRevision: requireString(
        provenanceValue,
        "runtime_revision",
        "runtime_token_capture.provenance",
      ),
      modelFingerprint: requireString(
        provenanceValue,
        "model_fingerprint",
        "runtime_token_capture.provenance",
      ),
      tokenizerFingerprint: requireString(
        provenanceValue,
        "tokenizer_fingerprint",
        "runtime_token_capture.provenance",
      ),
      generationConfigFingerprint: requireString(
        provenanceValue,
        "generation_config_fingerprint",
        "runtime_token_capture.provenance",
      ),
    },
    completionReason: requireLiteral(
      value,
      "completion_reason",
      "max_tokens",
      "runtime_token_capture",
    ),
    promptTokenIds: requireNumberArray(
      value,
      "prompt_token_ids",
      "runtime_token_capture",
    ),
    outputTokenIds: requireNumberArray(
      value,
      "output_token_ids",
      "runtime_token_capture",
    ),
    terminal: {
      status: requireLiteral(
        terminalValue,
        "status",
        "completed",
        "runtime_token_capture.terminal",
      ),
      outputTokenCount: requireNumber(
        terminalValue,
        "output_token_count",
        "runtime_token_capture.terminal",
      ),
      iterationCount: requireNumber(
        terminalValue,
        "iteration_count",
        "runtime_token_capture.terminal",
      ),
    },
  } as const;

  if (role === "target_only") {
    const capture: TargetOnlyRuntimeCapture = {
      ...base,
      role,
    };
    validateRuntimeTokenCapture(capture);
    return capture;
  }

  const iterations = requireRecordArray(
    value,
    "iterations",
    "runtime_token_capture",
  ).map((iteration, index) => {
    const path = `runtime_token_capture.iterations[${index}]`;
    assertOnlyKeys(iteration, [
      "id",
      "output_offset",
      "proposal_token_ids",
      "target_token_ids",
      "committed_token_ids",
    ], path);
    return {
      id: requireString(iteration, "id", path),
      outputOffset: requireNumber(iteration, "output_offset", path),
      proposalTokenIds: requireNumberArray(
        iteration,
        "proposal_token_ids",
        path,
      ),
      targetTokenIds: requireNumberArray(
        iteration,
        "target_token_ids",
        path,
      ),
      committedTokenIds: requireNumberArray(
        iteration,
        "committed_token_ids",
        path,
      ),
    };
  });
  const capture: SpeculativeRuntimeCapture = {
    ...base,
    role,
    family: requireFamily(
      requireString(value, "family", "runtime_token_capture"),
    ),
    proposerFingerprint: requireString(
      value,
      "proposer_fingerprint",
      "runtime_token_capture",
    ),
    maxAdditionalTokens: requireNumber(
      value,
      "max_additional_tokens",
      "runtime_token_capture",
    ),
    iterations,
  };
  validateRuntimeTokenCapture(capture);
  return capture;
}

export function validateRuntimeTokenCapture(
  capture: RuntimeTokenCapture,
): void {
  const captureValue = requireRecord(capture, "runtimeTokenCapture");
  const role = requireString(captureValue, "role", "runtimeTokenCapture");
  const commonKeys = [
    "revision",
    "id",
    "role",
    "provenance",
    "completionReason",
    "promptTokenIds",
    "outputTokenIds",
    "terminal",
  ];
  if (role === "target_only") {
    assertOnlyKeys(captureValue, commonKeys, "runtimeTokenCapture");
  } else if (role === "speculative") {
    assertOnlyKeys(captureValue, [
      ...commonKeys,
      "family",
      "proposerFingerprint",
      "maxAdditionalTokens",
      "iterations",
    ], "runtimeTokenCapture");
  } else {
    throw new SpeculativeRuntimeCaptureError(
      `unsupported runtime token capture role ${role}`,
    );
  }
  if (capture.revision !== SPECULATIVE_RUNTIME_CAPTURE_REVISION) {
    throw new SpeculativeRuntimeCaptureError(
      `unsupported runtime token capture revision ${capture.revision}`,
    );
  }
  if (capture.completionReason !== "max_tokens") {
    throw new SpeculativeRuntimeCaptureError(
      "runtime token capture completionReason must be max_tokens",
    );
  }
  const terminal = requireRecord(capture.terminal, "terminal");
  assertOnlyKeys(terminal, [
    "status",
    "outputTokenCount",
    "iterationCount",
  ], "terminal");
  if (requireString(terminal, "status", "terminal") !== "completed") {
    throw new SpeculativeRuntimeCaptureError(
      "runtime token capture terminal status must be completed",
    );
  }
  assertNonEmpty(capture.id, "capture id");
  validateTokenIds(capture.promptTokenIds, "promptTokenIds");
  validateTokenIds(capture.outputTokenIds, "outputTokenIds");
  if (capture.promptTokenIds.length === 0) {
    throw new SpeculativeRuntimeCaptureError(
      "runtime token capture prompt must be non-empty",
    );
  }
  if (capture.outputTokenIds.length === 0) {
    throw new SpeculativeRuntimeCaptureError(
      "runtime token capture output must be non-empty",
    );
  }
  const provenance = requireRecord(capture.provenance, "provenance");
  assertOnlyKeys(provenance, [
    "source",
    "runtimeRevision",
    "modelFingerprint",
    "tokenizerFingerprint",
    "generationConfigFingerprint",
  ], "provenance");
  for (const key of [
    "source",
    "runtimeRevision",
    "modelFingerprint",
    "tokenizerFingerprint",
    "generationConfigFingerprint",
  ]) {
    assertNonEmpty(
      requireString(provenance, key, "provenance"),
      `provenance.${key}`,
    );
  }
  assertNonNegativeSafeInteger(
    capture.terminal.outputTokenCount,
    "terminal.outputTokenCount",
  );
  assertNonNegativeSafeInteger(
    capture.terminal.iterationCount,
    "terminal.iterationCount",
  );
  if (capture.terminal.outputTokenCount !== capture.outputTokenIds.length) {
    throw new SpeculativeRuntimeCaptureError(
      "terminal output token count does not match the captured output",
    );
  }

  if (capture.role === "target_only") {
    if (capture.terminal.iterationCount !== 0) {
      throw new SpeculativeRuntimeCaptureError(
        "target-only capture terminal iteration count must be zero",
      );
    }
    return;
  }

  requireFamily(capture.family);
  assertNonEmpty(capture.proposerFingerprint, "proposerFingerprint");
  assertNonNegativeSafeInteger(
    capture.maxAdditionalTokens,
    "maxAdditionalTokens",
  );
  if (!Array.isArray(capture.iterations) || capture.iterations.length === 0) {
    throw new SpeculativeRuntimeCaptureError(
      "speculative capture must contain at least one iteration",
    );
  }
  if (capture.terminal.iterationCount !== capture.iterations.length) {
    throw new SpeculativeRuntimeCaptureError(
      "terminal iteration count does not match captured iterations",
    );
  }
  const ids = new Set<string>();
  for (let index = 0; index < capture.iterations.length; index++) {
    const iteration = capture.iterations[index] as SpeculativeRuntimeCaptureIteration;
    const iterationValue = requireRecord(
      iteration,
      `iterations[${index}]`,
    );
    assertOnlyKeys(iterationValue, [
      "id",
      "outputOffset",
      "proposalTokenIds",
      "targetTokenIds",
      "committedTokenIds",
    ], `iterations[${index}]`);
    assertNonEmpty(iteration.id, `iterations[${index}].id`);
    if (ids.has(iteration.id)) {
      throw new SpeculativeRuntimeCaptureError(
        `duplicate runtime capture iteration id ${iteration.id}`,
      );
    }
    ids.add(iteration.id);
    assertNonNegativeSafeInteger(
      iteration.outputOffset,
      `iterations[${index}].outputOffset`,
    );
    validateTokenIds(
      iteration.proposalTokenIds,
      `iterations[${index}].proposalTokenIds`,
    );
    validateTokenIds(
      iteration.targetTokenIds,
      `iterations[${index}].targetTokenIds`,
    );
    validateTokenIds(
      iteration.committedTokenIds,
      `iterations[${index}].committedTokenIds`,
    );
  }
}

export function bindSpeculativeRuntimeCaptures(
  targetOnly: TargetOnlyRuntimeCapture,
  speculative: SpeculativeRuntimeCapture,
): BoundSpeculativeRuntimeCapture {
  validateRuntimeTokenCapture(targetOnly);
  validateRuntimeTokenCapture(speculative);
  if (targetOnly.id === speculative.id) {
    throw new SpeculativeRuntimeCaptureError(
      "target-only and speculative captures must use distinct run ids",
    );
  }
  for (const key of [
    "source",
    "runtimeRevision",
    "modelFingerprint",
    "tokenizerFingerprint",
    "generationConfigFingerprint",
  ] as const) {
    if (targetOnly.provenance[key] !== speculative.provenance[key]) {
      throw new SpeculativeRuntimeCaptureError(
        `capture provenance mismatch for ${key}`,
      );
    }
  }
  assertEqualTokenIds(
    targetOnly.promptTokenIds,
    speculative.promptTokenIds,
    "target-only and speculative prompts differ",
  );
  if (targetOnly.outputTokenIds.length !== speculative.outputTokenIds.length) {
    throw new SpeculativeRuntimeCaptureError(
      "max-token captures produced different output lengths",
    );
  }

  const trace: SpeculativeTokenTrace = {
    revision: SPECULATIVE_TOKEN_TRACE_REVISION,
    id: `${targetOnly.id}:${speculative.id}`,
    provenance: {
      source: speculative.provenance.source,
      runtimeRevision: speculative.provenance.runtimeRevision,
      modelFingerprint: speculative.provenance.modelFingerprint,
      proposerFingerprint: speculative.proposerFingerprint,
      tokenizerFingerprint: speculative.provenance.tokenizerFingerprint,
      generationConfigFingerprint:
        speculative.provenance.generationConfigFingerprint,
      targetOnlyRunId: targetOnly.id,
      speculativeRunId: speculative.id,
    },
    family: speculative.family,
    promptTokenIds: [...targetOnly.promptTokenIds],
    expectedOutputTokenIds: [...targetOnly.outputTokenIds],
    maxAdditionalTokens: speculative.maxAdditionalTokens,
    iterations: speculative.iterations.map((iteration) => ({
      id: iteration.id,
      proposalTokenIds: [...iteration.proposalTokenIds],
      targetTokenIds: [...iteration.targetTokenIds],
    })),
  };
  const result = simulateSpeculativeTokenTrace(trace);
  assertEqualTokenIds(
    result.committedOutputTokenIds,
    speculative.outputTokenIds,
    "runtime speculative output differs from independently reconstructed commits",
  );
  for (let index = 0; index < speculative.iterations.length; index++) {
    const captured = speculative.iterations[index] as SpeculativeRuntimeCaptureIteration;
    const derived = result.iterations[index];
    if (!derived) {
      throw new SpeculativeRuntimeCaptureError(
        `missing derived iteration ${index}`,
      );
    }
    if (captured.outputOffset !== derived.outputOffset) {
      throw new SpeculativeRuntimeCaptureError(
        `${captured.id} output offset ${captured.outputOffset} does not match derived offset ${derived.outputOffset}`,
      );
    }
    assertEqualTokenIds(
      captured.committedTokenIds,
      derived.committedTokenIds,
      `${captured.id} runtime commits differ from the acceptance oracle`,
    );
  }
  return {
    targetOnlyCaptureId: targetOnly.id,
    speculativeCaptureId: speculative.id,
    trace,
    result,
  };
}

export function bindParsedRuntimeCaptures(
  targetOnlyInput: unknown,
  speculativeInput: unknown,
): BoundSpeculativeRuntimeCapture {
  const targetOnly = parseRuntimeTokenCapture(targetOnlyInput);
  const speculative = parseRuntimeTokenCapture(speculativeInput);
  if (targetOnly.role !== "target_only") {
    throw new SpeculativeRuntimeCaptureError(
      "first runtime capture must have role target_only",
    );
  }
  if (speculative.role !== "speculative") {
    throw new SpeculativeRuntimeCaptureError(
      "second runtime capture must have role speculative",
    );
  }
  return bindSpeculativeRuntimeCaptures(targetOnly, speculative);
}

function requireFamily(value: string): SpeculativeProposerFamily {
  if (
    value !== "prompt_lookup"
    && value !== "draft_model"
    && value !== "mtp"
    && value !== "eagle3"
    && value !== "shared_kv"
    && value !== "self_speculative"
  ) {
    throw new SpeculativeRuntimeCaptureError(
      `unsupported speculative family ${value}`,
    );
  }
  return value;
}

function validateTokenIds(tokens: readonly number[], path: string): void {
  if (!Array.isArray(tokens)) {
    throw new SpeculativeRuntimeCaptureError(`${path} must be an array`);
  }
  for (let index = 0; index < tokens.length; index++) {
    assertNonNegativeSafeInteger(tokens[index], `${path}[${index}]`);
  }
}

function assertEqualTokenIds(
  expected: readonly number[],
  actual: readonly number[],
  message: string,
): void {
  if (
    expected.length !== actual.length
    || expected.some((token, index) => token !== actual[index])
  ) {
    throw new SpeculativeRuntimeCaptureError(message);
  }
}

function assertNonEmpty(value: string, path: string): void {
  if (value.trim().length === 0) {
    throw new SpeculativeRuntimeCaptureError(`${path} must be non-empty`);
  }
}

function assertNonNegativeSafeInteger(
  value: number | undefined,
  path: string,
): void {
  if (
    value === undefined
    || !Number.isSafeInteger(value)
    || value < 0
  ) {
    throw new SpeculativeRuntimeCaptureError(
      `${path} must be a non-negative safe integer`,
    );
  }
}

function assertOnlyKeys(
  value: Record<string, unknown>,
  allowed: readonly string[],
  path: string,
): void {
  const allowedKeys = new Set(allowed);
  for (const key of Object.keys(value)) {
    if (!allowedKeys.has(key)) {
      throw new SpeculativeRuntimeCaptureError(
        `${path} has unknown field ${key}`,
      );
    }
  }
}

function requireRecord(
  value: unknown,
  path: string,
): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new SpeculativeRuntimeCaptureError(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireString(
  record: Record<string, unknown>,
  key: string,
  path: string,
): string {
  const value = record[key];
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new SpeculativeRuntimeCaptureError(
      `${path}.${key} must be a non-empty string`,
    );
  }
  return value;
}

function requireLiteral<T extends string>(
  record: Record<string, unknown>,
  key: string,
  expected: T,
  path: string,
): T {
  const value = requireString(record, key, path);
  if (value !== expected) {
    throw new SpeculativeRuntimeCaptureError(
      `${path}.${key} must be ${expected}`,
    );
  }
  return expected;
}

function requireNumber(
  record: Record<string, unknown>,
  key: string,
  path: string,
): number {
  const value = record[key];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new SpeculativeRuntimeCaptureError(
      `${path}.${key} must be a finite number`,
    );
  }
  return value;
}

function requireNumberArray(
  record: Record<string, unknown>,
  key: string,
  path: string,
): number[] {
  const value = record[key];
  if (
    !Array.isArray(value)
    || value.some((entry) => typeof entry !== "number" || !Number.isFinite(entry))
  ) {
    throw new SpeculativeRuntimeCaptureError(
      `${path}.${key} must be an array of finite numbers`,
    );
  }
  return value;
}

function requireRecordArray(
  record: Record<string, unknown>,
  key: string,
  path: string,
): Record<string, unknown>[] {
  const value = record[key];
  if (!Array.isArray(value)) {
    throw new SpeculativeRuntimeCaptureError(
      `${path}.${key} must be an array`,
    );
  }
  return value.map((entry, index) => (
    requireRecord(entry, `${path}.${key}[${index}]`)
  ));
}
