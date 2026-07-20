import {
  buildSpeculativeStateGroups,
  defaultSpeculativeEligibility,
  speculativeFamilyContract,
  type SpeculativeProposerFamily,
} from "./speculative-family.js";
import {
  decideSpeculativeIteration,
  planSpeculativeProposal,
  type SpeculativeIterationDecision,
} from "./speculative-iteration.js";
import {
  simulateSpeculativeWorkload,
  type SpeculativeWorkloadResult,
} from "./speculative-workload.js";

export const SPECULATIVE_TOKEN_TRACE_REVISION = 2;

export interface SpeculativeTokenTraceProvenance {
  readonly source: string;
  readonly runtimeRevision: string;
  readonly modelFingerprint: string;
  readonly proposerFingerprint: string;
  readonly tokenizerFingerprint: string;
  readonly generationConfigFingerprint: string;
  readonly targetOnlyRunId: string;
  readonly speculativeRunId: string;
}

export interface SpeculativeTokenTraceIteration {
  readonly id: string;
  readonly proposalTokenIds: readonly number[];
  readonly targetTokenIds: readonly number[];
}

export interface SpeculativeTokenTrace {
  readonly revision: typeof SPECULATIVE_TOKEN_TRACE_REVISION;
  readonly id: string;
  readonly provenance: SpeculativeTokenTraceProvenance;
  readonly family: SpeculativeProposerFamily;
  readonly promptTokenIds: readonly number[];
  readonly expectedOutputTokenIds: readonly number[];
  readonly maxAdditionalTokens: number;
  readonly iterations: readonly SpeculativeTokenTraceIteration[];
}

export interface SpeculativeTokenTraceIterationResult
  extends SpeculativeIterationDecision {
  readonly iteration: number;
  readonly id: string;
  readonly outputOffset: number;
  readonly proposalTokenIds: readonly number[];
  readonly targetTokenIds: readonly number[];
  readonly committedTokenIds: readonly number[];
}

export interface SpeculativeTokenMismatch {
  readonly outputIndex: number;
  readonly expectedTokenId: number;
  readonly actualTokenId: number;
}

export interface SpeculativeTokenDifferential {
  readonly matchesTargetOnly: boolean;
  readonly comparedTokenCount: number;
  readonly firstMismatch?: SpeculativeTokenMismatch;
}

export interface SpeculativeTokenTraceResult {
  readonly traceId: string;
  readonly revision: typeof SPECULATIVE_TOKEN_TRACE_REVISION;
  readonly family: SpeculativeProposerFamily;
  readonly provenance: SpeculativeTokenTraceProvenance;
  readonly promptTokenCount: number;
  readonly promptTokenIds: readonly number[];
  readonly expectedOutputTokenIds: readonly number[];
  readonly committedOutputTokenIds: readonly number[];
  readonly iterations: readonly SpeculativeTokenTraceIterationResult[];
  readonly differential: SpeculativeTokenDifferential;
  readonly workload: SpeculativeWorkloadResult;
}

export class SpeculativeTokenTraceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SpeculativeTokenTraceError";
  }
}

export function parseSpeculativeTokenTrace(
  input: unknown,
): SpeculativeTokenTrace {
  const root = requireRecord(input, "config");
  const value = requireRecord(
    root.speculative_token_trace ?? root,
    "speculative_token_trace",
  );
  if (root.speculative_token_trace !== undefined) {
    assertOnlyKeys(root, ["speculative_token_trace"], "config");
  }
  assertOnlyKeys(value, [
    "revision",
    "id",
    "provenance",
    "family",
    "prompt_token_ids",
    "expected_output_token_ids",
    "max_additional_tokens",
    "iterations",
  ], "speculative_token_trace");
  const provenanceValue = requireRecord(
    value.provenance,
    "speculative_token_trace.provenance",
  );
  assertOnlyKeys(provenanceValue, [
    "source",
    "runtime_revision",
    "model_fingerprint",
    "proposer_fingerprint",
    "tokenizer_fingerprint",
    "generation_config_fingerprint",
    "target_only_run_id",
    "speculative_run_id",
  ], "speculative_token_trace.provenance");
  const iterations = requireRecordArray(
    value,
    "iterations",
    "speculative_token_trace",
  ).map((iteration, index) => {
    const path = `speculative_token_trace.iterations[${index}]`;
    assertOnlyKeys(
      iteration,
      ["id", "proposal_token_ids", "target_token_ids"],
      path,
    );
    return {
      id: requireString(iteration, "id", path),
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
    };
  });
  const trace: SpeculativeTokenTrace = {
    revision: requireNumber(
      value,
      "revision",
      "speculative_token_trace",
    ) as typeof SPECULATIVE_TOKEN_TRACE_REVISION,
    id: requireString(value, "id", "speculative_token_trace"),
    provenance: {
      source: requireString(
        provenanceValue,
        "source",
        "speculative_token_trace.provenance",
      ),
      runtimeRevision: requireString(
        provenanceValue,
        "runtime_revision",
        "speculative_token_trace.provenance",
      ),
      modelFingerprint: requireString(
        provenanceValue,
        "model_fingerprint",
        "speculative_token_trace.provenance",
      ),
      proposerFingerprint: requireString(
        provenanceValue,
        "proposer_fingerprint",
        "speculative_token_trace.provenance",
      ),
      tokenizerFingerprint: requireString(
        provenanceValue,
        "tokenizer_fingerprint",
        "speculative_token_trace.provenance",
      ),
      generationConfigFingerprint: requireString(
        provenanceValue,
        "generation_config_fingerprint",
        "speculative_token_trace.provenance",
      ),
      targetOnlyRunId: requireString(
        provenanceValue,
        "target_only_run_id",
        "speculative_token_trace.provenance",
      ),
      speculativeRunId: requireString(
        provenanceValue,
        "speculative_run_id",
        "speculative_token_trace.provenance",
      ),
    },
    family: requireFamily(
      requireString(value, "family", "speculative_token_trace"),
    ),
    promptTokenIds: requireNumberArray(
      value,
      "prompt_token_ids",
      "speculative_token_trace",
    ),
    expectedOutputTokenIds: requireNumberArray(
      value,
      "expected_output_token_ids",
      "speculative_token_trace",
    ),
    maxAdditionalTokens: requireNumber(
      value,
      "max_additional_tokens",
      "speculative_token_trace",
    ),
    iterations,
  };
  validateSpeculativeTokenTrace(trace);
  return trace;
}

export function validateSpeculativeTokenTrace(
  trace: SpeculativeTokenTrace,
): void {
  if (trace.revision !== SPECULATIVE_TOKEN_TRACE_REVISION) {
    throw new SpeculativeTokenTraceError(
      `unsupported speculative token trace revision ${trace.revision}`,
    );
  }
  assertNonEmpty(trace.id, "trace id");
  const provenance = requireRecord(trace.provenance, "provenance");
  assertOnlyKeys(provenance, [
    "source",
    "runtimeRevision",
    "modelFingerprint",
    "proposerFingerprint",
    "tokenizerFingerprint",
    "generationConfigFingerprint",
    "targetOnlyRunId",
    "speculativeRunId",
  ], "provenance");
  for (const key of [
    "source",
    "runtimeRevision",
    "modelFingerprint",
    "proposerFingerprint",
    "tokenizerFingerprint",
    "generationConfigFingerprint",
    "targetOnlyRunId",
    "speculativeRunId",
  ]) {
    assertNonEmpty(
      requireString(provenance, key, "provenance"),
      `provenance.${key}`,
    );
  }
  if (
    trace.provenance.targetOnlyRunId
    === trace.provenance.speculativeRunId
  ) {
    throw new SpeculativeTokenTraceError(
      "target-only and speculative evidence must use distinct run ids",
    );
  }
  requireFamily(trace.family);
  assertNonNegativeSafeInteger(
    trace.maxAdditionalTokens,
    "maxAdditionalTokens",
  );
  validateTokenIds(trace.promptTokenIds, "promptTokenIds");
  validateTokenIds(
    trace.expectedOutputTokenIds,
    "expectedOutputTokenIds",
  );
  const iterationIds = new Set<string>();
  for (let index = 0; index < trace.iterations.length; index++) {
    const iteration = trace.iterations[index] as SpeculativeTokenTraceIteration;
    assertNonEmpty(iteration.id, `iterations[${index}].id`);
    if (iterationIds.has(iteration.id)) {
      throw new SpeculativeTokenTraceError(
        `duplicate speculative token trace iteration id ${iteration.id}`,
      );
    }
    iterationIds.add(iteration.id);
    validateTokenIds(
      iteration.proposalTokenIds,
      `iterations[${index}].proposalTokenIds`,
    );
    validateTokenIds(
      iteration.targetTokenIds,
      `iterations[${index}].targetTokenIds`,
    );
  }
}

export function simulateSpeculativeTokenTrace(
  trace: SpeculativeTokenTrace,
): SpeculativeTokenTraceResult {
  validateSpeculativeTokenTrace(trace);
  const contract = speculativeFamilyContract(trace.family);
  const committedOutputTokenIds: number[] = [];
  const iterations: SpeculativeTokenTraceIterationResult[] = [];
  const proposedAdditionalTokens: number[] = [];
  const acceptedAdditionalTokens: number[] = [];

  while (
    committedOutputTokenIds.length < trace.expectedOutputTokenIds.length
  ) {
    const iterationIndex = iterations.length;
    const iteration = trace.iterations[iterationIndex];
    if (!iteration) {
      throw new SpeculativeTokenTraceError(
        `trace ended before output token ${committedOutputTokenIds.length}`,
      );
    }
    const remainingOutputTokens =
      trace.expectedOutputTokenIds.length - committedOutputTokenIds.length;
    const guaranteedTargetTokens: 0 | 1 =
      contract.proposalPrefix === "guaranteed_target" ? 1 : 0;
    if (
      iteration.proposalTokenIds.length < guaranteedTargetTokens
    ) {
      throw new SpeculativeTokenTraceError(
        `${iteration.id} omits the guaranteed target proposal prefix`,
      );
    }
    const additionalWidth =
      iteration.proposalTokenIds.length - guaranteedTargetTokens;
    if (additionalWidth > trace.maxAdditionalTokens) {
      throw new SpeculativeTokenTraceError(
        `${iteration.id} proposes ${additionalWidth} additional tokens but the configured maximum is ${trace.maxAdditionalTokens}`,
      );
    }
    const proposal = planSpeculativeProposal({
      proposalPrefix: contract.proposalPrefix,
      remainingOutputTokens,
      maxAdditionalTokens: additionalWidth,
    });
    if (
      proposal.proposedDraftTokens !== iteration.proposalTokenIds.length
    ) {
      throw new SpeculativeTokenTraceError(
        `${iteration.id} proposal width exceeds the remaining output budget`,
      );
    }
    if (iteration.targetTokenIds.length > iteration.proposalTokenIds.length + 1) {
      throw new SpeculativeTokenTraceError(
        `${iteration.id} contains target-selected tokens beyond the proposal and optional bonus row`,
      );
    }

    const acceptedDraftTokens = matchingPrefixLength(
      iteration.proposalTokenIds,
      iteration.targetTokenIds,
    );
    if (
      guaranteedTargetTokens === 1
      && acceptedDraftTokens === 0
    ) {
      throw new SpeculativeTokenTraceError(
        `${iteration.id} rejects its guaranteed target proposal prefix`,
      );
    }
    const observedMismatch =
      acceptedDraftTokens < iteration.proposalTokenIds.length
      && iteration.targetTokenIds.length > acceptedDraftTokens;
    if (
      observedMismatch
      && iteration.targetTokenIds.length !== acceptedDraftTokens + 1
    ) {
      throw new SpeculativeTokenTraceError(
        `${iteration.id} contains target-selected tokens after the first mismatch`,
      );
    }
    if (
      !observedMismatch
      && iteration.targetTokenIds.length < iteration.proposalTokenIds.length
    ) {
      throw new SpeculativeTokenTraceError(
        `${iteration.id} omits a target-selected token before proposal verification completed`,
      );
    }
    const acceptedAdditional = acceptedDraftTokens - guaranteedTargetTokens;
    const decision = decideSpeculativeIteration(
      proposal,
      acceptedAdditional,
      remainingOutputTokens,
    );
    const expectedTargetTokenCount = decision.outcome === "bonus"
      ? iteration.proposalTokenIds.length + 1
      : decision.outcome === "accepted_tail"
        ? iteration.proposalTokenIds.length
        : acceptedDraftTokens + 1;
    if (iteration.targetTokenIds.length !== expectedTargetTokenCount) {
      throw new SpeculativeTokenTraceError(
        `${iteration.id} records ${iteration.targetTokenIds.length} target-selected tokens but ${decision.outcome} requires ${expectedTargetTokenCount}`,
      );
    }
    const committedTokenIds = [
      ...iteration.proposalTokenIds.slice(0, acceptedDraftTokens),
      ...(decision.targetAuthoritativeTokens === 1
        ? [iteration.targetTokenIds[acceptedDraftTokens] as number]
        : []),
    ];
    if (committedTokenIds.length !== decision.committedTokens) {
      throw new SpeculativeTokenTraceError(
        `${iteration.id} token values diverge from its commit decision`,
      );
    }
    proposedAdditionalTokens.push(additionalWidth);
    acceptedAdditionalTokens.push(acceptedAdditional);
    iterations.push({
      ...decision,
      iteration: iterationIndex,
      id: iteration.id,
      outputOffset: committedOutputTokenIds.length,
      proposalTokenIds: [...iteration.proposalTokenIds],
      targetTokenIds: [...iteration.targetTokenIds],
      committedTokenIds,
    });
    committedOutputTokenIds.push(...committedTokenIds);
  }

  if (iterations.length !== trace.iterations.length) {
    throw new SpeculativeTokenTraceError(
      `trace contains ${trace.iterations.length - iterations.length} iteration(s) after the output budget was complete`,
    );
  }

  const workload = simulateSpeculativeWorkload({
    family: trace.family,
    eligibility: defaultSpeculativeEligibility(trace.family),
    initialTokenLength: trace.promptTokenIds.length,
    outputTokenCount: trace.expectedOutputTokenIds.length,
    maxAdditionalTokens: trace.maxAdditionalTokens,
    proposal: {
      kind: "replay",
      proposedAdditionalTokens,
    },
    acceptance: {
      kind: "replay",
      acceptedDraftTokens: acceptedAdditionalTokens,
    },
    maxIterations: Math.max(1, trace.iterations.length),
    stateGroups: buildSpeculativeStateGroups(
      trace.family,
      trace.promptTokenIds.length + trace.expectedOutputTokenIds.length,
      trace.maxAdditionalTokens,
    ),
  });
  assertDecisionParity(iterations, workload);
  const firstMismatch = findFirstMismatch(
    trace.expectedOutputTokenIds,
    committedOutputTokenIds,
  );
  return {
    traceId: trace.id,
    revision: SPECULATIVE_TOKEN_TRACE_REVISION,
    family: trace.family,
    provenance: trace.provenance,
    promptTokenCount: trace.promptTokenIds.length,
    promptTokenIds: [...trace.promptTokenIds],
    expectedOutputTokenIds: [...trace.expectedOutputTokenIds],
    committedOutputTokenIds,
    iterations,
    differential: {
      matchesTargetOnly: firstMismatch === undefined,
      comparedTokenCount: trace.expectedOutputTokenIds.length,
      ...(firstMismatch ? { firstMismatch } : {}),
    },
    workload,
  };
}

function assertDecisionParity(
  traceIterations: readonly SpeculativeTokenTraceIterationResult[],
  workload: SpeculativeWorkloadResult,
): void {
  if (traceIterations.length !== workload.iterations.length) {
    throw new SpeculativeTokenTraceError(
      "token trace and state transaction produced different iteration counts",
    );
  }
  for (let index = 0; index < traceIterations.length; index++) {
    const trace = traceIterations[index] as SpeculativeTokenTraceIterationResult;
    const state = workload.iterations[index];
    if (
      !state
      || trace.proposedDraftTokens !== state.proposedDraftTokens
      || trace.acceptedDraftTokens !== state.acceptedDraftTokens
      || trace.targetAuthoritativeTokens !== state.targetAuthoritativeTokens
      || trace.committedTokens !== state.committedTokens
      || trace.outcome !== state.outcome
    ) {
      throw new SpeculativeTokenTraceError(
        `token/state decision parity failed at iteration ${index}`,
      );
    }
  }
}

function matchingPrefixLength(
  proposal: readonly number[],
  target: readonly number[],
): number {
  let accepted = 0;
  while (
    accepted < proposal.length
    && proposal[accepted] === target[accepted]
  ) {
    accepted++;
  }
  return accepted;
}

function findFirstMismatch(
  expected: readonly number[],
  actual: readonly number[],
): SpeculativeTokenMismatch | undefined {
  for (let index = 0; index < expected.length; index++) {
    if (expected[index] !== actual[index]) {
      return {
        outputIndex: index,
        expectedTokenId: expected[index] as number,
        actualTokenId: actual[index] as number,
      };
    }
  }
  return undefined;
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
    throw new SpeculativeTokenTraceError(
      `unsupported speculative family ${value}`,
    );
  }
  return value;
}

function validateTokenIds(values: readonly number[], path: string): void {
  for (let index = 0; index < values.length; index++) {
    assertNonNegativeSafeInteger(values[index] as number, `${path}[${index}]`);
  }
}

function assertNonEmpty(value: string, path: string): void {
  if (value.trim().length === 0) {
    throw new SpeculativeTokenTraceError(`${path} must be non-empty`);
  }
}

function assertNonNegativeSafeInteger(value: number, path: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new SpeculativeTokenTraceError(
      `${path} must be a non-negative safe integer; got ${value}`,
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
      throw new SpeculativeTokenTraceError(`${path} has unknown field ${key}`);
    }
  }
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new SpeculativeTokenTraceError(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireString(
  record: Record<string, unknown>,
  key: string,
  path: string,
): string {
  const value = record[key];
  if (typeof value !== "string") {
    throw new SpeculativeTokenTraceError(`${path}.${key} must be a string`);
  }
  return value;
}

function requireNumber(
  record: Record<string, unknown>,
  key: string,
  path: string,
): number {
  const value = record[key];
  if (typeof value !== "number") {
    throw new SpeculativeTokenTraceError(`${path}.${key} must be a number`);
  }
  return value;
}

function requireNumberArray(
  record: Record<string, unknown>,
  key: string,
  path: string,
): readonly number[] {
  const value = record[key];
  if (
    !Array.isArray(value)
    || value.some((entry) => typeof entry !== "number")
  ) {
    throw new SpeculativeTokenTraceError(
      `${path}.${key} must be an array of numbers`,
    );
  }
  return value as number[];
}

function requireRecordArray(
  record: Record<string, unknown>,
  key: string,
  path: string,
): readonly Record<string, unknown>[] {
  const value = record[key];
  if (!Array.isArray(value)) {
    throw new SpeculativeTokenTraceError(
      `${path}.${key} must be an array`,
    );
  }
  return value.map((entry, index) => (
    requireRecord(entry, `${path}.${key}[${index}]`)
  ));
}
