import {
  SpeculativeTransactionSimulator,
  type SpeculativeStateGroupConfig,
  type SpeculativeStateGroupSnapshot,
} from "./speculative.js";
import {
  PagedKvCacheSimulator,
  replayPagedKvTrace,
  type PagedKvConfig,
  type PagedKvSnapshot,
  type PagedKvTraceEvent,
} from "./paged-kv.js";
import {
  validateSpeculativeEligibility,
  validateSpeculativeStateGroups,
  type SpeculativeEligibility,
  type SpeculativeFamilyContract,
  type SpeculativeProposerFamily,
} from "./speculative-family.js";
export type { SpeculativeProposerFamily } from "./speculative-family.js";

export type SpeculativeAcceptanceModel =
  | {
      readonly kind: "replay";
      readonly acceptedDraftTokens: readonly number[];
    }
  | {
      readonly kind: "conditional_empirical";
      readonly matchProbabilityByPosition: readonly number[];
      readonly seed: number;
    }
  | {
      readonly kind: "conditional_heuristic";
      readonly matchProbabilityByPosition: readonly number[];
      readonly seed: number;
    };

export interface SpeculativeWorkloadConfig {
  readonly family: SpeculativeProposerFamily;
  readonly eligibility: SpeculativeEligibility;
  readonly initialTokenLength: number;
  readonly outputTokenCount: number;
  readonly maxAdditionalTokens: number;
  readonly stateGroups: readonly SpeculativeStateGroupConfig[];
  readonly acceptance: SpeculativeAcceptanceModel;
  readonly maxIterations?: number;
  readonly pagedKv?: Omit<
    PagedKvConfig,
    "sequenceId" | "initialTokenLength"
  > & {
    readonly sequenceId?: string;
  };
}

export interface SpeculativeWorkloadIteration {
  readonly iteration: number;
  readonly baseTokenLength: number;
  readonly proposedDraftTokens: number;
  readonly acceptedDraftTokens: number;
  readonly rejectedDraftTokens: number;
  readonly committedTokens: number;
  readonly finalTokenLength: number;
  readonly outcome: "correction" | "bonus" | "target_only";
}

export interface SpeculativeWorkloadMetrics {
  readonly iterations: number;
  readonly targetForwards: number;
  readonly proposedDraftTokens: number;
  readonly acceptedDraftTokens: number;
  readonly rejectedDraftTokens: number;
  readonly committedTokens: number;
  readonly correctionTokens: number;
  readonly bonusTokens: number;
  readonly targetOnlyTokens: number;
  readonly committedTokensPerTargetForward: number;
  readonly acceptedPrefixHistogram: readonly number[];
  readonly acceptanceByPosition: readonly number[];
  readonly kvPagesAllocated: number;
  readonly kvPagesReleased: number;
  readonly kvHighWaterReservedBytes: number;
  readonly kvFinalReservedBytes: number;
}

export interface SpeculativePagedKvResult {
  readonly snapshot: PagedKvSnapshot;
  readonly trace: readonly PagedKvTraceEvent[];
}

export interface SpeculativeWorkloadResult {
  readonly family: SpeculativeProposerFamily;
  readonly familyContract: SpeculativeFamilyContract;
  readonly initialTokenLength: number;
  readonly finalTokenLength: number;
  readonly targetOnlyFinalTokenLength: number;
  readonly iterations: readonly SpeculativeWorkloadIteration[];
  readonly metrics: SpeculativeWorkloadMetrics;
  readonly stateGroups: readonly SpeculativeStateGroupSnapshot[];
  readonly pagedKv?: SpeculativePagedKvResult;
}

export class SpeculativeWorkloadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SpeculativeWorkloadError";
  }
}

export function simulateSpeculativeWorkload(
  config: SpeculativeWorkloadConfig,
): SpeculativeWorkloadResult {
  validateConfig(config);
  const familyContract = validateSpeculativeEligibility(
    config.family,
    config.eligibility,
  );
  validateSpeculativeStateGroups(config.family, config.stateGroups);
  const transaction = new SpeculativeTransactionSimulator(
    config.initialTokenLength,
    config.stateGroups,
  );
  const pagedKvConfig: PagedKvConfig | undefined = config.pagedKv
    ? {
        ...config.pagedKv,
        sequenceId: config.pagedKv.sequenceId ?? "target",
        initialTokenLength: config.initialTokenLength,
      }
    : undefined;
  const pagedKv = pagedKvConfig
    ? new PagedKvCacheSimulator(pagedKvConfig)
    : undefined;
  const rng = config.acceptance.kind === "replay"
    ? undefined
    : new DeterministicRng(config.acceptance.seed);
  const maxIterations = config.maxIterations
    ?? Math.max(1, config.outputTokenCount);
  const acceptedPrefixHistogram = Array.from(
    { length: config.maxAdditionalTokens + 1 },
    () => 0,
  );
  const proposedByPosition = Array.from(
    { length: config.maxAdditionalTokens },
    () => 0,
  );
  const acceptedByPosition = Array.from(
    { length: config.maxAdditionalTokens },
    () => 0,
  );
  const iterations: SpeculativeWorkloadIteration[] = [];
  let remaining = config.outputTokenCount;
  let proposedDraftTokens = 0;
  let acceptedDraftTokens = 0;
  let correctionTokens = 0;
  let bonusTokens = 0;
  let targetOnlyTokens = 0;

  while (remaining > 0) {
    if (iterations.length >= maxIterations) {
      throw new SpeculativeWorkloadError(
        `workload exceeded maximum iteration count ${maxIterations}`,
      );
    }
    const draftTokenCount = Math.min(
      config.maxAdditionalTokens,
      Math.max(0, remaining - 1),
    );
    const accepted = chooseAcceptedPrefix(
      config.acceptance,
      rng,
      iterations.length,
      draftTokenCount,
    );
    const kvCheckpoint = pagedKv?.checkpoint();
    pagedKv?.append(draftTokenCount);
    const transactionResult = transaction.runIteration({
      draftTokenCount,
      acceptedDraftTokenCount: accepted,
    });
    if (pagedKv && kvCheckpoint) {
      pagedKv.restore(kvCheckpoint, accepted);
      pagedKv.append(1);
    }
    if (transactionResult.committedTokenCount > remaining) {
      throw new SpeculativeWorkloadError(
        `iteration ${iterations.length} over-committed output budget`,
      );
    }

    const outcome = draftTokenCount === 0
      ? "target_only"
      : accepted === draftTokenCount
        ? "bonus"
        : "correction";
    if (outcome === "target_only") {
      targetOnlyTokens++;
    } else if (outcome === "bonus") {
      bonusTokens++;
    } else {
      correctionTokens++;
    }
    acceptedPrefixHistogram[accepted]++;
    for (let position = 0; position < draftTokenCount; position++) {
      proposedByPosition[position]++;
      if (position < accepted) {
        acceptedByPosition[position]++;
      }
    }
    proposedDraftTokens += draftTokenCount;
    acceptedDraftTokens += accepted;
    remaining -= transactionResult.committedTokenCount;
    iterations.push({
      iteration: iterations.length,
      baseTokenLength: transactionResult.baseTokenLength,
      proposedDraftTokens: draftTokenCount,
      acceptedDraftTokens: accepted,
      rejectedDraftTokens: draftTokenCount - accepted,
      committedTokens: transactionResult.committedTokenCount,
      finalTokenLength: transactionResult.finalTokenLength,
      outcome,
    });
  }

  const finalTokenLength = transaction.tokenLength;
  const targetOnlyFinalTokenLength = checkedAdd(
    config.initialTokenLength,
    config.outputTokenCount,
    "target-only final length",
  );
  if (finalTokenLength !== targetOnlyFinalTokenLength) {
    throw new SpeculativeWorkloadError(
      `speculative final length ${finalTokenLength} diverges from target-only ${targetOnlyFinalTokenLength}`,
    );
  }
  const pagedKvTrace = pagedKv?.trace();
  const pagedKvSnapshot = pagedKv?.snapshot();
  if (
    pagedKvSnapshot
    && (
      pagedKvSnapshot.logicalTokenLength !== finalTokenLength
      || !pagedKvConfig
      || replayPagedKvTrace(pagedKvConfig, pagedKvTrace ?? []).snapshot
        .logicalTokenLength !== finalTokenLength
    )
  ) {
    throw new SpeculativeWorkloadError(
      "paged KV logical length diverges from committed output",
    );
  }
  const kvPagesAllocated = pagedKvTrace?.reduce((sum, event) => (
    event.kind === "initialize" || event.kind === "append"
      ? sum + event.allocatedPageIds.length
      : sum
  ), 0) ?? 0;
  const kvPagesReleased = pagedKvTrace?.reduce((sum, event) => (
    event.kind === "restore" ? sum + event.releasedPageIds.length : sum
  ), 0) ?? 0;
  const kvPageBytes = pagedKvSnapshot?.pageBytes ?? 0;

  const committedTokens = config.outputTokenCount;
  const metrics: SpeculativeWorkloadMetrics = {
    iterations: iterations.length,
    targetForwards: iterations.length,
    proposedDraftTokens,
    acceptedDraftTokens,
    rejectedDraftTokens: proposedDraftTokens - acceptedDraftTokens,
    committedTokens,
    correctionTokens,
    bonusTokens,
    targetOnlyTokens,
    committedTokensPerTargetForward:
      iterations.length === 0 ? 0 : committedTokens / iterations.length,
    acceptedPrefixHistogram,
    acceptanceByPosition: proposedByPosition.map((proposed, position) => (
      proposed === 0 ? 0 : acceptedByPosition[position] / proposed
    )),
    kvPagesAllocated,
    kvPagesReleased,
    kvHighWaterReservedBytes: pagedKvSnapshot
      ? Math.ceil(
          pagedKvSnapshot.highWaterTokenLength
          / pagedKvSnapshot.pageSizeTokens,
        ) * kvPageBytes
      : 0,
    kvFinalReservedBytes: pagedKvSnapshot?.reservedBytes ?? 0,
  };
  return {
    family: config.family,
    familyContract,
    initialTokenLength: config.initialTokenLength,
    finalTokenLength,
    targetOnlyFinalTokenLength,
    iterations,
    metrics,
    stateGroups: transaction.snapshot(),
    ...(pagedKvSnapshot && pagedKvTrace
      ? {
          pagedKv: {
            snapshot: pagedKvSnapshot,
            trace: pagedKvTrace,
          },
        }
      : {}),
  };
}

function chooseAcceptedPrefix(
  model: SpeculativeAcceptanceModel,
  rng: DeterministicRng | undefined,
  iteration: number,
  draftTokenCount: number,
): number {
  if (draftTokenCount === 0) {
    return 0;
  }
  if (model.kind === "replay") {
    const accepted = model.acceptedDraftTokens[iteration];
    if (accepted === undefined) {
      throw new SpeculativeWorkloadError(
        `acceptance replay ended before iteration ${iteration}`,
      );
    }
    if (
      !Number.isSafeInteger(accepted)
      || accepted < 0
      || accepted > draftTokenCount
    ) {
      throw new SpeculativeWorkloadError(
        `iteration ${iteration} accepted ${accepted} of ${draftTokenCount} drafts`,
      );
    }
    return accepted;
  }

  if (!rng) {
    throw new SpeculativeWorkloadError("conditional model lacks deterministic RNG");
  }
  let accepted = 0;
  for (let position = 0; position < draftTokenCount; position++) {
    const probability = model.matchProbabilityByPosition[position];
    if (rng.nextFloat() >= probability) {
      break;
    }
    accepted++;
  }
  return accepted;
}

function validateConfig(config: SpeculativeWorkloadConfig): void {
  assertNonNegative(config.initialTokenLength, "initialTokenLength");
  assertNonNegative(config.outputTokenCount, "outputTokenCount");
  assertNonNegative(config.maxAdditionalTokens, "maxAdditionalTokens");
  if (
    config.maxIterations !== undefined
    && (!Number.isSafeInteger(config.maxIterations) || config.maxIterations <= 0)
  ) {
    throw new SpeculativeWorkloadError(
      `maxIterations must be a positive safe integer; got ${config.maxIterations}`,
    );
  }
  if (config.acceptance.kind !== "replay") {
    assertNonNegative(config.acceptance.seed, "acceptance seed");
    if (
      config.acceptance.matchProbabilityByPosition.length
      < config.maxAdditionalTokens
    ) {
      throw new SpeculativeWorkloadError(
        "conditional acceptance probabilities do not cover the draft width",
      );
    }
    for (
      let position = 0;
      position < config.maxAdditionalTokens;
      position++
    ) {
      const probability =
        config.acceptance.matchProbabilityByPosition[position];
      if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
        throw new SpeculativeWorkloadError(
          `acceptance probability at position ${position} must be in [0, 1]`,
        );
      }
    }
  }
}

class DeterministicRng {
  private state: number;

  constructor(seed: number) {
    this.state = (seed >>> 0) || 0x9e3779b9;
  }

  nextFloat(): number {
    let value = this.state;
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    this.state = value >>> 0;
    return this.state / 0x1_0000_0000;
  }
}

function assertNonNegative(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new SpeculativeWorkloadError(
      `${label} must be a non-negative safe integer; got ${value}`,
    );
  }
}

function checkedAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    throw new SpeculativeWorkloadError(
      `${label} exceeds Number.MAX_SAFE_INTEGER`,
    );
  }
  return result;
}
