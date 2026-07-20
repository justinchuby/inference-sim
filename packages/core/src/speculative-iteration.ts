import type { SpeculativeProposalPrefix } from "./speculative-family.js";

export const SPECULATIVE_ITERATION_CONTRACT_REVISION = 1;

export interface SpeculativeProposalShape {
  readonly guaranteedTargetTokens: 0 | 1;
  readonly proposedAdditionalTokens: number;
  readonly proposedDraftTokens: number;
  readonly targetTokenWidth: number;
}

export interface SpeculativeIterationDecision extends SpeculativeProposalShape {
  readonly acceptedAdditionalTokens: number;
  readonly acceptedDraftTokens: number;
  readonly rejectedDraftTokens: number;
  readonly targetAuthoritativeTokens: 0 | 1;
  readonly committedTokens: number;
  readonly outcome:
    | "target_only"
    | "correction"
    | "bonus"
    | "accepted_tail";
}

export class SpeculativeIterationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SpeculativeIterationError";
  }
}

export function planSpeculativeProposal(input: {
  readonly proposalPrefix: SpeculativeProposalPrefix;
  readonly remainingOutputTokens: number;
  readonly maxAdditionalTokens: number;
  readonly maxTargetTokenWidth?: number;
}): SpeculativeProposalShape {
  assertPositiveSafeInteger(
    input.remainingOutputTokens,
    "remainingOutputTokens",
  );
  assertNonNegativeSafeInteger(
    input.maxAdditionalTokens,
    "maxAdditionalTokens",
  );
  if (input.maxTargetTokenWidth !== undefined) {
    assertPositiveSafeInteger(
      input.maxTargetTokenWidth,
      "maxTargetTokenWidth",
    );
  }

  const maxProposalTokens = Math.min(
    input.remainingOutputTokens,
    input.maxTargetTokenWidth === undefined
      ? input.remainingOutputTokens
      : Math.max(0, input.maxTargetTokenWidth - 1),
  );
  const guaranteedTargetTokens: 0 | 1 =
    input.proposalPrefix === "guaranteed_target"
    && maxProposalTokens >= 1
      ? 1
      : 0;
  const proposedAdditionalTokens =
    input.proposalPrefix === "guaranteed_target"
    && guaranteedTargetTokens === 0
      ? 0
      : Math.min(
          input.maxAdditionalTokens,
          maxProposalTokens - guaranteedTargetTokens,
        );
  const proposedDraftTokens =
    guaranteedTargetTokens + proposedAdditionalTokens;
  return {
    guaranteedTargetTokens,
    proposedAdditionalTokens,
    proposedDraftTokens,
    targetTokenWidth: proposedDraftTokens + 1,
  };
}

export function decideSpeculativeIteration(
  proposal: SpeculativeProposalShape,
  acceptedAdditionalTokens: number,
  remainingOutputTokens: number,
): SpeculativeIterationDecision {
  assertNonNegativeSafeInteger(
    acceptedAdditionalTokens,
    "acceptedAdditionalTokens",
  );
  assertPositiveSafeInteger(remainingOutputTokens, "remainingOutputTokens");
  if (acceptedAdditionalTokens > proposal.proposedAdditionalTokens) {
    throw new SpeculativeIterationError(
      `accepted ${acceptedAdditionalTokens} additional tokens but only ${proposal.proposedAdditionalTokens} were proposed`,
    );
  }
  if (proposal.proposedDraftTokens > remainingOutputTokens) {
    throw new SpeculativeIterationError(
      "proposal exceeds the remaining output budget",
    );
  }

  const acceptedDraftTokens =
    proposal.guaranteedTargetTokens + acceptedAdditionalTokens;
  const targetAuthoritativeTokens: 0 | 1 =
    proposal.proposedDraftTokens > 0
    && acceptedDraftTokens === proposal.proposedDraftTokens
    && proposal.proposedDraftTokens === remainingOutputTokens
      ? 0
      : 1;
  const outcome: SpeculativeIterationDecision["outcome"] =
    proposal.proposedDraftTokens === 0
      ? "target_only"
      : acceptedDraftTokens < proposal.proposedDraftTokens
        ? "correction"
        : targetAuthoritativeTokens === 0
          ? "accepted_tail"
          : "bonus";
  return {
    ...proposal,
    acceptedAdditionalTokens,
    acceptedDraftTokens,
    rejectedDraftTokens:
      proposal.proposedDraftTokens - acceptedDraftTokens,
    targetAuthoritativeTokens,
    committedTokens: acceptedDraftTokens + targetAuthoritativeTokens,
    outcome,
  };
}

function assertNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new SpeculativeIterationError(
      `${label} must be a non-negative safe integer; got ${value}`,
    );
  }
}

function assertPositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new SpeculativeIterationError(
      `${label} must be a positive safe integer; got ${value}`,
    );
  }
}
