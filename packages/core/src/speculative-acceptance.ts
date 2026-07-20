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

export class SpeculativeAcceptanceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SpeculativeAcceptanceError";
  }
}

export class SpeculativeAcceptanceCursor {
  private iteration = 0;
  private readonly streamHash: number;

  constructor(
    private readonly model: SpeculativeAcceptanceModel,
    streamId = "default",
  ) {
    validateAcceptanceModel(model);
    this.streamHash = hashString(streamId);
  }

  next(draftTokenCount: number): number {
    assertNonNegativeSafeInteger(draftTokenCount, "draftTokenCount");
    const iteration = this.iteration++;
    if (draftTokenCount === 0) {
      return 0;
    }
    if (this.model.kind === "replay") {
      const accepted = this.model.acceptedDraftTokens[iteration];
      if (accepted === undefined) {
        throw new SpeculativeAcceptanceError(
          `acceptance replay ended before iteration ${iteration}`,
        );
      }
      if (
        !Number.isSafeInteger(accepted)
        || accepted < 0
        || accepted > draftTokenCount
      ) {
        throw new SpeculativeAcceptanceError(
          `iteration ${iteration} accepted ${accepted} of ${draftTokenCount} drafts`,
        );
      }
      return accepted;
    }

    let accepted = 0;
    for (let position = 0; position < draftTokenCount; position++) {
      const probability = this.model.matchProbabilityByPosition[position];
      if (probability === undefined) {
        throw new SpeculativeAcceptanceError(
          `acceptance probabilities do not cover draft position ${position}`,
        );
      }
      if (
        deterministicFloat(
          this.model.seed,
          this.streamHash,
          iteration,
          position,
        ) >= probability
      ) {
        break;
      }
      accepted++;
    }
    return accepted;
  }
}

export function validateAcceptanceCoverage(
  model: SpeculativeAcceptanceModel,
  maxAdditionalTokens: number,
): void {
  assertNonNegativeSafeInteger(maxAdditionalTokens, "maxAdditionalTokens");
  if (
    model.kind !== "replay"
    && model.matchProbabilityByPosition.length < maxAdditionalTokens
  ) {
    throw new SpeculativeAcceptanceError(
      "conditional acceptance probabilities do not cover the draft width",
    );
  }
}

function validateAcceptanceModel(model: SpeculativeAcceptanceModel): void {
  if (model.kind === "replay") {
    return;
  }
  assertNonNegativeSafeInteger(model.seed, "acceptance seed");
  for (
    let position = 0;
    position < model.matchProbabilityByPosition.length;
    position++
  ) {
    const probability = model.matchProbabilityByPosition[position];
    if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
      throw new SpeculativeAcceptanceError(
        `acceptance probability at position ${position} must be in [0, 1]`,
      );
    }
  }
}

function hashString(streamId: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < streamId.length; index++) {
    hash ^= streamId.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function deterministicFloat(
  seed: number,
  streamHash: number,
  iteration: number,
  position: number,
): number {
  let value = (seed >>> 0) ^ streamHash;
  value = Math.imul(value ^ (iteration >>> 0), 0x85ebca6b);
  value = Math.imul(value ^ (position >>> 0), 0xc2b2ae35);
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb352d);
  value ^= value >>> 15;
  value = Math.imul(value, 0x846ca68b);
  value ^= value >>> 16;
  return (value >>> 0) / 0x1_0000_0000;
}

function assertNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new SpeculativeAcceptanceError(
      `${label} must be a non-negative safe integer; got ${value}`,
    );
  }
}
