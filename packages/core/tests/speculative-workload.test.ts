import { describe, expect, it } from "vitest";
import {
  SpeculativeWorkloadError,
  simulateSpeculativeWorkload,
  type SpeculativeStateGroupConfig,
  type SpeculativeWorkloadConfig,
} from "../src/index.js";

const stateGroups: readonly SpeculativeStateGroupConfig[] = [
  {
    id: "target-csa",
    owner: "target",
    capacityTokens: 256,
    rollbackProtection: {
      kind: "bounded_snapshot",
      maxRollbackTokens: 4,
    },
  },
  {
    id: "target-kv",
    owner: "target",
    capacityTokens: 256,
    rollbackProtection: { kind: "non_destructive_tail" },
  },
  {
    id: "mtp-kv",
    owner: "proposer",
    capacityTokens: 256,
    rollbackProtection: { kind: "non_destructive_tail" },
  },
];

function replayConfig(): SpeculativeWorkloadConfig {
  return {
    family: "mtp",
    initialTokenLength: 20,
    outputTokenCount: 10,
    maxAdditionalTokens: 4,
    stateGroups,
    acceptance: {
      kind: "replay",
      acceptedDraftTokens: [0, 2, 4],
    },
  };
}

describe("simulateSpeculativeWorkload", () => {
  it("matches target-only length across correction, bonus, and tail iterations", () => {
    const result = simulateSpeculativeWorkload(replayConfig());

    expect(result.finalTokenLength).toBe(30);
    expect(result.targetOnlyFinalTokenLength).toBe(30);
    expect(result.iterations.map((iteration) => ({
      proposed: iteration.proposedDraftTokens,
      accepted: iteration.acceptedDraftTokens,
      committed: iteration.committedTokens,
      outcome: iteration.outcome,
    }))).toEqual([
      { proposed: 4, accepted: 0, committed: 1, outcome: "correction" },
      { proposed: 4, accepted: 2, committed: 3, outcome: "correction" },
      { proposed: 4, accepted: 4, committed: 5, outcome: "bonus" },
      { proposed: 0, accepted: 0, committed: 1, outcome: "target_only" },
    ]);
    expect(result.metrics).toMatchObject({
      iterations: 4,
      targetForwards: 4,
      proposedDraftTokens: 12,
      acceptedDraftTokens: 6,
      rejectedDraftTokens: 6,
      committedTokens: 10,
      correctionTokens: 2,
      bonusTokens: 1,
      targetOnlyTokens: 1,
      committedTokensPerTargetForward: 2.5,
      acceptedPrefixHistogram: [2, 0, 1, 0, 1],
    });
    expect(result.metrics.acceptanceByPosition).toEqual([
      2 / 3,
      2 / 3,
      1 / 3,
      1 / 3,
    ]);
    expect(
      result.stateGroups.every((group) => group.logicalLength === 30),
    ).toBe(true);
  });

  it("is byte-for-byte deterministic for a seeded conditional model", () => {
    const config: SpeculativeWorkloadConfig = {
      ...replayConfig(),
      outputTokenCount: 40,
      acceptance: {
        kind: "conditional_empirical",
        matchProbabilityByPosition: [0.8, 0.7, 0.6, 0.5],
        seed: 42,
      },
    };

    expect(JSON.stringify(simulateSpeculativeWorkload(config))).toBe(
      JSON.stringify(simulateSpeculativeWorkload(config)),
    );
  });

  it("keeps paged KV aligned through speculative page allocation and rollback", () => {
    const result = simulateSpeculativeWorkload({
      ...replayConfig(),
      pagedKv: {
        sequenceId: "target-sequence",
        pageSizeTokens: 4,
        bytesPerToken: 8,
        capacityBytes: 12 * 4 * 8,
      },
    });

    expect(result.pagedKv?.snapshot.logicalTokenLength).toBe(30);
    expect(result.pagedKv?.snapshot.livePages).toHaveLength(8);
    expect(result.pagedKv?.snapshot.livePages.at(-1)?.validTokens).toBe(2);
    expect(result.metrics.kvPagesAllocated).toBeGreaterThanOrEqual(8);
    expect(result.metrics.kvPagesReleased).toBeGreaterThan(0);
    expect(result.metrics.kvFinalReservedBytes).toBe(8 * 4 * 8);
    expect(
      result.stateGroups.every(
        (group) => group.logicalLength === result.pagedKv?.snapshot.logicalTokenLength,
      ),
    ).toBe(true);
  });

  it("models conditional probabilities as first-mismatch, not averages", () => {
    const alwaysMatch = simulateSpeculativeWorkload({
      ...replayConfig(),
      outputTokenCount: 9,
      acceptance: {
        kind: "conditional_heuristic",
        matchProbabilityByPosition: [1, 1, 1, 1],
        seed: 1,
      },
    });
    expect(alwaysMatch.metrics.acceptedDraftTokens).toBe(7);
    expect(alwaysMatch.metrics.bonusTokens).toBe(2);

    const neverMatch = simulateSpeculativeWorkload({
      ...replayConfig(),
      outputTokenCount: 3,
      acceptance: {
        kind: "conditional_heuristic",
        matchProbabilityByPosition: [0, 1, 1, 1],
        seed: 1,
      },
    });
    expect(neverMatch.metrics.acceptedDraftTokens).toBe(0);
    expect(neverMatch.metrics.iterations).toBe(3);
  });

  it("rejects an impossible replay prefix at its iteration", () => {
    expect(() => simulateSpeculativeWorkload({
      ...replayConfig(),
      acceptance: {
        kind: "replay",
        acceptedDraftTokens: [5],
      },
    })).toThrowError(
      "iteration 0 accepted 5 of 4 drafts",
    );
  });

  it("propagates rollback-horizon protection instead of widening silently", () => {
    expect(() => simulateSpeculativeWorkload({
      ...replayConfig(),
      maxAdditionalTokens: 5,
      acceptance: {
        kind: "replay",
        acceptedDraftTokens: [0],
      },
    })).toThrowError(
      "target-csa rollback horizon 5 exceeds snapshot bound 4",
    );
  });

  it("bounds pathological iteration counts", () => {
    expect(() => simulateSpeculativeWorkload({
      ...replayConfig(),
      maxIterations: 1,
    })).toThrowError(SpeculativeWorkloadError);
  });
});
