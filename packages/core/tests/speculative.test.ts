import { describe, expect, it } from "vitest";
import {
  SpeculativeProtocolError,
  SpeculativeTransactionSimulator,
  decideSpeculativeIteration,
  planSpeculativeProposal,
  type SpeculativeStateGroupConfig,
} from "../src/index.js";

const groups: readonly SpeculativeStateGroupConfig[] = [
  {
    id: "target-csa",
    owner: "target",
    capacityTokens: 128,
    rollbackProtection: {
      kind: "bounded_snapshot",
      maxRollbackTokens: 4,
    },
  },
  {
    id: "target-paged-kv",
    owner: "target",
    capacityTokens: 128,
    rollbackProtection: { kind: "non_destructive_tail" },
  },
  {
    id: "mtp-kv",
    owner: "proposer",
    capacityTokens: 128,
    rollbackProtection: { kind: "non_destructive_tail" },
  },
  {
    id: "mtp-recurrent",
    owner: "proposer",
    capacityTokens: 128,
    rollbackProtection: {
      kind: "bounded_snapshot",
      maxRollbackTokens: 4,
    },
  },
];

describe("SpeculativeTransactionSimulator", () => {
  it("commits one target correction when every draft is rejected", () => {
    const simulator = new SpeculativeTransactionSimulator(20, groups);
    const result = simulator.runIteration({
      draftTokenCount: 4,
      acceptedDraftTokenCount: 0,
    });

    expect(result.committedTokenCount).toBe(1);
    expect(result.rejectedDraftTokenCount).toBe(4);
    expect(result.finalTokenLength).toBe(21);
    expect(result.stateGroups.every((group) => group.logicalLength === 21)).toBe(true);
    expect(
      result.stateGroups.find((group) => group.id === "target-csa")
        ?.highWaterLength,
    ).toBe(24);
    expect(
      result.stateGroups.find((group) => group.id === "mtp-kv")
        ?.highWaterLength,
    ).toBe(24);
  });

  it("commits accepted drafts plus one correction or bonus token", () => {
    const simulator = new SpeculativeTransactionSimulator(10, groups);

    const partial = simulator.runIteration({
      draftTokenCount: 4,
      acceptedDraftTokenCount: 2,
    });
    expect(partial.committedTokenCount).toBe(3);
    expect(partial.finalTokenLength).toBe(13);

    const full = simulator.runIteration({
      draftTokenCount: 4,
      acceptedDraftTokenCount: 4,
    });
    expect(full.committedTokenCount).toBe(5);
    expect(full.finalTokenLength).toBe(18);
    expect(simulator.snapshot().every((group) => group.logicalLength === 18)).toBe(true);
  });

  it("commits a fully accepted output tail without inventing a bonus token", () => {
    const simulator = new SpeculativeTransactionSimulator(10, groups);
    const result = simulator.runIteration({
      draftTokenCount: 2,
      acceptedDraftTokenCount: 2,
      targetAuthoritativeTokenCount: 0,
    });

    expect(result.targetAuthoritativeTokenCount).toBe(0);
    expect(result.committedTokenCount).toBe(2);
    expect(result.finalTokenLength).toBe(12);
    expect(simulator.snapshot().every((group) => group.logicalLength === 12))
      .toBe(true);
  });

  it("rejects a correction path that omits its target token", () => {
    const simulator = new SpeculativeTransactionSimulator(10, groups);

    expect(() => simulator.runIteration({
      draftTokenCount: 2,
      acceptedDraftTokenCount: 1,
      targetAuthoritativeTokenCount: 0,
    })).toThrow(
      "zero target-authoritative tokens require a non-empty, fully accepted proposal",
    );
    expect(simulator.tokenLength).toBe(10);
  });

  it("separates target verification width from proposal-local sidecars", () => {
    const separated: readonly SpeculativeStateGroupConfig[] = [
      {
        id: "target",
        owner: "target",
        capacityTokens: 32,
        rollbackProtection: {
          kind: "bounded_snapshot",
          maxRollbackTokens: 5,
        },
      },
      {
        id: "sidecar",
        owner: "proposer",
        lifetime: "proposal_local",
        capacityTokens: 4,
        rollbackProtection: {
          kind: "bounded_snapshot",
          maxRollbackTokens: 4,
        },
      },
    ];
    const simulator = new SpeculativeTransactionSimulator(10, separated);

    expect(simulator.runIteration({
      draftTokenCount: 5,
      acceptedDraftTokenCount: 1,
      proposalLocalTokenCount: 4,
    }).committedTokenCount).toBe(2);
  });

  it("plans guaranteed-prefix and accepted-tail semantics centrally", () => {
    const proposal = planSpeculativeProposal({
      proposalPrefix: "guaranteed_target",
      remainingOutputTokens: 4,
      maxAdditionalTokens: 4,
    });
    const decision = decideSpeculativeIteration(proposal, 3, 4);

    expect(proposal).toEqual({
      guaranteedTargetTokens: 1,
      proposedAdditionalTokens: 3,
      proposedDraftTokens: 4,
      targetTokenWidth: 5,
    });
    expect(decision).toMatchObject({
      acceptedDraftTokens: 4,
      targetAuthoritativeTokens: 0,
      committedTokens: 4,
      outcome: "accepted_tail",
    });
  });

  it("preflights capacity atomically before speculative writes", () => {
    const constrained = groups.map((group) => ({
      ...group,
      capacityTokens: 11,
    }));
    const simulator = new SpeculativeTransactionSimulator(10, constrained);

    expect(() => simulator.runIteration({
      draftTokenCount: 2,
      acceptedDraftTokenCount: 0,
    })).toThrowError(SpeculativeProtocolError);
    expect(simulator.tokenLength).toBe(10);
    expect(simulator.snapshot().every((group) => (
      group.logicalLength === 10 && group.highWaterLength === 10
    ))).toBe(true);
  });

  it("rejects a verification width beyond the rollback snapshot", () => {
    const simulator = new SpeculativeTransactionSimulator(10, groups);

    expect(() => simulator.runIteration({
      draftTokenCount: 5,
      acceptedDraftTokenCount: 1,
    })).toThrowError(
      "target-csa rollback horizon 5 exceeds snapshot bound 4",
    );
    expect(simulator.tokenLength).toBe(10);
  });

  it("rejects impossible acceptance counts", () => {
    const simulator = new SpeculativeTransactionSimulator(10, groups);

    expect(() => simulator.runIteration({
      draftTokenCount: 2,
      acceptedDraftTokenCount: 3,
    })).toThrowError(
      "accepted 3 drafts but only 2 were proposed",
    );
  });
});
