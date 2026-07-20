import { describe, expect, it } from "vitest";
import {
  SpeculativeProtocolError,
  SpeculativeTransactionSimulator,
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
