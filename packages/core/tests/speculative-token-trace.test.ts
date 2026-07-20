import { describe, expect, it } from "vitest";
import {
  SCENARIO_PRESET_NAMES,
  buildScenarioPreset,
  parseSpeculativeTokenTrace,
  simulateSpeculativeTokenTrace,
  simulateTopologyWorkload,
  topologyProfileFromSpeculative,
  type SpeculativeProposerFamily,
  type SpeculativeTokenTrace,
} from "../src/index.js";

const provenance = {
  source: "synthetic-test",
  runtimeRevision: "onnx-genai-test",
  modelFingerprint: "target-model-test",
  proposerFingerprint: "proposer-test",
  tokenizerFingerprint: "tokenizer-test",
  generationConfigFingerprint: "greedy-test",
  targetOnlyRunId: "target-only-test",
  speculativeRunId: "speculative-test",
} as const;

const mtpTrace: SpeculativeTokenTrace = {
  revision: 1,
  id: "mtp-correction-bonus-tail",
  provenance,
  family: "mtp",
  promptTokenIds: [101, 102, 103],
  expectedOutputTokenIds: [10, 20, 21, 30, 31, 32, 40, 41],
  maxAdditionalTokens: 2,
  iterations: [
    {
      id: "correction",
      proposalTokenIds: [10, 99, 100],
      targetTokenIds: [10, 20, 200, 201],
    },
    {
      id: "bonus",
      proposalTokenIds: [21, 30, 31],
      targetTokenIds: [21, 30, 31, 32],
    },
    {
      id: "accepted-tail",
      proposalTokenIds: [40, 41],
      targetTokenIds: [40, 41, 42],
    },
  ],
};

const families: readonly SpeculativeProposerFamily[] = [
  "prompt_lookup",
  "draft_model",
  "mtp",
  "eagle3",
  "shared_kv",
  "self_speculative",
];

describe("speculative token-value traces", () => {
  it("reconstructs correction, bonus, and accepted-tail token values", () => {
    const result = simulateSpeculativeTokenTrace(mtpTrace);

    expect(result.committedOutputTokenIds).toEqual(
      mtpTrace.expectedOutputTokenIds,
    );
    expect(result.differential).toEqual({
      matchesTargetOnly: true,
      comparedTokenCount: 8,
    });
    expect(result.iterations.map((iteration) => ({
      id: iteration.id,
      accepted: iteration.acceptedDraftTokens,
      committed: iteration.committedTokenIds,
      outcome: iteration.outcome,
    }))).toEqual([
      {
        id: "correction",
        accepted: 1,
        committed: [10, 20],
        outcome: "correction",
      },
      {
        id: "bonus",
        accepted: 3,
        committed: [21, 30, 31, 32],
        outcome: "bonus",
      },
      {
        id: "accepted-tail",
        accepted: 2,
        committed: [40, 41],
        outcome: "accepted_tail",
      },
    ]);
    expect(result.workload.finalTokenLength).toBe(11);
    expect(result.workload.stateGroups.every((group) => (
      group.logicalLength === (
        group.lifetime === "committed_prefix" ? 11 : 0
      )
    ))).toBe(true);
  });

  it("reports the first value mismatch against target-only output", () => {
    const result = simulateSpeculativeTokenTrace({
      ...mtpTrace,
      expectedOutputTokenIds: [10, 999, 21, 30, 31, 32, 40, 41],
    });

    expect(result.differential).toEqual({
      matchesTargetOnly: false,
      comparedTokenCount: 8,
      firstMismatch: {
        outputIndex: 1,
        expectedTokenId: 999,
        actualTokenId: 20,
      },
    });
  });

  it("fails closed on a rejected guaranteed prefix", () => {
    expect(() => simulateSpeculativeTokenTrace({
      ...mtpTrace,
      iterations: [{
        ...mtpTrace.iterations[0]!,
        proposalTokenIds: [11, 99, 100],
      }, ...mtpTrace.iterations.slice(1)],
    })).toThrow("rejects its guaranteed target proposal prefix");
  });

  it("requires distinct, fully bound target-only and speculative evidence", () => {
    expect(() => simulateSpeculativeTokenTrace({
      ...mtpTrace,
      provenance: {
        ...mtpTrace.provenance,
        speculativeRunId: mtpTrace.provenance.targetOnlyRunId,
      },
    })).toThrow("must use distinct run ids");
  });

  it("rejects missing target rows and trailing iterations", () => {
    expect(() => simulateSpeculativeTokenTrace({
      ...mtpTrace,
      iterations: [{
        ...mtpTrace.iterations[0]!,
        targetTokenIds: [10, 20, 200],
      }, ...mtpTrace.iterations.slice(1)],
    })).toThrow("exactly one target token per proposal position");

    expect(() => simulateSpeculativeTokenTrace({
      ...mtpTrace,
      iterations: [
        ...mtpTrace.iterations,
        {
          id: "extra",
          proposalTokenIds: [50],
          targetTokenIds: [50, 51],
        },
      ],
    })).toThrow("after the output budget was complete");
  });

  it("parses a revisioned snake-case contract and rejects typos", () => {
    const parsed = parseSpeculativeTokenTrace({
      speculative_token_trace: {
        revision: 1,
        id: "parsed",
        provenance: {
          source: "synthetic-test",
          runtime_revision: "onnx-genai-test",
          model_fingerprint: "target-model-test",
          proposer_fingerprint: "draft-model-test",
          tokenizer_fingerprint: "tokenizer-test",
          generation_config_fingerprint: "greedy-test",
          target_only_run_id: "target-only-test",
          speculative_run_id: "speculative-test",
        },
        family: "draft_model",
        prompt_token_ids: [1],
        expected_output_token_ids: [2],
        max_additional_tokens: 1,
        iterations: [{
          id: "tail",
          proposal_token_ids: [2],
          target_token_ids: [2, 3],
        }],
      },
    });
    expect(simulateSpeculativeTokenTrace(parsed).differential.matchesTargetOnly)
      .toBe(true);

    expect(() => parseSpeculativeTokenTrace({
      speculative_token_trace: {
        revision: 1,
        id: "typo",
        provenance: {
          source: "synthetic-test",
          runtime_revision: "onnx-genai-test",
          model_fingerprint: "target-model-test",
          proposer_fingerprint: "draft-model-test",
          tokenizer_fingerprint: "tokenizer-test",
          generation_config_fingerprint: "greedy-test",
          target_only_run_id: "target-only-test",
          speculative_run_id: "speculative-test",
        },
        family: "draft_model",
        prompt_token_ids: [1],
        expected_output_token_ids: [2],
        max_additional_tokens: 1,
        iteration: [],
      },
    })).toThrow("unknown field iteration");
  });

  it("executes value-verified traces for every proposer and topology", () => {
    for (const family of families) {
      const result = simulateSpeculativeTokenTrace(traceForFamily(family));
      expect(result.differential.matchesTargetOnly, family).toBe(true);
      const profile = topologyProfileFromSpeculative(result.workload);
      for (const scenarioName of SCENARIO_PRESET_NAMES) {
        const topology = simulateTopologyWorkload(
          buildScenarioPreset(scenarioName),
          profile,
        );
        expect(topology.execution.status, `${family}:${scenarioName}`)
          .toBe("succeeded");
        expect(topology.metrics.committedTokens, `${family}:${scenarioName}`)
          .toBe(5);
      }
    }
  });
});

function traceForFamily(
  family: SpeculativeProposerFamily,
): SpeculativeTokenTrace {
  const guaranteed =
    family === "mtp" || family === "eagle3" || family === "shared_kv";
  return {
    revision: 1,
    id: `matrix-${family}`,
    provenance: {
      ...provenance,
      proposerFingerprint: `proposer-${family}`,
      targetOnlyRunId: `target-${family}`,
      speculativeRunId: `speculative-${family}`,
    },
    family,
    promptTokenIds: [101, 102],
    expectedOutputTokenIds: [1, 2, 3, 4, 5],
    maxAdditionalTokens: 2,
    iterations: guaranteed
      ? [
          {
            id: "bonus",
            proposalTokenIds: [1, 2, 3],
            targetTokenIds: [1, 2, 3, 4],
          },
          {
            id: "tail",
            proposalTokenIds: [5],
            targetTokenIds: [5, 6],
          },
        ]
      : [
          {
            id: "bonus",
            proposalTokenIds: [1, 2],
            targetTokenIds: [1, 2, 3],
          },
          {
            id: "tail",
            proposalTokenIds: [4, 5],
            targetTokenIds: [4, 5, 6],
          },
        ],
  };
}
