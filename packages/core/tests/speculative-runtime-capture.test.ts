import { describe, expect, it } from "vitest";
import {
  bindParsedRuntimeCaptures,
  bindSpeculativeRuntimeCaptures,
  parseRuntimeTokenCapture,
  type SpeculativeRuntimeCapture,
  type TargetOnlyRuntimeCapture,
} from "../src/index.js";

const provenance = {
  source: "onnx-genai-conformance",
  runtimeRevision: "onnx-genai@abc123",
  modelFingerprint: "sha256:target",
  tokenizerFingerprint: "sha256:tokenizer",
  generationConfigFingerprint: "sha256:greedy-max-token-config",
} as const;

const targetOnly: TargetOnlyRuntimeCapture = {
  revision: 1,
  id: "target-run-001",
  role: "target_only",
  provenance,
  completionReason: "max_tokens",
  promptTokenIds: [101, 102, 103],
  outputTokenIds: [10, 20, 21, 30, 31, 32, 40, 41],
  terminal: {
    status: "completed",
    outputTokenCount: 8,
    iterationCount: 0,
  },
};

const speculative: SpeculativeRuntimeCapture = {
  revision: 1,
  id: "speculative-run-001",
  role: "speculative",
  provenance,
  completionReason: "max_tokens",
  promptTokenIds: [101, 102, 103],
  outputTokenIds: [10, 20, 21, 30, 31, 32, 40, 41],
  terminal: {
    status: "completed",
    outputTokenCount: 8,
    iterationCount: 3,
  },
  family: "mtp",
  proposerFingerprint: "sha256:mtp",
  maxAdditionalTokens: 2,
  iterations: [
    {
      id: "iteration-0",
      outputOffset: 0,
      proposalTokenIds: [10, 99, 100],
      targetTokenIds: [10, 20],
      committedTokenIds: [10, 20],
    },
    {
      id: "iteration-1",
      outputOffset: 2,
      proposalTokenIds: [21, 30, 31],
      targetTokenIds: [21, 30, 31, 32],
      committedTokenIds: [21, 30, 31, 32],
    },
    {
      id: "iteration-2",
      outputOffset: 6,
      proposalTokenIds: [40, 41],
      targetTokenIds: [40, 41],
      committedTokenIds: [40, 41],
    },
  ],
};

describe("speculative runtime captures", () => {
  it("binds two completed runs and independently verifies iteration commits", () => {
    const bound = bindSpeculativeRuntimeCaptures(targetOnly, speculative);

    expect(bound.trace.revision).toBe(2);
    expect(bound.result.differential.matchesTargetOnly).toBe(true);
    expect(bound.result.iterations.map((iteration) => iteration.outcome))
      .toEqual(["correction", "bonus", "accepted_tail"]);
  });

  it("parses snake-case capture artifacts and binds their run identities", () => {
    const targetInput = toInput(targetOnly);
    const speculativeInput = toInput(speculative);
    expect(parseRuntimeTokenCapture(targetInput).role).toBe("target_only");

    const bound = bindParsedRuntimeCaptures(targetInput, speculativeInput);
    expect(bound.targetOnlyCaptureId).toBe("target-run-001");
    expect(bound.speculativeCaptureId).toBe("speculative-run-001");
  });

  it("fails closed on incomplete, misbound, or reordered evidence", () => {
    expect(() => parseRuntimeTokenCapture({
      runtime_token_capture: {
        ...toInput(targetOnly).runtime_token_capture,
        terminal: {
          status: "completed",
          output_token_count: 7,
          iteration_count: 0,
        },
      },
    })).toThrow("output token count does not match");

    expect(() => bindSpeculativeRuntimeCaptures(targetOnly, {
      ...speculative,
      provenance: {
        ...speculative.provenance,
        modelFingerprint: "sha256:different-model",
      },
    })).toThrow("provenance mismatch for modelFingerprint");

    expect(() => bindSpeculativeRuntimeCaptures(targetOnly, {
      ...speculative,
      iterations: [
        {
          ...speculative.iterations[0]!,
          outputOffset: 1,
        },
        ...speculative.iterations.slice(1),
      ],
    })).toThrow("does not match derived offset");
  });

  it("rejects runtime claims that disagree with the independent oracle", () => {
    expect(() => bindSpeculativeRuntimeCaptures(targetOnly, {
      ...speculative,
      iterations: [
        {
          ...speculative.iterations[0]!,
          committedTokenIds: [10, 99],
        },
        ...speculative.iterations.slice(1),
      ],
    })).toThrow("runtime commits differ from the acceptance oracle");

    expect(() => bindSpeculativeRuntimeCaptures(targetOnly, {
      ...speculative,
      outputTokenIds: [10, 99, 21, 30, 31, 32, 40, 41],
    })).toThrow("runtime speculative output differs");
  });

  it("reports a well-formed differential without calling the evidence malformed", () => {
    const bound = bindSpeculativeRuntimeCaptures({
      ...targetOnly,
      outputTokenIds: [10, 999, 21, 30, 31, 32, 40, 41],
    }, speculative);

    expect(bound.result.differential).toEqual({
      matchesTargetOnly: false,
      comparedTokenCount: 8,
      firstMismatch: {
        outputIndex: 1,
        expectedTokenId: 999,
        actualTokenId: 20,
      },
    });
  });

  it("does not admit unsupported early-termination captures", () => {
    const input = toInput(targetOnly);
    input.runtime_token_capture.completion_reason = "eos";

    expect(() => parseRuntimeTokenCapture(input))
      .toThrow("completion_reason must be max_tokens");
  });
});

function toInput(
  capture: TargetOnlyRuntimeCapture | SpeculativeRuntimeCapture,
): {
  runtime_token_capture: Record<string, unknown>;
} {
  const common: Record<string, unknown> = {
    revision: capture.revision,
    id: capture.id,
    role: capture.role,
    provenance: {
      source: capture.provenance.source,
      runtime_revision: capture.provenance.runtimeRevision,
      model_fingerprint: capture.provenance.modelFingerprint,
      tokenizer_fingerprint: capture.provenance.tokenizerFingerprint,
      generation_config_fingerprint:
        capture.provenance.generationConfigFingerprint,
    },
    completion_reason: capture.completionReason,
    prompt_token_ids: capture.promptTokenIds,
    output_token_ids: capture.outputTokenIds,
    terminal: {
      status: capture.terminal.status,
      output_token_count: capture.terminal.outputTokenCount,
      iteration_count: capture.terminal.iterationCount,
    },
  };
  if (capture.role === "speculative") {
    common.family = capture.family;
    common.proposer_fingerprint = capture.proposerFingerprint;
    common.max_additional_tokens = capture.maxAdditionalTokens;
    common.iterations = capture.iterations.map((iteration) => ({
      id: iteration.id,
      output_offset: iteration.outputOffset,
      proposal_token_ids: iteration.proposalTokenIds,
      target_token_ids: iteration.targetTokenIds,
      committed_token_ids: iteration.committedTokenIds,
    }));
  }
  return { runtime_token_capture: common };
}
