import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { runCli, type CliIo } from "../src/main.js";

function captureIo(): {
  io: CliIo;
  stdout: () => string;
  stderr: () => string;
} {
  let stdout = "";
  let stderr = "";
  return {
    io: {
      stdout: (text) => {
        stdout += text;
      },
      stderr: (text) => {
        stderr += text;
      },
    },
    stdout: () => stdout,
    stderr: () => stderr,
  };
}

describe("CLI", () => {
  it("lists scenario, hardware, and model presets", async () => {
    const capture = captureIo();
    expect(await runCli(["presets"], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      scenarios: string[];
      hardware: { topologies: string[] };
      models: string[];
    };
    expect(output.scenarios).toContain("gpu-npu");
    expect(output.hardware.topologies).toContain("dgx-h100");
    expect(output.models).toContain("deepseek-v2");
  });

  it("prints a validated scenario and exact ledger", async () => {
    const capture = captureIo();
    expect(await runCli(["scenario", "unified-memory"], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      scenario: { family: string };
      memoryLedger: Array<{ reservedBytes: number }>;
    };
    expect(output.scenario.family).toBe("unified");
    expect(output.memoryLedger[0].reservedBytes).toBe(
      76 * 1024 ** 3 + 256 * 1024 ** 2,
    );
  });

  it("runs a speculative YAML workload deterministically", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "speculative.yaml");
    await writeFile(path, `
speculative:
  family: mtp
  initial_token_length: 10
  output_token_count: 6
  max_additional_tokens: 2
  acceptance:
    kind: replay
    accepted_draft_tokens: [2, 1]
  paged_kv:
    page_size_tokens: 4
    bytes_per_token: 128
    capacity_bytes: 4096
`, "utf8");
    const capture = captureIo();
    expect(await runCli(["speculative", path], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      finalTokenLength: number;
      metrics: { committedTokens: number };
      pagedKv: {
        snapshot: {
          logicalTokenLength: number;
          reservedBytes: number;
        };
      };
    };
    expect(output.finalTokenLength).toBe(16);
    expect(output.metrics.committedTokens).toBe(6);
    expect(output.pagedKv.snapshot.logicalTokenLength).toBe(16);
    expect(output.pagedKv.snapshot.reservedBytes).toBe(2048);
  });

  it("verifies token values and optionally executes their topology", async () => {
    const path = new URL(
      "../../../examples/speculative-token-trace-mtp.yaml",
      import.meta.url,
    ).pathname;
    const capture = captureIo();

    expect(await runCli([
      "speculative-trace",
      path,
      "single-gpu-cpu",
    ], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      trace: {
        differential: { matchesTargetOnly: boolean };
        committedOutputTokenIds: number[];
      };
      topology: {
        execution: { status: string };
        metrics: { committedTokens: number };
      };
    };
    expect(output.trace.differential.matchesTargetOnly).toBe(true);
    expect(output.trace.committedOutputTokenIds)
      .toEqual([10, 20, 21, 30, 31, 32, 40, 41]);
    expect(output.topology.execution.status).toBe("succeeded");
    expect(output.topology.metrics.committedTokens).toBe(8);
  });

  it("returns status 2 with the first token differential", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "mismatch.yaml");
    await writeFile(path, `
speculative_token_trace:
  revision: 2
  id: mismatch
  provenance:
    source: synthetic-test
    runtime_revision: onnx-genai-test
    model_fingerprint: target-model-test
    proposer_fingerprint: draft-model-test
    tokenizer_fingerprint: tokenizer-test
    generation_config_fingerprint: greedy-test
    target_only_run_id: target-only-test
    speculative_run_id: speculative-test
  family: draft_model
  prompt_token_ids: [1]
  expected_output_token_ids: [7]
  max_additional_tokens: 1
  iterations:
    - id: tail
      proposal_token_ids: [8]
      target_token_ids: [8]
`, "utf8");
    const capture = captureIo();

    expect(await runCli(["speculative-trace", path], capture.io)).toBe(2);
    const output = JSON.parse(capture.stdout()) as {
      differential: {
        matchesTargetOnly: boolean;
        firstMismatch: {
          outputIndex: number;
          expectedTokenId: number;
          actualTokenId: number;
        };
      };
    };
    expect(output.differential).toEqual({
      matchesTargetOnly: false,
      comparedTokenCount: 1,
      firstMismatch: {
        outputIndex: 0,
        expectedTokenId: 7,
        actualTokenId: 8,
      },
    });
  });

  it("returns a nonzero status with a concise error", async () => {
    const capture = captureIo();
    expect(await runCli(["scenario", "does-not-exist"], capture.io)).toBe(1);
    expect(capture.stderr()).toContain("unknown scenario preset");
  });

  it("runs an exact-capacity expert-cache workload", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "expert-cache.yaml");
    await writeFile(path, `
expert_cache:
  hot_capacity_bytes: 128
  warm_capacity_bytes: 128
  warm_to_hot_latency_ns: 5
  cold_to_hot_latency_ns: 20
  cold_to_warm_latency_ns: 12
  routing_seed: 7
  initial_hot_expert_ids: [e0]
  experts:
    - { id: e0, bytes: 64 }
    - { id: e1, bytes: 64 }
    - { id: e2, bytes: 64 }
workload:
  token_count: 3
  top_k: 2
  token_interval_ns: 10
`, "utf8");
    const capture = captureIo();
    expect(await runCli(["expert-cache", path], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      routes: unknown[];
      snapshot: {
        metrics: { routes: number };
        hotResidentBytes: number;
      };
    };
    expect(output.routes).toHaveLength(3);
    expect(output.snapshot.metrics.routes).toBe(3);
    expect(output.snapshot.hotResidentBytes).toBeLessThanOrEqual(128);
  });

  it("runs continuous serving through a selected topology", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "serving.yaml");
    await writeFile(path, `
serving:
  max_batch_size: 2
  max_batch_tokens: 4
  prefill_chunk_tokens: 2
  max_kv_tokens: 20
  requests:
    - { id: a, arrival_ns: 0, prompt_tokens: 4, output_tokens: 3 }
    - { id: b, arrival_ns: 10, prompt_tokens: 2, output_tokens: 2, priority: 1 }
`, "utf8");
    const capture = captureIo();

    expect(await runCli(
      ["serving", "single-gpu-cpu", path],
      capture.io,
    )).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      scenarioId: string;
      serving: {
        metrics: { requests: number; outputTokens: number };
        replay: { completedRequests: number };
      };
      batches: unknown[];
    };
    expect(output.scenarioId).toBe("single-gpu-cpu");
    expect(output.serving.metrics.requests).toBe(2);
    expect(output.serving.metrics.outputTokens).toBe(5);
    expect(output.serving.replay.completedRequests).toBe(2);
    expect(output.batches.length).toBeGreaterThan(1);
  });

  it("runs speculative continuous serving with request-scoped acceptance", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "serving-speculative.yaml");
    await writeFile(path, `
serving:
  max_batch_size: 2
  max_batch_tokens: 8
  prefill_chunk_tokens: 4
  max_kv_tokens: 24
  speculative:
    family: mtp
    max_additional_tokens: 2
    acceptance:
      kind: replay
      accepted_draft_tokens_by_request:
        a: [2]
        b: [0, 1]
  requests:
    - { id: a, arrival_ns: 0, prompt_tokens: 4, output_tokens: 5 }
    - { id: b, arrival_ns: 10, prompt_tokens: 4, output_tokens: 5 }
`, "utf8");
    const capture = captureIo();

    expect(await runCli(["serving", "multi-gpu", path], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      serving: {
        metrics: {
          outputTokens: number;
          proposedDraftTokens: number;
          acceptedDraftTokens: number;
          targetForwards: number;
        };
        replay: { completedRequests: number; finalKvTokens: number };
      };
      batches: Array<{
        work: {
          decode: Array<{ mode: string; outcome: string }>;
        };
      }>;
    };
    expect(output.serving.metrics).toMatchObject({
      outputTokens: 10,
      proposedDraftTokens: 8,
      acceptedDraftTokens: 6,
      targetForwards: 3,
    });
    expect(output.serving.replay).toMatchObject({
      completedRequests: 2,
      finalKvTokens: 0,
    });
    expect(output.batches.some((batch) => (
      batch.work.decode.some((decode) => decode.mode === "speculative")
    ))).toBe(true);
  });

  it("compares one serving workload across all topology presets", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "serving-compare.yaml");
    await writeFile(path, `
serving:
  max_batch_size: 2
  max_batch_tokens: 8
  prefill_chunk_tokens: 4
  max_kv_tokens: 24
  speculative:
    family: mtp
    max_additional_tokens: 2
    acceptance:
      kind: conditional_empirical
      match_probability_by_position: [0.8, 0.6]
      seed: 42
  requests:
    - { id: a, arrival_ns: 0, prompt_tokens: 4, output_tokens: 5 }
    - { id: b, arrival_ns: 10, prompt_tokens: 4, output_tokens: 5 }
`, "utf8");
    const capture = captureIo();

    expect(await runCli(["serving-compare", path], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      comparison: Array<{
        rank: number;
        scenarioId: string;
        relativeToFastest: number;
        replayAppliedEvents: number;
      }>;
    };
    expect(output.comparison).toHaveLength(6);
    expect(output.comparison.map((entry) => entry.rank))
      .toEqual([1, 2, 3, 4, 5, 6]);
    expect(output.comparison[0]).toMatchObject({
      scenarioId: "multi-gpu",
      relativeToFastest: 1,
    });
    expect(output.comparison.at(-1)?.scenarioId).toBe("cpu-only");
    expect(output.comparison.every((entry) => (
      entry.replayAppliedEvents > 0
    ))).toBe(true);
  });

  it("fits and imports a scoped calibration dataset", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const calibrationPath = join(directory, "calibration.json");
    const servingPath = join(directory, "serving.yaml");
    await writeFile(
      calibrationPath,
      JSON.stringify(calibrationConfig(["multi-gpu"])),
      "utf8",
    );
    await writeFile(servingPath, `
serving:
  max_batch_size: 1
  max_batch_tokens: 4
  prefill_chunk_tokens: 2
  max_kv_tokens: 12
  requests:
    - { id: a, arrival_ns: 0, prompt_tokens: 2, output_tokens: 2 }
`, "utf8");

    const fitCapture = captureIo();
    expect(await runCli(
      ["calibrate", calibrationPath],
      fitCapture.io,
    )).toBe(0);
    const fit = JSON.parse(fitCapture.stdout()) as {
      confidence: string;
      datasetFingerprint: string;
      diagnostics: unknown[];
    };
    expect(fit.confidence).toBe("heuristic");
    expect(fit.datasetFingerprint).toMatch(/^fnv1a32:/);
    expect(fit.diagnostics).toHaveLength(15);

    const runCapture = captureIo();
    expect(await runCli(
      ["serving", "multi-gpu", servingPath, calibrationPath],
      runCapture.io,
    )).toBe(0);
    const run = JSON.parse(runCapture.stdout()) as {
      confidence: string;
      assumptions: string[];
    };
    expect(run.confidence).toBe("heuristic");
    expect(run.assumptions[0]).toContain("cli-calibration-fixture");

    const rejectedCapture = captureIo();
    expect(await runCli(
      ["serving", "cpu-only", servingPath, calibrationPath],
      rejectedCapture.io,
    )).toBe(1);
    expect(rejectedCapture.stderr()).toContain(
      "cost model is not applicable to scenario cpu-only",
    );
  });

  it("runs a workload through topology resources", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "target-only.yaml");
    await writeFile(path, `
target_only:
  token_count: 8
`, "utf8");
    const capture = captureIo();
    expect(await runCli(["run", "multi-gpu", path], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      status: string;
      operationCounts: { collective: number };
      metrics: { committedTokens: number; totalDurationNs: number };
    };
    expect(output.status).toBe("succeeded");
    expect(output.operationCounts.collective).toBe(8);
    expect(output.metrics.committedTokens).toBe(8);
    expect(output.metrics.totalDurationNs).toBeGreaterThan(0);
  });

  it("compares one workload across all topology presets", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "target-only.yaml");
    await writeFile(path, `
target_only:
  token_count: 8
`, "utf8");
    const capture = captureIo();
    expect(await runCli(["compare", path], capture.io)).toBe(0);
    const output = JSON.parse(capture.stdout()) as {
      comparison: Array<{ scenarioId: string; rank: number }>;
    };
    expect(output.comparison).toHaveLength(6);
    expect(output.comparison[0].scenarioId).toBe("multi-gpu");
    expect(
      output.comparison.find((entry) => entry.scenarioId === "cpu-only")?.rank,
    ).toBe(6);
  });

  it("runs and independently replays a deterministic fault campaign", async () => {
    const directory = await mkdtemp(join(tmpdir(), "inference-sim-"));
    const path = join(directory, "fault-workload.yaml");
    await writeFile(path, `
target_only:
  token_count: 2
`, "utf8");
    const firstCapture = captureIo();
    const secondCapture = captureIo();

    expect(
      await runCli(
        ["fault-campaign", "multi-gpu", path],
        firstCapture.io,
      ),
    ).toBe(0);
    expect(
      await runCli(
        ["fault-campaign", "multi-gpu", path],
        secondCapture.io,
      ),
    ).toBe(0);
    expect(firstCapture.stdout()).toBe(secondCapture.stdout());
    const output = JSON.parse(firstCapture.stdout()) as {
      baseline: { status: string; submittedSteps: number };
      cases: Array<{
        id: string;
        status: string;
        replayAppliedEvents: number;
      }>;
    };
    expect(output.baseline.status).toBe("succeeded");
    expect(output.baseline.submittedSteps).toBeGreaterThan(0);
    expect(output.cases.map((entry) => entry.id)).toEqual([
      "device:node0:gpu0",
      "device:node0:gpu1",
      "link:node0:nvlink:forward",
      "epoch:1",
    ]);
    expect(output.cases.every((entry) => (
      entry.status !== "succeeded" && entry.replayAppliedEvents > 0
    ))).toBe(true);
  });
});

function calibrationConfig(scenarioIds: readonly string[]) {
  const costs = {
    cpu: {
      invocation: 400_000,
      attention: 220_000,
      ffn: 300_000,
      draft: 110_000,
      lookup: 8_000,
    },
    gpu: {
      invocation: 120_000,
      attention: 28_000,
      ffn: 38_000,
      draft: 17_000,
      lookup: 8_000,
    },
    npu: {
      invocation: 100_000,
      attention: 22_000,
      ffn: 52_000,
      draft: 24_000,
      lookup: 8_000,
    },
  } as const;
  const observations = Object.entries(costs).flatMap(([deviceKind, device]) => [
    {
      id: `${deviceKind}-invocation-0`,
      device_kind: deviceKind,
      capability: "invocation",
      work_items: 0,
      durations_ns: [device.invocation - 1, device.invocation, device.invocation + 1],
      regime: "test no-op",
    },
    ...(["attention", "ffn", "draft", "lookup"] as const).flatMap(
      (capability) => [1, 8].map((workItems) => {
        const center = device.invocation + device[capability] * workItems;
        return {
          id: `${deviceKind}-${capability}-${workItems}`,
          device_kind: deviceKind,
          capability,
          work_items: workItems,
          durations_ns: [center - 1, center, center + 1],
          regime: "test batch",
        };
      }),
    ),
  ]);
  return {
    calibration: {
      revision: 1,
      id: "cli-calibration-fixture",
      provenance: {
        kind: "synthetic",
        source: "CLI test fixture",
        software_stack: "test stack",
        model_artifact: "test model",
      },
      applicability: {
        scenario_ids: scenarioIds,
        device_kind_labels: {
          cpu: "test CPU",
          gpu: "test GPU",
          npu: "test NPU",
        },
      },
      model_constants: {
        activation_bytes_per_token: 1_048_576,
        collective_bytes_per_token: 524_288,
        cold_load_byte_multiplier: 2,
      },
      quality: {
        min_samples_per_point: 3,
        max_normalized_rmse: 0.01,
        max_p95_relative_error: 0.01,
      },
      observations,
    },
  };
}
