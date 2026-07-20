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
    accepted_draft_tokens: [2, 2]
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
