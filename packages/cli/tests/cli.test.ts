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
    expect(output.memoryLedger[0].reservedBytes).toBe(76 * 1024 ** 3);
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
});
