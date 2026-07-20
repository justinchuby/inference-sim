import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { parse as parseYaml } from "yaml";
import {
  MAX_TOKEN_TRACE_FILE_BYTES,
  parseTokenTraceFileText,
} from "./token-trace-import.js";

const exampleUrl = new URL(
  "../../../examples/speculative-token-trace-mtp.yaml",
  import.meta.url,
);

describe("token trace file import", () => {
  it("imports and previews revisioned YAML through the core oracle", async () => {
    const text = await readFile(exampleUrl, "utf8");
    const result = await parseTokenTraceFileText(text, "trace.yaml");

    expect(result.trace.id).toBe("mtp-correction-bonus-tail");
    expect(result.preview.differential.matchesTargetOnly).toBe(true);
    expect(result.preview.iterations.map((iteration) => iteration.outcome))
      .toEqual(["correction", "bonus", "accepted_tail"]);
  });

  it("imports equivalent JSON", async () => {
    const text = await readFile(exampleUrl, "utf8");
    const result = await parseTokenTraceFileText(
      JSON.stringify(parseYaml(text)),
      "trace.json",
    );

    expect(result.preview.committedOutputTokenIds)
      .toEqual([10, 20, 21, 30, 31, 32, 40, 41]);
  });

  it("rejects unsupported and oversized files before simulation", async () => {
    await expect(parseTokenTraceFileText("{}", "trace.txt"))
      .rejects.toThrow("must use .yaml, .yml, or .json");
    await expect(parseTokenTraceFileText(
      "x".repeat(MAX_TOKEN_TRACE_FILE_BYTES + 1),
      "trace.yaml",
    )).rejects.toThrow("exceeds the 1 MiB limit");
  });
});
