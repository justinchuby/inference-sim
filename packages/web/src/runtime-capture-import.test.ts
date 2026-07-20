import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import {
  MAX_RUNTIME_CAPTURE_FILE_BYTES,
  parseRuntimeCapturePairFileTexts,
} from "./runtime-capture-import.js";

const targetUrl = new URL(
  "../../../examples/runtime-capture-target-only.yaml",
  import.meta.url,
);
const speculativeUrl = new URL(
  "../../../examples/runtime-capture-speculative.yaml",
  import.meta.url,
);

describe("runtime capture pair import", () => {
  it("binds target-only and speculative artifacts in either file order", async () => {
    const target = await readFile(targetUrl, "utf8");
    const speculative = await readFile(speculativeUrl, "utf8");
    const parsed = await parseRuntimeCapturePairFileTexts([
      { fileName: "speculative.yaml", text: speculative },
      { fileName: "target.yaml", text: target },
    ]);

    expect(parsed.targetOnly.capture.id).toBe("target-only-synthetic-001");
    expect(parsed.speculative.capture.id).toBe("speculative-synthetic-001");
    expect(parsed.preview.differential.matchesTargetOnly).toBe(true);
    expect(parsed.preview.iterations.map((iteration) => iteration.outcome))
      .toEqual(["correction", "bonus", "accepted_tail"]);
  });

  it("rejects missing and duplicate roles", async () => {
    const target = await readFile(targetUrl, "utf8");

    await expect(parseRuntimeCapturePairFileTexts([
      { fileName: "target-a.yaml", text: target },
    ])).rejects.toThrow("exactly two files");
    await expect(parseRuntimeCapturePairFileTexts([
      { fileName: "target-a.yaml", text: target },
      { fileName: "target-b.yaml", text: target },
    ])).rejects.toThrow("two target-only captures");
  });

  it("rejects oversized capture content before binding", async () => {
    await expect(parseRuntimeCapturePairFileTexts([
      {
        fileName: "target.yaml",
        text: "x".repeat(MAX_RUNTIME_CAPTURE_FILE_BYTES + 1),
      },
      { fileName: "speculative.yaml", text: "{}" },
    ])).rejects.toThrow("exceeds the 1 MiB limit");
  });
});
