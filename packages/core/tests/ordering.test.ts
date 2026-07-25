import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { compareIds } from "../src/index.js";

const SOURCE_DIR = new URL("../src/", import.meta.url).pathname;

describe("deterministic ordering", () => {
  it("orders identifiers by code unit, independent of locale", () => {
    expect(compareIds("a", "a")).toBe(0);
    expect(compareIds("a", "b")).toBeLessThan(0);
    expect(compareIds("b", "a")).toBeGreaterThan(0);
    // Locale collation folds case and punctuation; code-unit order does not.
    expect(compareIds("B", "a")).toBeLessThan(0);
    expect(compareIds("rank-2", "rank10")).toBeLessThan(0);
    expect(compareIds("_", "A")).toBeGreaterThan(0);
  });

  it("is a total order that sorts stably", () => {
    const ids = ["b", "A", "a", "B", "_", "rank-2", "rank10", "Ä", "ä"];
    const sorted = [...ids].sort(compareIds);

    expect([...ids].reverse().sort(compareIds)).toEqual(sorted);
    for (let index = 1; index < sorted.length; index++) {
      expect(compareIds(sorted[index - 1]!, sorted[index]!)).toBeLessThan(0);
    }
  });

  it("keeps localeCompare out of core simulation sources", () => {
    // localeCompare depends on the host locale and ICU build, so it can order
    // the same identifiers differently on different machines.
    const offenders = readdirSync(SOURCE_DIR)
      .filter((file) => file.endsWith(".ts"))
      .filter((file) => (
        readFileSync(join(SOURCE_DIR, file), "utf8").includes(".localeCompare(")
      ));

    expect(offenders).toEqual([]);
  });
});
