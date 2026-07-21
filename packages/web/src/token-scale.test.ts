import { describe, expect, it } from "vitest";
import {
  formatApproxTokenCount,
  formatTokenCount,
  nearestTokenStepIndex,
  OUTPUT_TOKEN_STEPS,
  PROMPT_TOKEN_STEPS,
} from "./token-scale.js";

describe("token slider scale", () => {
  it("covers long input and bounded exact output traces", () => {
    expect(PROMPT_TOKEN_STEPS.at(-1)).toBe(1_048_576);
    expect(OUTPUT_TOKEN_STEPS.slice(0, 4)).toEqual([1, 2, 4, 8]);
    expect(OUTPUT_TOKEN_STEPS.at(-1)).toBe(32_768);
  });

  it("maps arbitrary imported values to the nearest logarithmic position", () => {
    expect(PROMPT_TOKEN_STEPS[nearestTokenStepIndex(
      600_000,
      PROMPT_TOKEN_STEPS,
    )]).toBe(524_288);
    expect(formatTokenCount(1_048_576)).toBe("1M");
    expect(formatTokenCount(1_048_576 + 32_768)).toBe("1.03M");
    expect(formatTokenCount(32_768)).toBe("32K");
    expect(formatTokenCount(768)).toBe("768");
    expect(formatApproxTokenCount(99_157)).toBe("96.8K");
    expect(formatApproxTokenCount(1_184_000)).toBe("1.13M");
  });
});
