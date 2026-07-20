import { describe, expect, it } from "vitest";
import { panLogDomain, zoomLogDomain } from "./roofline-viewport.js";

describe("roofline viewport", () => {
  it("zooms around the requested logarithmic anchor", () => {
    const centered = zoomLogDomain([1, 1000], 0.5);
    expect(centered[0]).toBeCloseTo(10 ** 0.75);
    expect(centered[1]).toBeCloseTo(10 ** 2.25);

    const left = zoomLogDomain([1, 1000], 0.5, 0);
    expect(left[0]).toBeCloseTo(1);
    expect(left[1]).toBeCloseTo(10 ** 1.5);
  });

  it("pans without changing logarithmic span", () => {
    const shifted = panLogDomain([10, 1000], 0.25);
    expect(shifted[0]).toBeCloseTo(10 ** 1.5);
    expect(shifted[1]).toBeCloseTo(10 ** 3.5);
  });

  it("rejects invalid domains", () => {
    expect(() => zoomLogDomain([0, 1], 2)).toThrow("log domain");
    expect(() => panLogDomain([10, 1], 1)).toThrow("log domain");
  });
});
