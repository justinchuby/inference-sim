import { describe, expect, it } from "vitest";
import {
  SIMULATION_RESULT_ARTIFACT_KIND,
  SIMULATION_RESULT_ARTIFACT_REVISION,
  createSimulationResultArtifact,
  parseSimulationResultArtifact,
  serializeSimulationResultArtifact,
} from "../src/index.js";

describe("simulation result artifacts", () => {
  it("creates deterministic fingerprints independent of object key order", () => {
    const first = createSimulationResultArtifact(
      "dashboard/speculative",
      { topology: 9, scenario: 4 },
      { seed: 42, profile: { width: 4, family: "mtp" } },
      { trace: [{ atNs: 3, kind: "commit" }], tokens: 8 },
    );
    const second = createSimulationResultArtifact(
      "dashboard/speculative",
      { scenario: 4, topology: 9 },
      { profile: { family: "mtp", width: 4 }, seed: 42 },
      { tokens: 8, trace: [{ kind: "commit", atNs: 3 }] },
    );

    expect(first).toMatchObject({
      kind: SIMULATION_RESULT_ARTIFACT_KIND,
      revision: SIMULATION_RESULT_ARTIFACT_REVISION,
      contracts: { scenario: 4, topology: 9 },
    });
    expect(first.inputFingerprint).toBe(second.inputFingerprint);
    expect(first.outputFingerprint).toBe(second.outputFingerprint);
    expect(first.artifactFingerprint).toBe(second.artifactFingerprint);
    expect(serializeSimulationResultArtifact(first)).toBe(
      serializeSimulationResultArtifact(second),
    );
    expect(parseSimulationResultArtifact(JSON.parse(
      serializeSimulationResultArtifact(first, true),
    ))).toEqual(first);
  });

  it("fails closed when input, output, or envelope metadata is changed", () => {
    const artifact = createSimulationResultArtifact(
      "dashboard/serving",
      { scenario: 4 },
      { seed: 7 },
      { completed: true },
    );

    expect(() => parseSimulationResultArtifact({
      ...artifact,
      input: { seed: 8 },
    })).toThrow("input fingerprint mismatch");
    expect(() => parseSimulationResultArtifact({
      ...artifact,
      output: { completed: false },
    })).toThrow("output fingerprint mismatch");
    expect(() => parseSimulationResultArtifact({
      ...artifact,
      runKind: "dashboard/speculative",
    })).toThrow("artifact fingerprint mismatch");
    expect(() => parseSimulationResultArtifact({
      ...artifact,
      extra: true,
    })).toThrow("unknown fields extra");
  });

  it("rejects invalid contracts and non-JSON evidence", () => {
    expect(() => createSimulationResultArtifact(
      "dashboard/serving",
      {},
      {},
      {},
    )).toThrow("contracts must not be empty");
    expect(() => createSimulationResultArtifact(
      "dashboard/serving",
      { scenario: 0 },
      {},
      {},
    )).toThrow("positive safe integer");
    expect(() => createSimulationResultArtifact(
      "dashboard/serving",
      { scenario: 4 },
      {},
      { durationNs: Number.POSITIVE_INFINITY },
    )).toThrow("finite JSON numbers");
    expect(() => createSimulationResultArtifact(
      "dashboard/serving",
      { scenario: 4 },
      {},
      new Map(),
    )).toThrow("plain JSON objects");

    const circular: { self?: unknown } = {};
    circular.self = circular;
    expect(() => createSimulationResultArtifact(
      "dashboard/serving",
      { scenario: 4 },
      {},
      circular,
    )).toThrow("circular reference");
    expect(() => createSimulationResultArtifact(
      "dashboard/serving",
      { scenario: 4 },
      {},
      Array(1),
    )).toThrow("sparse array entries");
  });
});
