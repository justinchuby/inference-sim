import "fake-indexeddb/auto";
import { beforeEach, describe, expect, it } from "vitest";
import {
  clearArtifactHistory,
  deleteArtifactFromHistory,
  listArtifactHistory,
  readArtifactFromHistory,
  saveArtifactToHistory,
} from "./artifact-history.js";
import { parseDashboardArtifactFileText } from "./dashboard-artifact.js";
import { executeDashboardWorkerRun } from "./dashboard-worker-run.js";
import type {
  DashboardResult,
  DashboardRunConfig,
} from "./types.js";

function config(seed: number): DashboardRunConfig {
  return {
    scenarioName: "multi-gpu",
    multiGpuRanks: 2,
    mode: "speculative",
    seed,
    speculative: {
      family: "mtp",
      outputTokens: 16,
      draftWidth: 2,
      firstPositionAcceptance: 0.8,
    },
    serving: {
      compareTopologies: false,
      useExpertCache: false,
      decodeMode: "target_only",
      draftWidth: 2,
      firstPositionAcceptance: 0.8,
      requestCount: 1,
      arrivalGapUs: 0,
      promptTokens: 16,
      outputTokens: 4,
      maxBatchSize: 1,
      maxBatchTokens: 16,
      prefillChunkTokens: 16,
    },
    expertCache: {
      placementStrategy: "contiguous",
      tokenCount: 16,
      topK: 1,
      expertCount: 8,
      hotSlots: 2,
      warmSlots: 2,
      adaptivePrefetch: false,
    },
  };
}

function run(seed: number) {
  const execution = executeDashboardWorkerRun(config(seed));
  const result: DashboardResult = {
    ...execution.summary,
    durationMs: 0,
  };
  return { execution, result };
}

describe("artifact history", () => {
  beforeEach(async () => {
    await clearArtifactHistory();
  });

  it("persists complete artifacts, deduplicates fingerprints, and revalidates on read", async () => {
    const first = run(1);
    await saveArtifactToHistory(first.execution.artifact, first.result, {
      maxEntries: 20,
      maxTotalBytes: 1024 ** 2,
    }, 100);
    await saveArtifactToHistory(first.execution.artifact, first.result, {
      maxEntries: 20,
      maxTotalBytes: 1024 ** 2,
    }, 200);

    const entries = await listArtifactHistory();
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({
      fingerprint: first.execution.artifact.artifactFingerprint,
      savedAtMs: 100,
      lastOpenedAtMs: 200,
      scenarioId: "multi-gpu",
      mode: "speculative",
    });
    const stored = await readArtifactFromHistory(entries[0].fingerprint, 300);
    const parsed = parseDashboardArtifactFileText(
      stored.text,
      stored.fileName,
    );
    expect(parsed.expectation.artifactFingerprint).toBe(entries[0].fingerprint);
    expect((await listArtifactHistory())[0].lastOpenedAtMs).toBe(300);
  });

  it("evicts least-recently-opened artifacts and supports deletion", async () => {
    const limits = { maxEntries: 2, maxTotalBytes: 1024 ** 2 };
    const first = run(1);
    const second = run(2);
    const third = run(3);
    await saveArtifactToHistory(first.execution.artifact, first.result, limits, 100);
    await saveArtifactToHistory(second.execution.artifact, second.result, limits, 200);
    await readArtifactFromHistory(first.execution.artifact.artifactFingerprint, 300);
    await saveArtifactToHistory(third.execution.artifact, third.result, limits, 400);

    const retained = await listArtifactHistory();
    expect(retained.map((entry) => entry.fingerprint)).toEqual([
      third.execution.artifact.artifactFingerprint,
      first.execution.artifact.artifactFingerprint,
    ]);
    const afterDelete = await deleteArtifactFromHistory(
      first.execution.artifact.artifactFingerprint,
    );
    expect(afterDelete.map((entry) => entry.fingerprint)).toEqual([
      third.execution.artifact.artifactFingerprint,
    ]);
  });
});
