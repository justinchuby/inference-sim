/// <reference lib="webworker" />

import { executeDashboardWorkerRun } from "./dashboard-worker-run.js";
import type { WorkerRequest, WorkerResponse } from "./types.js";

const worker = self as DedicatedWorkerGlobalScope;

worker.onmessage = (event: MessageEvent<WorkerRequest>) => {
  if (event.data.type !== "run") {
    return;
  }
  const { runId, config, expectedArtifact } = event.data;
  try {
    post({
      type: "progress",
      runId,
      progress: 12,
      phase: "Validating scenario",
    });
    const startedAt = performance.now();
    post({
      type: "progress",
      runId,
      progress: config.calibration === undefined ? 34 : 22,
      phase: expectedArtifact
        ? "Re-executing imported artifact"
        : config.mode === "speculative" && config.speculative.trace
          ? "Verifying token trace"
          : config.calibration === undefined
            ? "Running workload"
            : "Fitting calibration and running workload",
    });

    const result = executeDashboardWorkerRun(config, expectedArtifact);
    post({
      type: "progress",
      runId,
      progress: 92,
      phase: expectedArtifact
        ? "Comparing artifact fingerprints"
        : config.mode === "speculative" && config.speculative.trace
          ? "Checking token and state parity"
          : "Checking replay",
    });
    post({
      type: "result",
      runId,
      ...result,
      durationMs: performance.now() - startedAt,
    });
  } catch (error) {
    post({
      type: "error",
      runId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};

function post(message: WorkerResponse): void {
  worker.postMessage(message);
}

export {};
