/// <reference lib="webworker" />

import { serializeSimulationResultArtifact } from "@inference-sim/core";
import {
  createDashboardArtifact,
  dashboardArtifactFileName,
} from "./dashboard-artifact.js";
import { simulateDashboardExecution } from "./dashboard-simulation.js";
import type { WorkerRequest, WorkerResponse } from "./types.js";

const worker = self as DedicatedWorkerGlobalScope;

worker.onmessage = (event: MessageEvent<WorkerRequest>) => {
  if (event.data.type !== "run") {
    return;
  }
  const { runId, config } = event.data;
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
      phase: config.mode === "speculative" && config.speculative.trace
        ? "Verifying token trace"
        : config.calibration === undefined
          ? "Running workload"
          : "Fitting calibration and running workload",
    });

    const output = simulateDashboardExecution(config);
    post({
      type: "progress",
      runId,
      progress: 92,
      phase: config.mode === "speculative" && config.speculative.trace
        ? "Checking token and state parity"
        : "Checking replay",
    });
    const artifact = createDashboardArtifact(config, output);
    post({
      type: "result",
      runId,
      summary: output.summary,
      artifact: {
        blob: new Blob(
          [serializeSimulationResultArtifact(artifact, true)],
          { type: "application/json" },
        ),
        fileName: dashboardArtifactFileName(artifact),
        artifactFingerprint: artifact.artifactFingerprint,
      },
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
