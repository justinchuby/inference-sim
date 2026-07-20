/// <reference lib="webworker" />

import { simulateDashboard } from "./dashboard-simulation.js";
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
      progress: 34,
      phase: "Running workload",
    });

    const base = simulateDashboard(config);
    post({
      type: "progress",
      runId,
      progress: 92,
      phase: "Checking replay",
    });
    post({
      type: "result",
      runId,
      result: {
        ...base,
        durationMs: performance.now() - startedAt,
      },
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
