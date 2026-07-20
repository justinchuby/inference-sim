/// <reference lib="webworker" />

import { executeDashboardWorkerRun } from "./dashboard-worker-run.js";
import { executeFrozenPlanWorkerRun } from "./frozen-plan-worker-run.js";
import { executeOnnxStaticWorkerRun } from "./onnx-static-worker-run.js";
import { executeOnnxSearchWorkerRun } from "./onnx-search-worker-run.js";
import type {
  WorkerRequest,
  WorkerResponse,
  WorkerRunProgress,
} from "./types.js";

const worker = self as DedicatedWorkerGlobalScope;

worker.onmessage = (event: MessageEvent<WorkerRequest>) => {
  if (event.data.type === "run-onnx-search") {
    const {
      runId,
      artifactText,
      sourceFileName,
      baseConfig,
      searchConfig,
    } = event.data;
    try {
      const startedAt = performance.now();
      const result = executeOnnxSearchWorkerRun(
        artifactText,
        sourceFileName,
        baseConfig,
        searchConfig,
        progressReporter(runId),
      );
      post({
        type: "onnx-search-result",
        runId,
        result,
        durationMs: performance.now() - startedAt,
      });
    } catch (error) {
      post({
        type: "error",
        runId,
        message: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }
  if (event.data.type === "run-onnx-static") {
    const {
      runId,
      artifactText,
      sourceFileName,
      config,
    } = event.data;
    try {
      const startedAt = performance.now();
      const result = executeOnnxStaticWorkerRun(
        artifactText,
        sourceFileName,
        config,
        progressReporter(runId),
      );
      post({
        type: "onnx-static-result",
        runId,
        result,
        durationMs: performance.now() - startedAt,
      });
    } catch (error) {
      post({
        type: "error",
        runId,
        message: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }
  if (event.data.type === "run-frozen-plan") {
    const { runId, artifactText, sourceFileName } = event.data;
    try {
      const startedAt = performance.now();
      const result = executeFrozenPlanWorkerRun(
        artifactText,
        sourceFileName,
        progressReporter(runId),
      );
      post({
        type: "frozen-plan-result",
        runId,
        result,
        durationMs: performance.now() - startedAt,
      });
    } catch (error) {
      post({
        type: "error",
        runId,
        message: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }
  const { runId, config, expectedArtifact } = event.data;
  try {
    const startedAt = performance.now();
    const result = executeDashboardWorkerRun(
      config,
      expectedArtifact,
      progressReporter(runId),
    );
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

function progressReporter(
  runId: number,
): (update: WorkerRunProgress) => void {
  return ({ progress, phase }) => {
    post({
      type: "progress",
      runId,
      progress,
      phase,
    });
  };
}

export {};
