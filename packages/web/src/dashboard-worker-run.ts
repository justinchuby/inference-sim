import { serializeSimulationResultArtifact } from "@inference-sim/core";
import {
  compareDashboardArtifact,
  createDashboardArtifact,
  dashboardArtifactFileName,
} from "./dashboard-artifact.js";
import { simulateDashboardExecution } from "./dashboard-simulation.js";
import type {
  DashboardArtifactExpectation,
  DashboardArtifactReplay,
  DashboardArtifactDownload,
  DashboardResult,
  DashboardRunConfig,
  WorkerRunProgressReporter,
} from "./types.js";

export interface DashboardWorkerRunResult {
  readonly summary: Omit<DashboardResult, "durationMs">;
  readonly artifact: DashboardArtifactDownload;
  readonly artifactReplay?: DashboardArtifactReplay;
}

export function executeDashboardWorkerRun(
  config: DashboardRunConfig,
  expectedArtifact?: DashboardArtifactExpectation,
  reportProgress: WorkerRunProgressReporter = () => {},
): DashboardWorkerRunResult {
  const output = simulateDashboardExecution(config, reportProgress);
  reportProgress({
    progress: 84,
    phase: "Creating deterministic result artifact",
  });
  const artifact = createDashboardArtifact(config, output);
  reportProgress({
    progress: 89,
    phase: "Serializing result artifact",
  });
  const blob = new Blob(
    [serializeSimulationResultArtifact(artifact, true)],
    { type: "application/json" },
  );
  reportProgress({
    progress: 94,
    phase: expectedArtifact === undefined
      ? "Finalizing replay evidence"
      : "Comparing artifact fingerprints",
  });
  return {
    summary: output.summary,
    artifact: {
      blob,
      fileName: dashboardArtifactFileName(artifact),
      artifactFingerprint: artifact.artifactFingerprint,
    },
    ...(expectedArtifact === undefined
      ? {}
      : {
          artifactReplay: compareDashboardArtifact(
            artifact,
            expectedArtifact,
          ),
        }),
  };
}
