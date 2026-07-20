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
} from "./types.js";

export interface DashboardWorkerRunResult {
  readonly summary: Omit<DashboardResult, "durationMs">;
  readonly artifact: DashboardArtifactDownload;
  readonly artifactReplay?: DashboardArtifactReplay;
}

export function executeDashboardWorkerRun(
  config: DashboardRunConfig,
  expectedArtifact?: DashboardArtifactExpectation,
): DashboardWorkerRunResult {
  const output = simulateDashboardExecution(config);
  const artifact = createDashboardArtifact(config, output);
  return {
    summary: output.summary,
    artifact: {
      blob: new Blob(
        [serializeSimulationResultArtifact(artifact, true)],
        { type: "application/json" },
      ),
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
