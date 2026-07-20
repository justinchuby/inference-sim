import {
  CALIBRATION_DATASET_REVISION,
  CONCURRENT_PLAN_TRACE_REVISION,
  EXPERT_CACHE_CONTRACT_REVISION,
  PAGED_KV_CONTRACT_REVISION,
  PLAN_CONTRACT_REVISION,
  SCENARIO_SCHEMA_VERSION,
  SERVING_EXPERT_CACHE_CONTRACT_REVISION,
  SERVING_TRACE_CONTRACT_REVISION,
  SPECULATIVE_FAMILY_CONTRACT_REVISION,
  SPECULATIVE_ITERATION_CONTRACT_REVISION,
  SPECULATIVE_RUNTIME_CAPTURE_REVISION,
  SPECULATIVE_TOKEN_TRACE_REVISION,
  TOPOLOGY_COST_MODEL_REVISION,
  createSimulationResultArtifact,
} from "@inference-sim/core";
import type {
  DashboardArtifact,
  DashboardArtifactOutput,
  DashboardRunConfig,
} from "./types.js";

const DASHBOARD_ARTIFACT_CONTRACTS = {
  calibration_dataset: CALIBRATION_DATASET_REVISION,
  concurrent_plan_trace: CONCURRENT_PLAN_TRACE_REVISION,
  expert_cache: EXPERT_CACHE_CONTRACT_REVISION,
  frozen_plan: PLAN_CONTRACT_REVISION,
  paged_kv: PAGED_KV_CONTRACT_REVISION,
  scenario_schema: SCENARIO_SCHEMA_VERSION,
  serving_expert_cache: SERVING_EXPERT_CACHE_CONTRACT_REVISION,
  serving_trace: SERVING_TRACE_CONTRACT_REVISION,
  speculative_family: SPECULATIVE_FAMILY_CONTRACT_REVISION,
  speculative_iteration: SPECULATIVE_ITERATION_CONTRACT_REVISION,
  speculative_runtime_capture: SPECULATIVE_RUNTIME_CAPTURE_REVISION,
  speculative_token_trace: SPECULATIVE_TOKEN_TRACE_REVISION,
  topology_cost_model: TOPOLOGY_COST_MODEL_REVISION,
} as const;

export function createDashboardArtifact(
  config: DashboardRunConfig,
  output: DashboardArtifactOutput,
): DashboardArtifact {
  const comparisonSuffix =
    config.mode === "serving" && config.serving.compareTopologies
      ? "/comparison"
      : "";
  return createSimulationResultArtifact(
    `dashboard/${config.mode}${comparisonSuffix}`,
    DASHBOARD_ARTIFACT_CONTRACTS,
    config,
    output,
  );
}

export function dashboardArtifactFileName(
  artifact: DashboardArtifact,
): string {
  const fingerprint = artifact.artifactFingerprint.slice("fnv1a32:".length);
  const scenario = artifact.output.summary.scenario.id.replaceAll(
    /[^a-zA-Z0-9_-]/g,
    "-",
  );
  return `inference-sim-${scenario}-${artifact.input.mode}-${fingerprint}.json`;
}
