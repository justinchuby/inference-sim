import {
  parseFrozenPlanArtifact,
  type FrozenPlanArtifact,
} from "@inference-sim/core";

export const MAX_FROZEN_PLAN_FILE_BYTES = 128 * 1024 * 1024;

export function parseFrozenPlanArtifactFileText(
  text: string,
  fileName: string,
): FrozenPlanArtifact {
  if (new TextEncoder().encode(text).byteLength > MAX_FROZEN_PLAN_FILE_BYTES) {
    throw new Error("FrozenPlan artifact exceeds the 128 MiB limit");
  }
  if (!fileName.toLowerCase().endsWith(".json")) {
    throw new Error("FrozenPlan artifact file must use .json");
  }
  let input: unknown;
  try {
    input = JSON.parse(text) as unknown;
  } catch (error) {
    throw new Error(
      `invalid FrozenPlan artifact JSON: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  return parseFrozenPlanArtifact(input);
}
