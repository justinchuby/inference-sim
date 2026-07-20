import {
  parseSimulationScenario,
  type SimulationScenario,
} from "@inference-sim/core";

export const MAX_SCENARIO_FILE_BYTES = 4 * 1024 * 1024;

export interface ParsedScenarioFile {
  readonly scenario: SimulationScenario;
}

export async function parseScenarioFileText(
  text: string,
  fileName: string,
): Promise<ParsedScenarioFile> {
  if (new TextEncoder().encode(text).byteLength > MAX_SCENARIO_FILE_BYTES) {
    throw new Error("scenario file exceeds the 4 MiB limit");
  }
  const extension = fileName.split(".").at(-1)?.toLowerCase();
  if (extension !== "json" && extension !== "yaml" && extension !== "yml") {
    throw new Error("scenario file must use .json, .yaml, or .yml");
  }
  let input: unknown;
  try {
    if (extension === "json") {
      input = JSON.parse(text) as unknown;
    } else {
      const { parse } = await import("yaml");
      input = parse(text) as unknown;
    }
  } catch (error) {
    throw new Error(
      `invalid scenario ${extension.toUpperCase()}: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  return { scenario: parseSimulationScenario(input) };
}
