import { describe, expect, it } from "vitest";
import {
  CALIBRATION_DATASET_REVISION,
  SCENARIO_PRESET_NAMES,
  buildScenarioPreset,
  fitTopologyCostModel,
  simulateTopologyWorkload,
  targetOnlyTopologyProfile,
  type CalibratedCapability,
  type CalibrationDataset,
  type CalibrationObservation,
  type SimDeviceKind,
} from "../src/index.js";

const DEVICES = ["cpu", "gpu", "npu"] as const;
const CAPABILITIES = ["attention", "ffn", "draft", "lookup"] as const;
const EXPECTED = {
  cpu: {
    invocation: 400_000,
    attention: 220_000,
    ffn: 300_000,
    draft: 110_000,
    lookup: 8_000,
  },
  gpu: {
    invocation: 120_000,
    attention: 28_000,
    ffn: 38_000,
    draft: 17_000,
    lookup: 8_000,
  },
  npu: {
    invocation: 100_000,
    attention: 22_000,
    ffn: 52_000,
    draft: 24_000,
    lookup: 8_000,
  },
} as const;

describe("topology cost calibration", () => {
  it("fits complete repeated observations with stable provenance", () => {
    const dataset = calibrationDataset("measured");
    const first = fitTopologyCostModel(dataset);
    const second = fitTopologyCostModel({
      ...dataset,
      provenance: {
        modelArtifact: dataset.provenance.modelArtifact,
        softwareStack: dataset.provenance.softwareStack,
        source: dataset.provenance.source,
        kind: dataset.provenance.kind,
        measuredAt: dataset.provenance.measuredAt,
      },
      observations: [...dataset.observations].reverse(),
    });

    expect(first.datasetFingerprint).toBe(second.datasetFingerprint);
    expect(first.confidence).toBe("calibrated");
    expect(first.costModel.confidence).toBe("calibrated");
    expect(first.diagnostics).toHaveLength(15);
    expect(first.costModel.deviceCosts.gpu).toEqual({
      invocationOverheadNs: 120_000,
      attentionNsPerToken: 28_000,
      ffnNsPerToken: 38_000,
      draftNsPerToken: 17_000,
      lookupNsPerToken: 8_000,
    });
    expect(first.diagnostics.every((diagnostic) => (
      diagnostic.samples >= 3
      && diagnostic.normalizedRmse < 0.01
      && diagnostic.p95RelativeError < 0.01
    ))).toBe(true);

    const result = simulateTopologyWorkload(
      buildScenarioPreset("multi-gpu"),
      targetOnlyTopologyProfile(2),
      first.costModel,
    );
    expect(result.confidence).toBe("heuristic");
    expect(result.assumptions[0]).toContain(first.datasetFingerprint);
    expect(result.assumptions[1]).toContain(
      "scenario device, memory, or link evidence is weaker",
    );
  });

  it("reports calibrated timing only when every performance input is calibrated", () => {
    const base = buildScenarioPreset("single-gpu-cpu");
    const calibratedProvenance = {
      confidence: "calibrated" as const,
      source: "measured topology fixture",
      measuredAt: "2026-07-19",
    };
    const scenario = {
      ...base,
      devices: base.devices.map((device) => ({
        ...device,
        provenance: calibratedProvenance,
      })),
      memoryDomains: base.memoryDomains.map((domain) => ({
        ...domain,
        provenance: calibratedProvenance,
      })),
      links: base.links.map((link) => ({
        ...link,
        provenance: calibratedProvenance,
      })),
    };
    const result = simulateTopologyWorkload(
      scenario,
      targetOnlyTopologyProfile(2),
      fitTopologyCostModel(calibrationDataset("measured")).costModel,
    );
    expect(result.confidence).toBe("calibrated");
    expect(result.assumptions[1]).toBe(
      "overall timing confidence is calibrated",
    );
  });

  it("keeps synthetic imports heuristic", () => {
    const result = fitTopologyCostModel(calibrationDataset("synthetic"));
    expect(result.confidence).toBe("heuristic");
    expect(result.costModel.confidence).toBe("heuristic");
  });

  it("rejects incomplete capability coverage", () => {
    const dataset = calibrationDataset("measured");
    expect(() => fitTopologyCostModel({
      ...dataset,
      observations: dataset.observations.filter((observation) => !(
        observation.deviceKind === "npu"
        && observation.capability === "lookup"
      )),
    })).toThrow(
      "npu lookup requires at least 2 distinct work-item points",
    );
  });

  it("rejects measured evidence without a measurement date", () => {
    const dataset = calibrationDataset("measured");
    expect(() => fitTopologyCostModel({
      ...dataset,
      provenance: {
        ...dataset.provenance,
        measuredAt: undefined,
      },
    })).toThrow("provenance measuredAt must be non-empty");
  });

  it("rejects a non-positive fitted capability cost", () => {
    const dataset = calibrationDataset("measured");
    expect(() => fitTopologyCostModel({
      ...dataset,
      observations: dataset.observations.map((observation) => (
        observation.deviceKind === "gpu"
        && observation.capability === "attention"
          ? {
              ...observation,
              durationsNs: observation.workItems === 1
                ? [100_000, 100_000, 100_000]
                : [20_000, 20_000, 20_000],
            }
          : observation
      )),
    })).toThrow(
      "gpu attention coefficient must fit to a positive safe integer",
    );
  });

  it("rejects a fit that exceeds its declared error budget", () => {
    const dataset = calibrationDataset("measured");
    expect(() => fitTopologyCostModel({
      ...dataset,
      quality: {
        ...dataset.quality,
        maxNormalizedRmse: 0.001,
      },
      observations: dataset.observations.map((observation) => (
        observation.id === "cpu-attention-8"
          ? {
              ...observation,
              durationsNs: [
                observation.durationsNs[0],
                observation.durationsNs[1],
                observation.durationsNs[2] + 400_000,
              ],
            }
          : observation
      )),
    })).toThrow("normalized RMSE");
  });

  it("fails closed when a calibrated model is used outside its scope", () => {
    const dataset = calibrationDataset("measured");
    const fit = fitTopologyCostModel({
      ...dataset,
      applicability: {
        ...dataset.applicability,
        scenarioIds: ["multi-gpu"],
      },
    });
    expect(() => simulateTopologyWorkload(
      buildScenarioPreset("cpu-only"),
      targetOnlyTopologyProfile(1),
      fit.costModel,
    )).toThrow("cost model is not applicable to scenario cpu-only");
  });

  it("fails closed outside a calibrated work-item range", () => {
    const fit = fitTopologyCostModel(calibrationDataset("measured"));
    expect(() => simulateTopologyWorkload(
      buildScenarioPreset("single-gpu-cpu"),
      {
        ...targetOnlyTopologyProfile(1),
        batchSize: 9,
      },
      fit.costModel,
    )).toThrow(
      "gpu attention work items 9 are outside calibrated range 1..8",
    );
  });
});

function calibrationDataset(
  kind: "measured" | "synthetic",
): CalibrationDataset {
  return {
    revision: CALIBRATION_DATASET_REVISION,
    id: "known-linear-costs",
    provenance: {
      kind,
      source: kind === "measured" ? "test backend trace" : "synthetic fixture",
      ...(kind === "measured" ? { measuredAt: "2026-07-19" } : {}),
      softwareStack: "onnxruntime test stack",
      modelArtifact: "fixture-model@sha256:test",
    },
    applicability: {
      scenarioIds: SCENARIO_PRESET_NAMES,
      deviceKindLabels: {
        cpu: "fixture CPU class",
        gpu: "fixture GPU class",
        npu: "fixture NPU class",
      },
    },
    modelConstants: {
      activationBytesPerToken: 1024 ** 2,
      collectiveBytesPerToken: 512 * 1024,
      coldLoadByteMultiplier: 2,
    },
    quality: {
      minSamplesPerPoint: 3,
      maxNormalizedRmse: 0.05,
      maxP95RelativeError: 0.05,
    },
    observations: DEVICES.flatMap((deviceKind) => (
      observationsForDevice(deviceKind)
    )),
  };
}

function observationsForDevice(
  deviceKind: SimDeviceKind,
): CalibrationObservation[] {
  const expected = EXPECTED[deviceKind];
  return [
    observation(deviceKind, "invocation", 0, expected.invocation, 0),
    ...CAPABILITIES.flatMap((capability) => [
      observation(
        deviceKind,
        capability,
        1,
        expected.invocation,
        expected[capability],
      ),
      observation(
        deviceKind,
        capability,
        8,
        expected.invocation,
        expected[capability],
      ),
    ]),
  ];
}

function observation(
  deviceKind: SimDeviceKind,
  capability: CalibratedCapability,
  workItems: number,
  overhead: number,
  slope: number,
): CalibrationObservation {
  const center = overhead + slope * workItems;
  const noise = Math.max(1, Math.floor(center / 1000));
  return {
    id: `${deviceKind}-${capability}-${workItems}`,
    deviceKind,
    capability,
    workItems,
    durationsNs: [center - noise, center, center + noise],
    regime: "batch=1 dtype=fp16 fixture",
  };
}
