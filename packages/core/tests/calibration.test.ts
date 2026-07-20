import { describe, expect, it } from "vitest";
import {
  CALIBRATION_DATASET_REVISION,
  SCENARIO_PRESET_NAMES,
  buildScenarioPreset,
  fitTopologyCostModel,
  parseCalibrationDataset,
  simulateTopologyWorkload,
  targetOnlyTopologyProfile,
  type CalibratedCapability,
  type CalibrationDataset,
  type CalibrationObservation,
  type SimDeviceKind,
  type TransportCalibrationObservation,
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
  it("parses the external snake-case contract in core", () => {
    const expected = calibrationDataset("synthetic");
    const parsed = parseCalibrationDataset(externalDataset(expected));

    expect(parsed).toEqual(expected);
    expect(fitTopologyCostModel(parsed).datasetFingerprint).toMatch(
      /^fnv1a32:/,
    );
  });

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
      transportObservations: [...dataset.transportObservations].reverse(),
    });

    expect(first.datasetFingerprint).toBe(second.datasetFingerprint);
    expect(first.confidence).toBe("calibrated");
    expect(first.costModel.confidence).toBe("calibrated");
    expect(first.diagnostics).toHaveLength(15);
    expect(first.transportDiagnostics).toHaveLength(1);
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
      "scenario performance evidence is weaker",
    );
  });

  it("reports calibrated timing only when every performance input is calibrated", () => {
    const base = buildScenarioPreset("multi-gpu");
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

    const heuristicLinks = simulateTopologyWorkload(
      {
        ...scenario,
        links: base.links,
      },
      targetOnlyTopologyProfile(2),
      fitTopologyCostModel(calibrationDataset("measured")).costModel,
    );
    expect(heuristicLinks.confidence).toBe("heuristic");
    expect(heuristicLinks.assumptions[1]).toContain(
      "scenario performance evidence is weaker",
    );
  });

  it("rejects calibrated AllToAllV without a traffic signature", () => {
    expect(() => simulateTopologyWorkload(
      buildScenarioPreset("multi-gpu"),
      {
        id: "calibrated-routed",
        batchSize: 1,
        expertPlacement: {
          strategy: "contiguous",
          expertIds: ["e0", "e1", "e2", "e3"],
        },
        expertTokenPlacement: "round_robin",
        units: [{
          id: "route-0",
          targetTokenWidth: 1,
          committedTokens: 1,
          draftTokens: 0,
          activeExperts: 2,
          expertRouted: true,
          routedExperts: [
            { expertId: "e0", sourceTier: "hot", loadBytes: 0 },
            { expertId: "e2", sourceTier: "hot", loadBytes: 0 },
          ],
          warmLoadBytes: 0,
          coldLoadBytes: 0,
        }],
      },
      fitTopologyCostModel(calibrationDataset("measured")).costModel,
    )).toThrow(
      "calibrated all_to_all_v requires a traffic-signature calibration contract",
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

  it("uses exact-path transport curves and rejects silent fallback", () => {
    const fit = fitTopologyCostModel(calibrationDataset("measured"));
    const scenario = buildScenarioPreset("multi-gpu");
    const result = simulateTopologyWorkload(
      scenario,
      twoTokenVerificationProfile(),
      fit.costModel,
    );
    const collective = result.plan.steps.find(
      (step) => step.operation.kind === "collective",
    );
    expect(collective?.operation.durationNs).toBe(2_279);

    expect(() => simulateTopologyWorkload(
      scenario,
      twoTokenVerificationProfile(),
      {
        ...fit.costModel,
        transportCurves: fit.costModel.transportCurves?.map((curve) => ({
          ...curve,
          algorithm: "tree",
        })),
      },
    )).toThrow("no calibrated transport curve");
  });

  it("rejects transport extrapolation beyond observed message sizes", () => {
    const fit = fitTopologyCostModel(calibrationDataset("measured"));
    expect(() => simulateTopologyWorkload(
      buildScenarioPreset("multi-gpu"),
      twoTokenVerificationProfile(),
      {
        ...fit.costModel,
        transportCurves: fit.costModel.transportCurves?.map((curve) => ({
          ...curve,
          points: [
            { bytes: 128 * 1024, durationNs: 800 },
            { bytes: 512 * 1024, durationNs: 1_400 },
          ],
        })),
      },
    )).toThrow("outside calibrated range");
  });

  it("rejects non-monotonic transport measurements", () => {
    const dataset = calibrationDataset("measured");
    expect(() => fitTopologyCostModel({
      ...dataset,
      transportObservations: dataset.transportObservations.map(
        (observation) => observation.bytes > 512 * 1024
          ? {
              ...observation,
              durationsNs: [900, 901, 902],
            }
          : observation,
      ),
    })).toThrow("median duration must be non-decreasing");
  });
});

function twoTokenVerificationProfile() {
  return {
    id: "two-token-verification",
    batchSize: 1,
    units: [{
      id: "verify-0",
      targetTokenWidth: 2,
      committedTokens: 2,
      draftTokens: 0,
      activeExperts: 1,
      warmLoadBytes: 0,
      coldLoadBytes: 0,
    }],
  };
}

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
    transportObservations: [
      transportObservation(512 * 1024, 1_400),
      transportObservation(64 * 1024 ** 2, 113_000),
    ],
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

function transportObservation(
  bytes: number,
  center: number,
): TransportCalibrationObservation {
  const noise = Math.max(1, Math.floor(center / 1000));
  return {
    id: `multi-gpu-collective-${bytes}`,
    scenarioId: "multi-gpu",
    operation: "collective",
    linkIds: ["node0:nvlink:forward", "node0:nvlink:reverse"],
    participantCount: 2,
    algorithm: "all_reduce_ring",
    bytes,
    durationsNs: [center - noise, center, center + noise],
    regime: "two-rank all-reduce fixture",
  };
}

function externalDataset(dataset: CalibrationDataset): unknown {
  return {
    calibration: {
      revision: dataset.revision,
      id: dataset.id,
      provenance: {
        kind: dataset.provenance.kind,
        source: dataset.provenance.source,
        measured_at: dataset.provenance.measuredAt,
        software_stack: dataset.provenance.softwareStack,
        model_artifact: dataset.provenance.modelArtifact,
        notes: dataset.provenance.notes,
      },
      applicability: {
        scenario_ids: dataset.applicability.scenarioIds,
        device_kind_labels: dataset.applicability.deviceKindLabels,
      },
      model_constants: {
        activation_bytes_per_token:
          dataset.modelConstants.activationBytesPerToken,
        collective_bytes_per_token:
          dataset.modelConstants.collectiveBytesPerToken,
        cold_load_byte_multiplier:
          dataset.modelConstants.coldLoadByteMultiplier,
      },
      quality: {
        min_samples_per_point: dataset.quality.minSamplesPerPoint,
        max_normalized_rmse: dataset.quality.maxNormalizedRmse,
        max_p95_relative_error: dataset.quality.maxP95RelativeError,
      },
      observations: dataset.observations.map((observation) => ({
        id: observation.id,
        device_kind: observation.deviceKind,
        capability: observation.capability,
        work_items: observation.workItems,
        durations_ns: observation.durationsNs,
        regime: observation.regime,
      })),
      transport_observations: dataset.transportObservations.map(
        (observation) => ({
          id: observation.id,
          scenario_id: observation.scenarioId,
          operation: observation.operation,
          link_ids: observation.linkIds,
          participant_count: observation.participantCount,
          algorithm: observation.algorithm,
          bytes: observation.bytes,
          durations_ns: observation.durationsNs,
          regime: observation.regime,
        }),
      ),
    },
  };
}
