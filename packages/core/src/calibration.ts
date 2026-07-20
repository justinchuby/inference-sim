import type {
  ConfidenceClass,
  SimDeviceKind,
} from "./scenario-types.js";
import {
  EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM,
  TOPOLOGY_COST_MODEL_REVISION,
  assertCanonicalAllToAllTrafficSignature,
  type DeviceCapabilityCost,
  type TopologyCostModel,
  type TransportCalibrationCurve,
  type TransportOperationKind,
} from "./topology-workload.js";

export const CALIBRATION_DATASET_REVISION = 3;

export type CalibrationEvidenceKind = "measured" | "synthetic";
export type CalibratedCapability =
  | "invocation"
  | "attention"
  | "ffn"
  | "draft"
  | "lookup";

export interface CalibrationDatasetProvenance {
  readonly kind: CalibrationEvidenceKind;
  readonly source: string;
  readonly measuredAt?: string;
  readonly softwareStack: string;
  readonly modelArtifact: string;
  readonly notes?: string;
}

export interface CalibrationApplicability {
  readonly scenarioIds: readonly string[];
  readonly deviceKindLabels: Readonly<Record<SimDeviceKind, string>>;
}

export interface CalibrationModelConstants {
  readonly activationBytesPerToken: number;
  readonly collectiveBytesPerToken: number;
  readonly coldLoadByteMultiplier: number;
}

export interface CalibrationQualityPolicy {
  readonly minSamplesPerPoint: number;
  readonly maxNormalizedRmse: number;
  readonly maxP95RelativeError: number;
}

export interface CalibrationObservation {
  readonly id: string;
  readonly deviceKind: SimDeviceKind;
  readonly capability: CalibratedCapability;
  readonly workItems: number;
  readonly durationsNs: readonly number[];
  readonly regime: string;
}

export interface TransportCalibrationObservation {
  readonly id: string;
  readonly scenarioId: string;
  readonly operation: TransportOperationKind;
  readonly linkIds: readonly string[];
  readonly participantCount: number;
  readonly algorithm: string;
  readonly trafficSignature?: string;
  readonly bytes: number;
  readonly durationsNs: readonly number[];
  readonly regime: string;
}

export interface CalibrationDataset {
  readonly revision: typeof CALIBRATION_DATASET_REVISION;
  readonly id: string;
  readonly provenance: CalibrationDatasetProvenance;
  readonly applicability: CalibrationApplicability;
  readonly modelConstants: CalibrationModelConstants;
  readonly quality: CalibrationQualityPolicy;
  readonly observations: readonly CalibrationObservation[];
  readonly transportObservations: readonly TransportCalibrationObservation[];
}

export interface CalibrationFitDiagnostic {
  readonly deviceKind: SimDeviceKind;
  readonly capability: CalibratedCapability;
  readonly observationPoints: number;
  readonly samples: number;
  readonly minWorkItems: number;
  readonly maxWorkItems: number;
  readonly coefficientNs: number;
  readonly normalizedRmse: number;
  readonly p95RelativeError: number;
}

export interface CalibrationFitResult {
  readonly datasetId: string;
  readonly datasetFingerprint: string;
  readonly confidence: ConfidenceClass;
  readonly costModel: TopologyCostModel;
  readonly diagnostics: readonly CalibrationFitDiagnostic[];
  readonly transportDiagnostics: readonly TransportCalibrationFitDiagnostic[];
}

export interface TransportCalibrationFitDiagnostic {
  readonly scenarioId: string;
  readonly operation: TransportOperationKind;
  readonly linkIds: readonly string[];
  readonly participantCount: number;
  readonly algorithm: string;
  readonly trafficSignature?: string;
  readonly observationPoints: number;
  readonly samples: number;
  readonly minBytes: number;
  readonly maxBytes: number;
  readonly normalizedRmse: number;
  readonly p95RelativeError: number;
}

export class CalibrationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CalibrationError";
  }
}

const DEVICE_KINDS = ["cpu", "gpu", "npu"] as const;
const COMPUTE_CAPABILITIES = [
  "attention",
  "ffn",
  "draft",
  "lookup",
] as const;
const ALL_CAPABILITIES = ["invocation", ...COMPUTE_CAPABILITIES] as const;

export function parseCalibrationDataset(input: unknown): CalibrationDataset {
  const root = requireRecord(input, "config");
  const dataset = requireRecord(root.calibration ?? root, "calibration");
  const provenance = requireRecord(
    dataset.provenance,
    "calibration.provenance",
  );
  const applicability = requireRecord(
    dataset.applicability,
    "calibration.applicability",
  );
  const deviceKindLabels = requireRecord(
    applicability.device_kind_labels,
    "calibration.applicability.device_kind_labels",
  );
  const modelConstants = requireRecord(
    dataset.model_constants,
    "calibration.model_constants",
  );
  const quality = requireRecord(dataset.quality, "calibration.quality");
  const kind = requireEvidenceKind(
    requireString(provenance, "kind", "calibration.provenance"),
  );
  const measuredAt = optionalString(
    provenance,
    "measured_at",
    "calibration.provenance",
  );
  const notes = optionalString(
    provenance,
    "notes",
    "calibration.provenance",
  );
  return {
    revision: requireNumber(
      dataset,
      "revision",
      "calibration",
    ) as typeof CALIBRATION_DATASET_REVISION,
    id: requireString(dataset, "id", "calibration"),
    provenance: {
      kind,
      source: requireString(
        provenance,
        "source",
        "calibration.provenance",
      ),
      ...(measuredAt === undefined ? {} : { measuredAt }),
      softwareStack: requireString(
        provenance,
        "software_stack",
        "calibration.provenance",
      ),
      modelArtifact: requireString(
        provenance,
        "model_artifact",
        "calibration.provenance",
      ),
      ...(notes === undefined ? {} : { notes }),
    },
    applicability: {
      scenarioIds: requireStringArray(
        applicability,
        "scenario_ids",
        "calibration.applicability",
      ),
      deviceKindLabels: {
        cpu: requireString(
          deviceKindLabels,
          "cpu",
          "calibration.applicability.device_kind_labels",
        ),
        gpu: requireString(
          deviceKindLabels,
          "gpu",
          "calibration.applicability.device_kind_labels",
        ),
        npu: requireString(
          deviceKindLabels,
          "npu",
          "calibration.applicability.device_kind_labels",
        ),
      },
    },
    modelConstants: {
      activationBytesPerToken: requireNumber(
        modelConstants,
        "activation_bytes_per_token",
        "calibration.model_constants",
      ),
      collectiveBytesPerToken: requireNumber(
        modelConstants,
        "collective_bytes_per_token",
        "calibration.model_constants",
      ),
      coldLoadByteMultiplier: requireNumber(
        modelConstants,
        "cold_load_byte_multiplier",
        "calibration.model_constants",
      ),
    },
    quality: {
      minSamplesPerPoint: requireNumber(
        quality,
        "min_samples_per_point",
        "calibration.quality",
      ),
      maxNormalizedRmse: requireNumber(
        quality,
        "max_normalized_rmse",
        "calibration.quality",
      ),
      maxP95RelativeError: requireNumber(
        quality,
        "max_p95_relative_error",
        "calibration.quality",
      ),
    },
    observations: requireRecordArray(
      dataset,
      "observations",
      "calibration",
    ).map((observation, index) => {
      const context = `calibration.observations[${index}]`;
      return {
        id: requireString(observation, "id", context),
        deviceKind: requireDeviceKind(
          requireString(observation, "device_kind", context),
          context,
        ),
        capability: requireCapability(
          requireString(observation, "capability", context),
          context,
        ),
        workItems: requireNumber(observation, "work_items", context),
        durationsNs: requireNumberArray(
          observation,
          "durations_ns",
          context,
        ),
        regime: requireString(observation, "regime", context),
      };
    }),
    transportObservations: requireRecordArray(
      dataset,
      "transport_observations",
      "calibration",
    ).map((observation, index) => {
      const context = `calibration.transport_observations[${index}]`;
      return {
        id: requireString(observation, "id", context),
        scenarioId: requireString(observation, "scenario_id", context),
        operation: requireTransportOperation(
          requireString(observation, "operation", context),
          context,
        ),
        linkIds: requireStringArray(observation, "link_ids", context),
        participantCount: requireNumber(
          observation,
          "participant_count",
          context,
        ),
        algorithm: requireString(observation, "algorithm", context),
        ...optionalTransportTrafficSignature(observation, context),
        bytes: requireNumber(observation, "bytes", context),
        durationsNs: requireNumberArray(
          observation,
          "durations_ns",
          context,
        ),
        regime: requireString(observation, "regime", context),
      };
    }),
  };
}

export function fitTopologyCostModel(
  dataset: CalibrationDataset,
): CalibrationFitResult {
  validateCalibrationDataset(dataset);
  const diagnostics: CalibrationFitDiagnostic[] = [];
  const deviceCosts = {} as Record<SimDeviceKind, DeviceCapabilityCost>;
  const validWorkItemRanges = {} as Record<
    SimDeviceKind,
    Record<
      Exclude<CalibratedCapability, "invocation">,
      { minWorkItems: number; maxWorkItems: number }
    >
  >;

  for (const deviceKind of DEVICE_KINDS) {
    const invocationObservations = observationsFor(
      dataset,
      deviceKind,
      "invocation",
    );
    const invocationSamples = invocationObservations.flatMap(
      (observation) => observation.durationsNs,
    );
    const invocationOverheadNs = roundedPositiveCoefficient(
      median(invocationSamples),
      `${deviceKind} invocation overhead`,
    );
    diagnostics.push(buildDiagnostic(
      dataset,
      deviceKind,
      "invocation",
      invocationObservations,
      invocationOverheadNs,
      invocationOverheadNs,
    ));

    const slopes = Object.fromEntries(COMPUTE_CAPABILITIES.map((capability) => {
      const observations = observationsFor(dataset, deviceKind, capability);
      const numerator = observations.reduce((sum, observation) => (
        sum + observation.durationsNs.reduce(
          (inner, durationNs) => (
            inner
            + observation.workItems * (durationNs - invocationOverheadNs)
          ),
          0,
        )
      ), 0);
      const denominator = observations.reduce((sum, observation) => (
        sum + observation.durationsNs.length * observation.workItems ** 2
      ), 0);
      const coefficient = roundedPositiveCoefficient(
        numerator / denominator,
        `${deviceKind} ${capability} coefficient`,
      );
      diagnostics.push(buildDiagnostic(
        dataset,
        deviceKind,
        capability,
        observations,
        invocationOverheadNs,
        coefficient,
      ));
      return [capability, coefficient];
    })) as Record<(typeof COMPUTE_CAPABILITIES)[number], number>;

    deviceCosts[deviceKind] = {
      invocationOverheadNs,
      attentionNsPerToken: slopes.attention,
      ffnNsPerToken: slopes.ffn,
      draftNsPerToken: slopes.draft,
      lookupNsPerToken: slopes.lookup,
    };
    validWorkItemRanges[deviceKind] = Object.fromEntries(
      COMPUTE_CAPABILITIES.map((capability) => {
        const workItems = observationsFor(
          dataset,
          deviceKind,
          capability,
        ).map((observation) => observation.workItems);
        return [
          capability,
          {
            minWorkItems: Math.min(...workItems),
            maxWorkItems: Math.max(...workItems),
          },
        ];
      }),
    ) as NonNullable<TopologyCostModel["validWorkItemRanges"]>[SimDeviceKind];
  }

  const confidence: ConfidenceClass = dataset.provenance.kind === "measured"
    ? "calibrated"
    : "heuristic";
  const {
    curves: transportCurves,
    diagnostics: transportDiagnostics,
  } = fitTransportCurves(dataset);
  const datasetFingerprint = fingerprintDataset(dataset);
  return {
    datasetId: dataset.id,
    datasetFingerprint,
    confidence,
    costModel: {
      revision: TOPOLOGY_COST_MODEL_REVISION,
      confidence,
      source: [
        `calibration dataset ${dataset.id}`,
        datasetFingerprint,
        dataset.provenance.source,
      ].join("; "),
      deviceCosts,
      activationBytesPerToken:
        dataset.modelConstants.activationBytesPerToken,
      collectiveBytesPerToken:
        dataset.modelConstants.collectiveBytesPerToken,
      coldLoadByteMultiplier:
        dataset.modelConstants.coldLoadByteMultiplier,
      applicability: {
        scenarioIds: [...dataset.applicability.scenarioIds].sort(),
        deviceKindLabels: { ...dataset.applicability.deviceKindLabels },
      },
      validWorkItemRanges,
      transportCurves,
    },
    diagnostics,
    transportDiagnostics,
  };
}

export function validateCalibrationDataset(
  dataset: CalibrationDataset,
): void {
  if (dataset.revision !== CALIBRATION_DATASET_REVISION) {
    throw new CalibrationError(
      `unsupported calibration dataset revision ${dataset.revision}`,
    );
  }
  assertNonEmpty(dataset.id, "dataset id");
  if (
    dataset.provenance.kind !== "measured"
    && dataset.provenance.kind !== "synthetic"
  ) {
    throw new CalibrationError(
      `unsupported calibration evidence kind ${String(dataset.provenance.kind)}`,
    );
  }
  assertNonEmpty(dataset.provenance.source, "provenance source");
  assertNonEmpty(dataset.provenance.softwareStack, "provenance software stack");
  assertNonEmpty(dataset.provenance.modelArtifact, "provenance model artifact");
  if (dataset.provenance.kind === "measured") {
    assertNonEmpty(dataset.provenance.measuredAt ?? "", "provenance measuredAt");
  }

  if (dataset.applicability.scenarioIds.length === 0) {
    throw new CalibrationError(
      "calibration applicability must name at least one scenario",
    );
  }
  assertUnique(
    dataset.applicability.scenarioIds,
    "calibration applicability scenario ids",
  );
  for (const scenarioId of dataset.applicability.scenarioIds) {
    assertNonEmpty(scenarioId, "applicable scenario id");
  }
  for (const deviceKind of DEVICE_KINDS) {
    assertNonEmpty(
      dataset.applicability.deviceKindLabels[deviceKind],
      `${deviceKind} applicability label`,
    );
  }

  assertPositiveSafeInteger(
    dataset.modelConstants.activationBytesPerToken,
    "activation bytes per token",
  );
  assertPositiveSafeInteger(
    dataset.modelConstants.collectiveBytesPerToken,
    "collective bytes per token",
  );
  assertPositiveSafeInteger(
    dataset.modelConstants.coldLoadByteMultiplier,
    "cold load byte multiplier",
  );
  assertPositiveSafeInteger(
    dataset.quality.minSamplesPerPoint,
    "minimum samples per point",
  );
  if (dataset.quality.minSamplesPerPoint < 3) {
    throw new CalibrationError(
      "minimum samples per point must be at least 3",
    );
  }
  assertUnitInterval(
    dataset.quality.maxNormalizedRmse,
    "maximum normalized RMSE",
  );
  assertUnitInterval(
    dataset.quality.maxP95RelativeError,
    "maximum p95 relative error",
  );
  if (dataset.observations.length === 0) {
    throw new CalibrationError("calibration dataset has no observations");
  }
  assertUnique(
    dataset.observations.map((observation) => observation.id),
    "calibration observation ids",
  );
  if (dataset.transportObservations.length === 0) {
    throw new CalibrationError(
      "calibration dataset has no transport observations",
    );
  }
  assertUnique(
    dataset.transportObservations.map((observation) => observation.id),
    "transport calibration observation ids",
  );

  for (const observation of dataset.observations) {
    assertNonEmpty(observation.id, "observation id");
    if (!DEVICE_KINDS.includes(observation.deviceKind)) {
      throw new CalibrationError(
        `observation ${observation.id} has unsupported device kind ${String(observation.deviceKind)}`,
      );
    }
    if (!ALL_CAPABILITIES.includes(observation.capability)) {
      throw new CalibrationError(
        `observation ${observation.id} has unsupported capability ${String(observation.capability)}`,
      );
    }
    assertNonEmpty(observation.regime, `observation ${observation.id} regime`);
    if (
      !Number.isSafeInteger(observation.workItems)
      || observation.workItems < 0
    ) {
      throw new CalibrationError(
        `observation ${observation.id} work items must be a non-negative safe integer`,
      );
    }
    if (
      observation.capability === "invocation"
      && observation.workItems !== 0
    ) {
      throw new CalibrationError(
        `invocation observation ${observation.id} must have zero work items`,
      );
    }
    if (
      observation.capability !== "invocation"
      && observation.workItems === 0
    ) {
      throw new CalibrationError(
        `compute observation ${observation.id} must have positive work items`,
      );
    }
    if (
      observation.durationsNs.length
      < dataset.quality.minSamplesPerPoint
    ) {
      throw new CalibrationError(
        `observation ${observation.id} has ${observation.durationsNs.length} samples; expected at least ${dataset.quality.minSamplesPerPoint}`,
      );
    }
    for (const durationNs of observation.durationsNs) {
      assertPositiveSafeInteger(
        durationNs,
        `observation ${observation.id} duration`,
      );
    }
  }

  for (const deviceKind of DEVICE_KINDS) {
    for (const capability of ALL_CAPABILITIES) {
      const observations = observationsFor(dataset, deviceKind, capability);
      const distinctWorkItems = new Set(
        observations.map((observation) => observation.workItems),
      );
      const requiredPoints = capability === "invocation" ? 1 : 2;
      if (distinctWorkItems.size < requiredPoints) {
        throw new CalibrationError(
          `${deviceKind} ${capability} requires at least ${requiredPoints} distinct work-item points`,
        );
      }
    }
  }

  const transportGroups = new Map<
    string,
    TransportCalibrationObservation[]
  >();
  for (const observation of dataset.transportObservations) {
    assertNonEmpty(observation.id, "transport observation id");
    assertNonEmpty(
      observation.scenarioId,
      `transport observation ${observation.id} scenario id`,
    );
    if (!dataset.applicability.scenarioIds.includes(observation.scenarioId)) {
      throw new CalibrationError(
        `transport observation ${observation.id} scenario ${observation.scenarioId} is outside applicability`,
      );
    }
    if (
      observation.operation !== "transfer"
      && observation.operation !== "collective"
    ) {
      throw new CalibrationError(
        `transport observation ${observation.id} has unsupported operation ${String(observation.operation)}`,
      );
    }
    if (observation.linkIds.length === 0) {
      throw new CalibrationError(
        `transport observation ${observation.id} must name at least one link`,
      );
    }
    if (
      new Set(observation.linkIds).size !== observation.linkIds.length
    ) {
      throw new CalibrationError(
        `transport observation ${observation.id} link ids must be unique`,
      );
    }
    if (
      observation.operation === "transfer"
      && observation.linkIds.length !== 1
    ) {
      throw new CalibrationError(
        `transfer observation ${observation.id} must name exactly one link`,
      );
    }
    assertPositiveSafeInteger(
      observation.participantCount,
      `transport observation ${observation.id} participant count`,
    );
    if (
      observation.operation === "transfer"
      && observation.participantCount !== 2
    ) {
      throw new CalibrationError(
        `transfer observation ${observation.id} must have two participants`,
      );
    }
    if (
      observation.operation === "collective"
      && observation.participantCount < 2
    ) {
      throw new CalibrationError(
        `collective observation ${observation.id} must have at least two participants`,
      );
    }
    assertNonEmpty(
      observation.algorithm,
      `transport observation ${observation.id} algorithm`,
    );
    if (
      observation.operation === "collective"
      && observation.algorithm === EXPERT_COLLECTIVE_CALIBRATION_ALGORITHM
    ) {
      if (observation.trafficSignature === undefined) {
        throw new CalibrationError(
          `transport observation ${observation.id} requires an AllToAllV traffic signature`,
        );
      }
      try {
        assertCanonicalAllToAllTrafficSignature(
          observation.trafficSignature,
          observation.participantCount,
        );
      } catch (error) {
        throw new CalibrationError(
          `transport observation ${observation.id}: ${errorMessage(error)}`,
        );
      }
    } else if (observation.trafficSignature !== undefined) {
      throw new CalibrationError(
        `transport observation ${observation.id} cannot declare a traffic signature`,
      );
    }
    assertNonEmpty(
      observation.regime,
      `transport observation ${observation.id} regime`,
    );
    assertPositiveSafeInteger(
      observation.bytes,
      `transport observation ${observation.id} bytes`,
    );
    if (
      observation.durationsNs.length
      < dataset.quality.minSamplesPerPoint
    ) {
      throw new CalibrationError(
        `transport observation ${observation.id} has ${observation.durationsNs.length} samples; expected at least ${dataset.quality.minSamplesPerPoint}`,
      );
    }
    for (const durationNs of observation.durationsNs) {
      assertPositiveSafeInteger(
        durationNs,
        `transport observation ${observation.id} duration`,
      );
    }
    const identity = transportObservationIdentity(observation);
    const group = transportGroups.get(identity) ?? [];
    group.push(observation);
    transportGroups.set(identity, group);
  }
  for (const [identity, observations] of transportGroups) {
    const bytePoints = observations.map((observation) => observation.bytes);
    if (new Set(bytePoints).size !== observations.length) {
      throw new CalibrationError(
        `transport curve ${identity} byte points must be unique`,
      );
    }
    if (observations.length < 2) {
      throw new CalibrationError(
        `transport curve ${identity} requires at least 2 distinct byte points`,
      );
    }
  }
}

function buildDiagnostic(
  dataset: CalibrationDataset,
  deviceKind: SimDeviceKind,
  capability: CalibratedCapability,
  observations: readonly CalibrationObservation[],
  invocationOverheadNs: number,
  coefficientNs: number,
): CalibrationFitDiagnostic {
  const residuals: number[] = [];
  const relativeErrors: number[] = [];
  let actualTotal = 0;
  let sampleCount = 0;
  for (const observation of observations) {
    const predicted = capability === "invocation"
      ? coefficientNs
      : invocationOverheadNs + coefficientNs * observation.workItems;
    for (const actual of observation.durationsNs) {
      const residual = predicted - actual;
      residuals.push(residual);
      relativeErrors.push(Math.abs(residual) / actual);
      actualTotal += actual;
      sampleCount++;
    }
  }
  const rmse = Math.sqrt(
    residuals.reduce((sum, residual) => sum + residual ** 2, 0)
    / sampleCount,
  );
  const normalizedRmse = rmse / (actualTotal / sampleCount);
  const p95RelativeError = percentile(relativeErrors, 0.95);
  if (normalizedRmse > dataset.quality.maxNormalizedRmse) {
    throw new CalibrationError(
      `${deviceKind} ${capability} normalized RMSE ${formatRatio(normalizedRmse)} exceeds limit ${formatRatio(dataset.quality.maxNormalizedRmse)}`,
    );
  }
  if (p95RelativeError > dataset.quality.maxP95RelativeError) {
    throw new CalibrationError(
      `${deviceKind} ${capability} p95 relative error ${formatRatio(p95RelativeError)} exceeds limit ${formatRatio(dataset.quality.maxP95RelativeError)}`,
    );
  }
  const workItems = observations.map((observation) => observation.workItems);
  return {
    deviceKind,
    capability,
    observationPoints: observations.length,
    samples: sampleCount,
    minWorkItems: Math.min(...workItems),
    maxWorkItems: Math.max(...workItems),
    coefficientNs,
    normalizedRmse,
    p95RelativeError,
  };
}

function fitTransportCurves(dataset: CalibrationDataset): {
  readonly curves: readonly TransportCalibrationCurve[];
  readonly diagnostics: readonly TransportCalibrationFitDiagnostic[];
} {
  const grouped = new Map<string, TransportCalibrationObservation[]>();
  for (const observation of dataset.transportObservations) {
    const identity = transportObservationIdentity(observation);
    const group = grouped.get(identity) ?? [];
    group.push(observation);
    grouped.set(identity, group);
  }
  const curves: TransportCalibrationCurve[] = [];
  const diagnostics: TransportCalibrationFitDiagnostic[] = [];
  for (const [identity, unsorted] of [...grouped.entries()].sort(
    ([left], [right]) => left.localeCompare(right),
  )) {
    const observations = [...unsorted].sort(
      (left, right) => left.bytes - right.bytes,
    );
    const first = observations[0];
    const points = observations.map((observation) => ({
      bytes: observation.bytes,
      durationNs: roundedPositiveCoefficient(
        median(observation.durationsNs),
        `transport observation ${observation.id} median`,
      ),
    }));
    for (let index = 1; index < points.length; index++) {
      if (points[index].durationNs < points[index - 1].durationNs) {
        throw new CalibrationError(
          `transport curve ${identity} median duration must be non-decreasing with bytes`,
        );
      }
    }
    const residuals: number[] = [];
    const relativeErrors: number[] = [];
    let actualTotal = 0;
    let sampleCount = 0;
    for (let index = 0; index < observations.length; index++) {
      const predicted = points[index].durationNs;
      for (const actual of observations[index].durationsNs) {
        const residual = predicted - actual;
        residuals.push(residual);
        relativeErrors.push(Math.abs(residual) / actual);
        actualTotal += actual;
        sampleCount++;
      }
    }
    const normalizedRmse = Math.sqrt(
      residuals.reduce((sum, residual) => sum + residual ** 2, 0)
      / sampleCount,
    ) / (actualTotal / sampleCount);
    const p95RelativeError = percentile(relativeErrors, 0.95);
    if (normalizedRmse > dataset.quality.maxNormalizedRmse) {
      throw new CalibrationError(
        `transport curve ${identity} normalized RMSE ${formatRatio(normalizedRmse)} exceeds limit ${formatRatio(dataset.quality.maxNormalizedRmse)}`,
      );
    }
    if (p95RelativeError > dataset.quality.maxP95RelativeError) {
      throw new CalibrationError(
        `transport curve ${identity} p95 relative error ${formatRatio(p95RelativeError)} exceeds limit ${formatRatio(dataset.quality.maxP95RelativeError)}`,
      );
    }
    curves.push({
      scenarioId: first.scenarioId,
      operation: first.operation,
      linkIds: [...first.linkIds],
      participantCount: first.participantCount,
      algorithm: first.algorithm,
      ...(first.trafficSignature === undefined
        ? {}
        : { trafficSignature: first.trafficSignature }),
      points,
    });
    diagnostics.push({
      scenarioId: first.scenarioId,
      operation: first.operation,
      linkIds: [...first.linkIds],
      participantCount: first.participantCount,
      algorithm: first.algorithm,
      ...(first.trafficSignature === undefined
        ? {}
        : { trafficSignature: first.trafficSignature }),
      observationPoints: observations.length,
      samples: sampleCount,
      minBytes: points[0].bytes,
      maxBytes: points[points.length - 1].bytes,
      normalizedRmse,
      p95RelativeError,
    });
  }
  return { curves, diagnostics };
}

function observationsFor(
  dataset: CalibrationDataset,
  deviceKind: SimDeviceKind,
  capability: CalibratedCapability,
): readonly CalibrationObservation[] {
  return dataset.observations.filter((observation) => (
    observation.deviceKind === deviceKind
    && observation.capability === capability
  ));
}

function fingerprintDataset(dataset: CalibrationDataset): string {
  const canonical = JSON.stringify({
    revision: dataset.revision,
    id: dataset.id,
    provenance: {
      kind: dataset.provenance.kind,
      source: dataset.provenance.source,
      measuredAt: dataset.provenance.measuredAt,
      softwareStack: dataset.provenance.softwareStack,
      modelArtifact: dataset.provenance.modelArtifact,
      notes: dataset.provenance.notes,
    },
    applicability: {
      scenarioIds: [...dataset.applicability.scenarioIds].sort(),
      deviceKindLabels: {
        cpu: dataset.applicability.deviceKindLabels.cpu,
        gpu: dataset.applicability.deviceKindLabels.gpu,
        npu: dataset.applicability.deviceKindLabels.npu,
      },
    },
    modelConstants: {
      activationBytesPerToken:
        dataset.modelConstants.activationBytesPerToken,
      collectiveBytesPerToken:
        dataset.modelConstants.collectiveBytesPerToken,
      coldLoadByteMultiplier:
        dataset.modelConstants.coldLoadByteMultiplier,
    },
    quality: {
      minSamplesPerPoint: dataset.quality.minSamplesPerPoint,
      maxNormalizedRmse: dataset.quality.maxNormalizedRmse,
      maxP95RelativeError: dataset.quality.maxP95RelativeError,
    },
    observations: [...dataset.observations]
      .map((observation) => ({
        id: observation.id,
        deviceKind: observation.deviceKind,
        capability: observation.capability,
        workItems: observation.workItems,
        durationsNs: [...observation.durationsNs].sort((left, right) => (
          left - right
        )),
        regime: observation.regime,
      }))
      .sort((left, right) => left.id.localeCompare(right.id)),
    transportObservations: [...dataset.transportObservations]
      .map((observation) => ({
        id: observation.id,
        scenarioId: observation.scenarioId,
        operation: observation.operation,
        linkIds: [...observation.linkIds],
        participantCount: observation.participantCount,
        algorithm: observation.algorithm,
        trafficSignature: observation.trafficSignature,
        bytes: observation.bytes,
        durationsNs: [...observation.durationsNs].sort((left, right) => (
          left - right
        )),
        regime: observation.regime,
      }))
      .sort((left, right) => left.id.localeCompare(right.id)),
  });
  let hash = 0x811c9dc5;
  for (let index = 0; index < canonical.length; index++) {
    hash ^= canonical.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return `fnv1a32:${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function roundedPositiveCoefficient(value: number, label: string): number {
  const rounded = Math.round(value);
  if (!Number.isSafeInteger(rounded) || rounded <= 0) {
    throw new CalibrationError(
      `${label} must fit to a positive safe integer; got ${String(value)}`,
    );
  }
  return rounded;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function percentile(values: readonly number[], quantile: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(quantile * sorted.length) - 1)];
}

function transportObservationIdentity(
  observation: TransportCalibrationObservation,
): string {
  return JSON.stringify({
    scenarioId: observation.scenarioId,
    operation: observation.operation,
    algorithm: observation.algorithm,
    participantCount: observation.participantCount,
    linkIds: observation.linkIds,
    trafficSignature: observation.trafficSignature,
  });
}

function optionalTransportTrafficSignature(
  observation: Record<string, unknown>,
  context: string,
): { readonly trafficSignature?: string } {
  const trafficSignature = optionalString(
    observation,
    "traffic_signature",
    context,
  );
  return trafficSignature === undefined ? {} : { trafficSignature };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function assertPositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new CalibrationError(`${label} must be a positive safe integer`);
  }
}

function assertUnitInterval(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0 || value > 1) {
    throw new CalibrationError(`${label} must be in (0, 1]`);
  }
}

function assertNonEmpty(value: string, label: string): void {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new CalibrationError(`${label} must be non-empty`);
  }
}

function assertUnique(values: readonly string[], label: string): void {
  if (new Set(values).size !== values.length) {
    throw new CalibrationError(`${label} must be unique`);
  }
}

function formatRatio(value: number): string {
  return value.toFixed(4);
}

function requireRecord(
  value: unknown,
  path: string,
): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new CalibrationError(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireString(
  record: Record<string, unknown>,
  key: string,
  path: string,
): string {
  const value = record[key];
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new CalibrationError(`${path}.${key} must be a non-empty string`);
  }
  return value;
}

function optionalString(
  record: Record<string, unknown>,
  key: string,
  path: string,
): string | undefined {
  return record[key] === undefined
    ? undefined
    : requireString(record, key, path);
}

function requireNumber(
  record: Record<string, unknown>,
  key: string,
  path: string,
): number {
  const value = record[key];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new CalibrationError(`${path}.${key} must be a finite number`);
  }
  return value;
}

function requireStringArray(
  record: Record<string, unknown>,
  key: string,
  path: string,
): string[] {
  const value = record[key];
  if (
    !Array.isArray(value)
    || value.some((entry) => (
      typeof entry !== "string" || entry.trim().length === 0
    ))
  ) {
    throw new CalibrationError(
      `${path}.${key} must be an array of non-empty strings`,
    );
  }
  return value;
}

function requireNumberArray(
  record: Record<string, unknown>,
  key: string,
  path: string,
): number[] {
  const value = record[key];
  if (
    !Array.isArray(value)
    || value.some((entry) => (
      typeof entry !== "number" || !Number.isFinite(entry)
    ))
  ) {
    throw new CalibrationError(
      `${path}.${key} must be an array of finite numbers`,
    );
  }
  return value;
}

function requireRecordArray(
  record: Record<string, unknown>,
  key: string,
  path: string,
): Record<string, unknown>[] {
  const value = record[key];
  if (!Array.isArray(value)) {
    throw new CalibrationError(`${path}.${key} must be an array`);
  }
  return value.map((entry, index) => (
    requireRecord(entry, `${path}.${key}[${index}]`)
  ));
}

function requireEvidenceKind(value: string): CalibrationEvidenceKind {
  if (value !== "measured" && value !== "synthetic") {
    throw new CalibrationError(
      `unsupported calibration evidence kind ${value}`,
    );
  }
  return value;
}

function requireDeviceKind(
  value: string,
  context: string,
): SimDeviceKind {
  if (!DEVICE_KINDS.includes(value as SimDeviceKind)) {
    throw new CalibrationError(
      `${context}.device_kind must be cpu, gpu, or npu`,
    );
  }
  return value as SimDeviceKind;
}

function requireCapability(
  value: string,
  context: string,
): CalibratedCapability {
  if (!ALL_CAPABILITIES.includes(value as CalibratedCapability)) {
    throw new CalibrationError(
      `${context}.capability must be invocation, attention, ffn, draft, or lookup`,
    );
  }
  return value as CalibratedCapability;
}

function requireTransportOperation(
  value: string,
  context: string,
): TransportOperationKind {
  if (value !== "transfer" && value !== "collective") {
    throw new CalibrationError(
      `${context}.operation must be transfer or collective`,
    );
  }
  return value;
}
