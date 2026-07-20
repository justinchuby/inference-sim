import type {
  SimulationScenario,
  TopologyCostModel,
  TopologyServingResult,
  TopologyWorkloadResult,
} from "@inference-sim/core";
import type {
  DashboardModelBinding,
  DashboardRooflineResult,
  RooflinePhase,
} from "./types.js";

export const ROOFLINE_SUMMARY_REVISION = 1;

interface RooflineInput {
  readonly scenario: SimulationScenario;
  readonly model?: DashboardModelBinding;
  readonly costModel: TopologyCostModel;
  readonly topology?: TopologyWorkloadResult;
  readonly serving?: TopologyServingResult;
  readonly mode: "serving" | "pipeline" | "speculative" | "expert-cache";
}

interface PointSeed {
  readonly id: string;
  readonly label: string;
  readonly phase: RooflinePhase;
  readonly width: number;
  readonly tokens?: number;
  readonly durationNs: number;
  readonly componentId?: string;
  readonly weightBytes?: number;
}

export function buildDashboardRoofline(
  input: RooflineInput,
): DashboardRooflineResult {
  const bandwidthRoofs = buildBandwidthRoofs(input.scenario);
  const model = input.model;
  if (model === undefined) {
    return unavailable(
      bandwidthRoofs,
      "Import or select a model to calculate arithmetic intensity and compute work.",
    );
  }
  const profile = model.executionProfile;
  if (
    profile.forwardFlopsPerToken <= 0
    || profile.attentionWeightBytesPerToken + profile.ffnWeightBytesPerToken <= 0
  ) {
    return unavailable(
      bandwidthRoofs,
      "The selected model has no usable FLOP and active-weight execution profile.",
    );
  }
  const targetDevices = targetDeviceIds(input.scenario);
  const computeDtype = model.modelFormat?.weightQuantization === "none"
    ? model.modelFormat.weightDtypes[0]?.toLowerCase() ?? "fp16"
    : model.modelFormat?.weightQuantization ?? "unknown";
  const computeRoof = effectiveComputeRoof(
    input,
    targetDevices,
    computeDtype,
  );
  const localRoof = preferredLocalRoof(input.scenario, targetDevices)
    ?? bandwidthRoofs[0];
  const seeds = pointSeeds(input);
  const defaultWeightBytes = profile.attentionWeightBytesPerToken
    + profile.ffnWeightBytesPerToken;
  const points = seeds.flatMap((seed) => {
    const activeBytes = seed.weightBytes ?? defaultWeightBytes;
    const flopsPerByte = profile.forwardFlopsPerToken / defaultWeightBytes;
    const workFlops = seed.weightBytes === undefined
      ? profile.forwardFlopsPerToken * Math.max(1, seed.width)
      : seed.weightBytes * flopsPerByte * Math.max(1, seed.width);
    if (!(activeBytes > 0 && workFlops > 0 && seed.durationNs > 0)) {
      return [];
    }
    const arithmeticIntensity = workFlops / activeBytes;
    const predictedFlopsPerSecond = workFlops * 1e9 / seed.durationNs;
    const memoryCeiling = localRoof === undefined
      ? Infinity
      : localRoof.bytesPerSecond * arithmeticIntensity;
    const limitingRoofId = computeRoof === undefined
      ? "unresolved"
      : computeRoof.flopsPerSecond <= memoryCeiling
        ? "compute"
        : localRoof?.id ?? "unresolved";
    return [{
      id: seed.id,
      label: seed.label,
      phase: seed.phase,
      ...(seed.componentId === undefined
        ? {}
        : { componentId: seed.componentId }),
      deviceIds: targetDevices,
      workFlops,
      activeBytes,
      durationNs: seed.durationNs,
      arithmeticIntensity,
      predictedFlopsPerSecond,
      ...(seed.tokens === undefined || seed.tokens <= 0
        ? {}
        : { predictedTokensPerSecond: seed.tokens * 1e9 / seed.durationNs }),
      limitingRoofId,
      confidence: input.topology?.confidence
        ?? input.serving?.confidence
        ?? "heuristic",
      notes: [
        `Active weights are amortized across ${Math.max(1, seed.width)} token${seed.width === 1 ? "" : "s"} in this execution step.`,
        "Predicted throughput uses simulated replay wall time; diagonal roofs use declared resource bandwidth.",
      ],
    }];
  });
  return {
    revision: ROOFLINE_SUMMARY_REVISION,
    status: points.length === 0 ? "unavailable" : "available",
    confidence: input.topology?.confidence
      ?? input.serving?.confidence
      ?? "heuristic",
    assumptions: [
      "Arithmetic intensity is model FLOPs divided by active weight bytes at the selected execution width.",
      "The compute roof is an effective ceiling from the active topology cost model, not a vendor hardware peak.",
      "Interconnect and storage roofs are counterfactual ceilings unless the replay actually routes model data through that tier.",
      "Pipeline component FLOPs are allocated in proportion to component weight bytes and are heuristic.",
      "Speculative proposer work is excluded unless a separate proposer execution profile is available.",
    ],
    ...(points.length === 0
      ? { unavailableReason: "No positive-duration model work was produced." }
      : {}),
    ...(computeRoof === undefined ? {} : { computeRoof }),
    bandwidthRoofs,
    points,
  };
}

function pointSeeds(input: RooflineInput): readonly PointSeed[] {
  if (input.serving !== undefined) {
    const aggregates = new Map<RooflinePhase, PointSeed>();
    for (const batch of input.serving.batches) {
      const hasPrefill = batch.work.prefill.length > 0;
      const hasDecode = batch.work.decode.length > 0;
      const phase: RooflinePhase = hasPrefill && hasDecode
        ? "mixed_batch"
        : hasPrefill
          ? "prefill"
          : batch.work.decode.some((slice) => slice.mode === "speculative")
            ? "spec_verify"
            : "decode";
      const current = aggregates.get(phase);
      aggregates.set(phase, {
        id: `serving-${phase}`,
        label: phaseLabel(phase),
        phase,
        width: (current?.width ?? 0) + Math.max(1, batch.work.tokenWork),
        tokens: (current?.tokens ?? 0) + Math.max(1, batch.work.expectedOutputTokens),
        durationNs: (current?.durationNs ?? 0) + Math.max(1, batch.durationNs),
      });
    }
    const routed = input.serving.batches.filter(
      (batch) => batch.expertRoutes.length > 0,
    );
    const moe = routed.length === 0 || input.model === undefined
      ? []
      : [{
          id: "serving-moe",
          label: "MoE routed FFN",
          phase: "moe" as const,
          width: routed.reduce(
            (sum, batch) => sum + Math.max(1, batch.work.tokenWork),
            0,
          ),
          tokens: routed.reduce(
            (sum, batch) => sum + batch.expertRoutes.length,
            0,
          ),
          durationNs: routed.reduce(
            (sum, batch) => sum + Math.max(1, batch.durationNs),
            0,
          ),
          weightBytes: input.model.executionProfile.ffnWeightBytesPerToken,
        }];
    return [...aggregates.values(), ...moe];
  }
  const topology = input.topology;
  if (topology === undefined) {
    return [];
  }
  if (input.mode === "pipeline" && input.model?.pipelineExecution !== undefined) {
    const components = input.model.pipelineExecution.components;
    const totalWeight = components.reduce((sum, component) => (
      sum + component.weightBytes
    ), 0);
    return components.map((component) => ({
      id: `component-${component.id}`,
      label: component.role,
      phase: "pipeline",
      componentId: component.id,
      width: Math.max(1, component.invocationMultiplier),
      weightBytes: component.weightBytes,
      durationNs: Math.max(
        1,
        topology.metrics.totalDurationNs * component.weightBytes / totalWeight,
      ),
    }));
  }
  const phase: RooflinePhase = input.mode === "speculative"
    ? "spec_verify"
    : input.mode === "expert-cache"
      ? "moe"
      : "decode";
  const tokens = Math.max(1, topology.metrics.committedTokens);
  return [{
    id: phase,
    label: phaseLabel(phase),
    phase,
    width: input.mode === "speculative"
      ? Math.max(1, topology.plan.steps.filter(
          (step) => step.operation.kind === "compute",
        ).length)
      : 1,
    tokens,
    durationNs: Math.max(1, topology.metrics.foregroundDurationNs),
  }];
}

function effectiveComputeRoof(
  input: RooflineInput,
  deviceIds: readonly string[],
  dtype: string,
): DashboardRooflineResult["computeRoof"] | undefined {
  if (["int4", "int2", "int1", "nf4", "mixed", "unknown"].includes(dtype)) {
    return undefined;
  }
  const profile = input.model?.executionProfile;
  if (profile === undefined) {
    return undefined;
  }
  let total = 0;
  for (const id of deviceIds) {
    const device = input.scenario.devices.find((candidate) => candidate.id === id);
    if (device === undefined || !device.supportedDtypes.includes(dtype)) {
      continue;
    }
    const cost = input.costModel.deviceCosts[device.kind];
    const serviceNs = cost.attentionNsPerToken + cost.ffnNsPerToken;
    total += profile.forwardFlopsPerToken * 1e9 / serviceNs;
  }
  if (!(total > 0)) {
    return undefined;
  }
  return {
    label: "Effective compute",
    flopsPerSecond: total,
    evidence: input.costModel.confidence === "calibrated"
      ? "calibrated_effective"
      : "heuristic_effective",
    dtype,
  };
}

function buildBandwidthRoofs(
  scenario: SimulationScenario,
): DashboardRooflineResult["bandwidthRoofs"] {
  const domains = scenario.memoryDomains.map((domain) => ({
    id: `memory:${domain.id}`,
    label: domain.id,
    kind: domain.kind === "device" || domain.kind === "unified"
      ? "device_memory" as const
      : domain.kind === "storage"
        ? "storage" as const
        : "host_memory" as const,
    bytesPerSecond: domain.bandwidthBytesPerSec,
    confidence: domain.provenance.confidence,
  }));
  const links = scenario.links.map((link) => ({
    id: `link:${link.id}`,
    label: link.id,
    kind: "interconnect" as const,
    bytesPerSecond: link.bandwidthBytesPerSec,
    confidence: link.provenance.confidence,
  }));
  const network = (scenario.networkResources ?? []).map((resource) => ({
    id: `network:${resource.id}`,
    label: resource.id,
    kind: "interconnect" as const,
    bytesPerSecond: resource.bandwidthBytesPerSec,
    confidence: resource.provenance.confidence,
  }));
  return [...domains, ...links, ...network]
    .filter((roof) => roof.bytesPerSecond > 0)
    .sort((left, right) => right.bytesPerSecond - left.bytesPerSecond);
}

function preferredLocalRoof(
  scenario: SimulationScenario,
  deviceIds: readonly string[],
): DashboardRooflineResult["bandwidthRoofs"][number] | undefined {
  const domains = new Set(deviceIds.flatMap((id) => (
    scenario.devices.find((device) => device.id === id)?.memoryDomainIds ?? []
  )));
  return buildBandwidthRoofs(scenario).find((roof) => (
    roof.kind === "device_memory"
    && domains.has(roof.id.replace(/^memory:/, ""))
  ));
}

function targetDeviceIds(scenario: SimulationScenario): readonly string[] {
  const ids = scenario.placements.filter((placement) => (
    placement.requiredCapabilities.includes("attention")
    || placement.requiredCapabilities.includes("ffn")
  )).map((placement) => placement.deviceId);
  return [...new Set(ids)];
}

function unavailable(
  bandwidthRoofs: DashboardRooflineResult["bandwidthRoofs"],
  reason: string,
): DashboardRooflineResult {
  return {
    revision: ROOFLINE_SUMMARY_REVISION,
    status: "unavailable",
    confidence: "heuristic",
    assumptions: [],
    unavailableReason: reason,
    bandwidthRoofs,
    points: [],
  };
}

function phaseLabel(phase: RooflinePhase): string {
  return phase.replaceAll("_", " ").replace(/^./, (value) => value.toUpperCase());
}
