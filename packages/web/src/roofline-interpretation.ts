import type { DashboardRooflineResult } from "./types.js";

type BandwidthRoof = DashboardRooflineResult["bandwidthRoofs"][number];
type RooflinePoint = DashboardRooflineResult["points"][number];

export interface RooflineInterpretation {
  readonly tone: "neutral" | "warning" | "danger";
  readonly verdict: string;
  readonly explanation: string;
  readonly nextStep: string;
  readonly knee?: number;
  readonly pointLabels: Readonly<Record<string, string>>;
}

export function interpretRoofline(
  roofline: DashboardRooflineResult,
  roof: BandwidthRoof,
  points: readonly RooflinePoint[],
): RooflineInterpretation {
  if (points.length === 0) {
    return {
      tone: "warning",
      verdict: "No work in this filter",
      explanation: "The selected phase did not produce a plottable model-work point.",
      nextStep: "Select All or another phase to restore the comparison.",
      pointLabels: {},
    };
  }
  const aggregateDeviceCount = new Set(points.flatMap(
    (point) => point.deviceIds,
  )).size;
  if (
    aggregateDeviceCount > 1
    && roof.kind === "device_memory"
    && roof.id.startsWith("memory:")
  ) {
    return {
      tone: "warning",
      verdict: "Resource scope mismatch",
      explanation: `The visible work is aggregated across ${aggregateDeviceCount} devices, while ${roof.label} describes one memory domain. Their rates are not directly comparable.`,
      nextStep: "Select All device memory for the aggregate workload, or inspect per-device work before drawing a bottleneck conclusion.",
      pointLabels: Object.fromEntries(points.map((point) => [
        point.id,
        "Aggregate model work cannot be classified against one device-memory roof.",
      ])),
    };
  }
  if (roofline.computeRoof === undefined) {
    return {
      tone: "warning",
      verdict: "Bandwidth ceiling only",
      explanation: `${roof.label} supplies a declared bandwidth roof, but this model dtype has no compatible compute ceiling. A limiting resource cannot be proven from this chart.`,
      nextStep: "Import calibrated compute evidence for this dtype, or compare bandwidth tiers without treating either one as the final bottleneck.",
      pointLabels: Object.fromEntries(points.map((point) => [
        point.id,
        `Compute limit unknown; ${roof.label} is only a bandwidth upper bound.`,
      ])),
    };
  }

  const knee = roofline.computeRoof.flopsPerSecond / roof.bytesPerSecond;
  const classifications = points.map((point) => {
    const bandwidthCeiling = roof.bytesPerSecond * point.arithmeticIntensity;
    const ceiling = Math.min(
      roofline.computeRoof!.flopsPerSecond,
      bandwidthCeiling,
    );
    const ratio = point.predictedFlopsPerSecond / ceiling;
    const side = point.arithmeticIntensity < knee
      ? "bandwidth" as const
      : "compute" as const;
    return { point, ratio, side };
  });
  const conflicts = classifications.filter(({ ratio }) => ratio > 1.05);
  const pointLabels = Object.fromEntries(classifications.map((entry) => {
    if (entry.ratio > 1.05) {
      return [
        entry.point.id,
        `Predicted rate is ${formatPercent(entry.ratio)} of this roof. The timing and roof evidence are inconsistent, so do not claim utilization from this point.`,
      ];
    }
    return [
      entry.point.id,
      entry.side === "bandwidth"
        ? `Left of the ${formatIntensity(knee)} FLOP/B knee: ${roof.label} bandwidth is the lower modeled ceiling.`
        : `Right of the ${formatIntensity(knee)} FLOP/B knee: effective compute is the lower modeled ceiling.`,
    ];
  }));
  if (conflicts.length > 0) {
    const worst = conflicts.reduce((left, right) => (
      right.ratio > left.ratio ? right : left
    ));
    return {
      tone: "danger",
      verdict: "Evidence conflict",
      explanation: `${worst.point.label} predicts ${formatPercent(worst.ratio)} of the selected theoretical roof. The replay timing, model-work estimate, and roof coefficients cannot all describe the same operating regime.`,
      nextStep: "Calibrate this device, dtype, batch shape, and kernel regime before using the chart for capacity claims.",
      knee,
      pointLabels,
    };
  }
  const bandwidthCount = classifications.filter(
    (entry) => entry.side === "bandwidth",
  ).length;
  const computeCount = classifications.length - bandwidthCount;
  if (bandwidthCount === classifications.length) {
    return {
      tone: "neutral",
      verdict: `${roof.label} bandwidth-sensitive`,
      explanation: `All visible points sit left of the ${formatIntensity(knee)} FLOP/B knee. Moving more bytes, not issuing more arithmetic, is the lower modeled ceiling for this resource comparison.`,
      nextStep: roof.kind === "device_memory"
        ? "Try wider prefill or continuous batches to reuse weights, and keep active weights in the fastest local memory."
        : "Reduce traffic across this tier, improve placement, or raise its effective bandwidth before adding compute.",
      knee,
      pointLabels,
    };
  }
  if (computeCount === classifications.length) {
    return {
      tone: "neutral",
      verdict: "Effective compute-sensitive",
      explanation: `All visible points sit right of the ${formatIntensity(knee)} FLOP/B knee. The selected bandwidth is sufficient for the modeled intensity; effective compute is the lower ceiling.`,
      nextStep: "Compare a faster kernel/device or supported compute quantization. More bandwidth alone should not move this modeled ceiling.",
      knee,
      pointLabels,
    };
  }
  return {
    tone: "neutral",
    verdict: "Phase-dependent bottleneck",
    explanation: `${bandwidthCount} visible point${bandwidthCount === 1 ? " is" : "s are"} bandwidth-sensitive and ${computeCount} ${computeCount === 1 ? "is" : "are"} compute-sensitive around the ${formatIntensity(knee)} FLOP/B knee.`,
    nextStep: "Tune prefill, decode, verification, and pipeline components separately; one topology change will not improve every phase equally.",
    knee,
    pointLabels,
  };
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatIntensity(value: number): string {
  if (value >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (value >= 10) return value.toFixed(0);
  if (value >= 1) return value.toFixed(1);
  return value.toPrecision(2);
}
