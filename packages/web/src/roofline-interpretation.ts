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
    return withoutComputeRoof(roofline, roof, points);
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

/** Share of the bandwidth ceiling a point reaches at its own intensity. */
function bandwidthUtilisation(
  roof: BandwidthRoof,
  point: RooflinePoint,
): number {
  const ceiling = roof.bytesPerSecond * point.arithmeticIntensity;
  return ceiling > 0 ? point.predictedFlopsPerSecond / ceiling : 0;
}

/** Why the compute ceiling is missing, and what to do about that specific why. */
function absenceWording(
  roofline: DashboardRooflineResult,
  roof: BandwidthRoof,
): { readonly cause: string; readonly nextStep: string } {
  const absence = roofline.computeRoofAbsence;
  const devices = absence === undefined || absence.deviceLabels.length === 0
    ? "the selected devices"
    : absence.deviceLabels.join(", ");
  switch (absence?.reason) {
    case "profile_publishes_none":
      return {
        cause: `${devices} is bound to real hardware, but its vendor publishes no absolute compute peak for any dtype, so no ceiling exists to draw. Choosing a different dtype will not produce one.`,
        // Naming the control matters: a measured peak replaces the vendor
        // profile rather than supplementing it, and the two cannot coexist.
        nextStep: "In the topology editor set this device's Hardware compute profile to Custom and enter a measured peak, which replaces the vendor profile, or read this chart as a bandwidth bound only.",
      };
    case "dtype_not_published":
      return {
        // The dtype that matters here is what the arithmetic runs in, which
        // is the activation format. A reader looking at an INT4 weight badge
        // would otherwise declare a peak for the wrong one.
        cause: `${devices} publishes dense peaks for ${
          absence.publishedDtypes.join(", ")
        }, but not for ${absence.dtype}. That is this run's activation format, which is what the arithmetic executes in regardless of how the weights are stored.`,
        nextStep: `Run the model in one of the published dtypes to see a compute ceiling, or set this device's Hardware compute profile to Custom and enter a measured ${absence.dtype} peak, which replaces the vendor profile.`,
      };
    case "mixed_devices":
      return {
        cause: `The work spans devices of which ${devices} has no compute profile, so the peaks cannot be summed into one ceiling.`,
        nextStep: "Bind a compute profile to every device carrying this work, or filter the chart to devices that have one.",
      };
    case "no_profile_narrow_dtype":
      return {
        cause: `${devices} has no compute profile bound, and ${absence.dtype} is too narrow for a generic rate to be defensible.`,
        nextStep: "Select a concrete machine preset, or declare measured peaks for these devices.",
      };
    default:
      return {
        cause: `${devices} has no compute profile bound, so no peak rate is known.`,
        nextStep: "Select a concrete machine preset, or declare measured peaks for these devices.",
      };
  }
}

/**
 * Read the chart when only a bandwidth roof exists.
 *
 * A missing compute ceiling does not always mean nothing can be concluded.
 * Work sitting at its bandwidth ceiling is bandwidth bound whatever the
 * compute ceiling turns out to be, because a higher one cannot lift it; only
 * work comfortably below the ceiling leaves the question genuinely open.
 */
function withoutComputeRoof(
  roofline: DashboardRooflineResult,
  roof: BandwidthRoof,
  points: readonly RooflinePoint[],
): RooflineInterpretation {
  const { cause, nextStep } = absenceWording(roofline, roof);
  const utilisations = points.map(
    (point) => bandwidthUtilisation(roof, point),
  );
  const saturated = utilisations.filter((value) => value >= 0.9).length;
  const percent = (value: number) => `${(value * 100).toFixed(0)}%`;
  const pointLabels = Object.fromEntries(points.map((point, index) => {
    const share = utilisations[index]!;
    return [
      point.id,
      share >= 0.9
        ? `At ${percent(share)} of the ${roof.label} ceiling, so bandwidth binds here whatever the compute ceiling is.`
        : `At ${percent(share)} of the ${roof.label} ceiling, leaving ${
            percent(1 - share)
          } that only a compute ceiling could explain.`,
    ];
  }));

  if (saturated === points.length) {
    return {
      tone: "neutral",
      verdict: "Bandwidth bound",
      explanation: `Every visible point sits at or above 90% of the ${roof.label} ceiling, which is conclusive on its own: a compute ceiling can only be higher, so it cannot be what limits this work. ${cause}`,
      nextStep: `Raise bandwidth or arithmetic intensity to move these points. ${nextStep}`,
      pointLabels,
    };
  }
  const headroom = Math.max(...utilisations);
  return {
    tone: "warning",
    verdict: saturated > 0
      ? "Bandwidth bound in part"
      : "Limiting resource unproven",
    explanation: saturated > 0
      ? `${saturated} of ${points.length} points sit at the ${roof.label} ceiling and are bandwidth bound. The rest reach only ${percent(headroom)} of it, and what holds them there cannot be shown without a compute ceiling. ${cause}`
      : `The fastest point reaches ${percent(headroom)} of the ${roof.label} ceiling, so bandwidth alone does not explain the rate, and no compute ceiling is available to test against. ${cause}`,
    nextStep,
    pointLabels,
  };
}
