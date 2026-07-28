import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Move, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import type { DashboardResult } from "./types.js";
import { memoryTimeline } from "./memory-timeline.js";
import { interpretRoofline } from "./roofline-interpretation.js";
import {
  panLogDomain,
  zoomLogDomain,
  type LogDomain,
} from "./roofline-viewport.js";

export default function ResultCharts({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  return (
    <>
      {result.comparison
        ? (
            <section className="panel">
              <SectionHeading
                title="Topology comparison"
                detail="Same workload · lower is better"
              />
              <TopologyComparisonChart result={result} />
            </section>
          )
        : null}
      {result.roofline
        ? (
            <section className="panel">
              <SectionHeading
                title="Hierarchical roofline"
                detail="Predicted replay rate · declared bandwidth roofs"
              />
              <RooflineChart result={result} />
            </section>
          )
        : null}
      {result.coResidency
        ? (
            <section className="panel">
              <SectionHeading
                title="Residency"
                detail={result.coResidency.metrics.fitsWithoutSwapping
                  ? "every model stayed loaded"
                  : "the device swapped models in and out"}
              />
              <CoResidencyChart result={result} />
            </section>
          )
        : null}
      {result.fault
        ? (
            <section className="panel">
              <SectionHeading
                title="Rank outcomes"
                detail={`${result.fault.failedNodeId} failed · quiesced ${
                  formatDurationMs(result.fault.quiescedAtNs)
                }`}
              />
              <FaultRankChart result={result} />
            </section>
          )
        : null}
      <section className="panel">
        <SectionHeading
          title="Memory domains"
          detail="Reserved bytes by what the allocation is for"
        />
        <MemoryChart result={result} />
      </section>
      <MemoryTimelineSection result={result} />
      <section className="panel">
        <SectionHeading
          title="Resource utilization"
          detail={`${result.topology.planSteps.toLocaleString()} replay-verified steps`}
        />
        <ResourceChart result={result} />
      </section>
      {result.mode === "pipeline" ? null : <section className="panel">
        <SectionHeading
          title={result.mode === "speculative"
            ? "Acceptance profile"
            : result.mode === "serving"
              ? "Request latency"
              : "Cache outcomes"}
          detail={result.mode === "speculative"
            ? "Conditional prefix positions"
            : result.mode === "serving"
              ? "First token and completion"
              : "Routed expert accesses"}
        />
        {result.mode === "speculative"
          ? <AcceptanceChart result={result} />
          : result.mode === "serving"
            ? <ServingLatencyChart result={result} />
            : <CacheOutcomeChart result={result} />}
      </section>}
      {result.expertCache
        ? (
            <section className="panel">
              <SectionHeading
                title="Cache partitions"
                detail={`${result.expertCache.hotPartitions.length} hot owners · ${result.expertCache.warmPartitions.length} warm nodes`}
              />
              <CachePartitionPressure result={result} />
              <CachePartitionChart result={result} />
            </section>
          )
        : null}
    </>
  );
}

function RooflineChart({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const roofline = result.roofline!;
  const [roofId, setRoofId] = useState(
    roofline.bandwidthRoofs[0]?.id ?? "",
  );
  const phases = useMemo(() => [
    "all",
    ...new Set(roofline.points.map((point) => point.phase)),
  ], [roofline.points]);
  const [phase, setPhase] = useState("all");
  const [viewport, setViewport] = useState<{
    readonly x: LogDomain;
    readonly y: LogDomain;
  } | null>(null);
  const drag = useRef<{
    readonly x: number;
    readonly y: number;
    readonly viewport: { readonly x: LogDomain; readonly y: LogDomain };
  } | null>(null);
  const frame = useRef<HTMLDivElement>(null);
  const selectedRoof = roofline.bandwidthRoofs.find(
    (roof) => roof.id === roofId,
  ) ?? roofline.bandwidthRoofs[0];
  const points = roofline.points.filter((point) => (
    phase === "all" || point.phase === phase
  ));
  useEffect(() => {
    setViewport(null);
    drag.current = null;
  }, [roofId, phase, roofline]);
  if (roofline.status === "unavailable" || selectedRoof === undefined) {
    return (
      <div className="border-l-2 border-amber-600 bg-amber-50 px-3 py-3 text-sm text-amber-950">
        <strong>Roofline unavailable.</strong>{" "}
        {roofline.unavailableReason ?? "The run has insufficient model evidence."}
      </div>
    );
  }
  const allX = points.map((point) => point.arithmeticIntensity);
  const minX = Math.max(1e-3, Math.min(...allX) / 4);
  const maxX = Math.max(minX * 100, Math.max(...allX) * 4);
  const automaticX: LogDomain = [minX, maxX];
  const automaticSamples = logSamples(minX, maxX, 48).map((intensity) => ({
    intensity,
    bandwidth: selectedRoof.bytesPerSecond * intensity,
    ...(roofline.computeRoof === undefined
      ? {}
      : { compute: roofline.computeRoof.flopsPerSecond }),
  }));
  const data = points.map((point) => ({
    ...point,
    rate: point.predictedFlopsPerSecond,
    name: point.label,
  }));
  const yValues = [
    ...data.map((point) => point.rate),
    ...automaticSamples.map((sample) => Math.min(
      sample.bandwidth,
      sample.compute ?? Infinity,
    )),
  ].filter(Number.isFinite);
  const minY = Math.max(1, Math.min(...yValues) / 4);
  const maxY = Math.max(minY * 100, Math.max(...yValues) * 2);
  const automaticY: LogDomain = [minY, maxY];
  const xDomain = viewport?.x ?? automaticX;
  const yDomain = viewport?.y ?? automaticY;
  const samples = logSamples(xDomain[0], xDomain[1], 64).map((intensity) => ({
    intensity,
    bandwidth: selectedRoof.bytesPerSecond * intensity,
    ...(roofline.computeRoof === undefined
      ? {}
      : { compute: roofline.computeRoof.flopsPerSecond }),
  }));
  const interpretation = interpretRoofline(roofline, selectedRoof, points);
  const interpretationTone = interpretation.tone === "danger"
    ? "border-rose-600 bg-rose-50 text-rose-950"
    : interpretation.tone === "warning"
      ? "border-amber-600 bg-amber-50 text-amber-950"
      : "border-sky-700 bg-sky-50 text-sky-950";
  const setZoom = (factor: number): void => {
    setViewport((current) => ({
      x: zoomLogDomain(current?.x ?? automaticX, factor),
      y: zoomLogDomain(current?.y ?? automaticY, factor),
    }));
  };
  const handlePointerDown = (
    event: ReactPointerEvent<HTMLDivElement>,
  ): void => {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = {
      x: event.clientX,
      y: event.clientY,
      viewport: { x: xDomain, y: yDomain },
    };
  };
  const handlePointerMove = (
    event: ReactPointerEvent<HTMLDivElement>,
  ): void => {
    const start = drag.current;
    const bounds = frame.current?.getBoundingClientRect();
    if (start === null || bounds === undefined) return;
    setViewport({
      x: panLogDomain(
        start.viewport.x,
        -(event.clientX - start.x) / Math.max(1, bounds.width),
      ),
      y: panLogDomain(
        start.viewport.y,
        (event.clientY - start.y) / Math.max(1, bounds.height),
      ),
    });
  };
  const handlePointerEnd = (
    event: ReactPointerEvent<HTMLDivElement>,
  ): void => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    drag.current = null;
  };
  const handleWheel = (event: ReactWheelEvent<HTMLDivElement>): void => {
    event.preventDefault();
    const bounds = frame.current?.getBoundingClientRect();
    if (bounds === undefined) return;
    const xAnchor = Math.min(1, Math.max(
      0,
      (event.clientX - bounds.left) / Math.max(1, bounds.width),
    ));
    const yAnchor = 1 - Math.min(1, Math.max(
      0,
      (event.clientY - bounds.top) / Math.max(1, bounds.height),
    ));
    const factor = event.deltaY > 0 ? 1.18 : 0.84;
    setViewport((current) => ({
      x: zoomLogDomain(current?.x ?? automaticX, factor, xAnchor),
      y: zoomLogDomain(current?.y ?? automaticY, factor, yAnchor),
    }));
  };
  const handleKeyDown = (
    event: ReactKeyboardEvent<HTMLDivElement>,
  ): void => {
    const current = viewport ?? { x: automaticX, y: automaticY };
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      setZoom(0.7);
    } else if (event.key === "-") {
      event.preventDefault();
      setZoom(1.4);
    } else if (event.key === "0") {
      event.preventDefault();
      setViewport(null);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      setViewport({
        x: panLogDomain(current.x, event.key === "ArrowLeft" ? -0.1 : 0.1),
        y: current.y,
      });
    } else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault();
      setViewport({
        x: current.x,
        y: panLogDomain(current.y, event.key === "ArrowUp" ? 0.1 : -0.1),
      });
    }
  };
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-3">
        <label className="min-w-56 text-xs font-semibold text-zinc-600">
          Resource roof
          <select
            className="mt-1 block h-8 w-full rounded-md border border-zinc-300 bg-white px-2 text-xs text-zinc-900"
            value={selectedRoof.id}
            onChange={(event) => setRoofId(event.target.value)}
          >
            {roofline.bandwidthRoofs.map((roof) => (
              <option key={roof.id} value={roof.id}>
                {roof.kind.replaceAll("_", " ")} · {roof.label} · {formatBytesPerSecond(roof.bytesPerSecond)}
              </option>
            ))}
          </select>
        </label>
        <div className="flex flex-wrap gap-1" role="group" aria-label="Roofline phase">
          {phases.map((candidate) => (
            <button
              key={candidate}
              type="button"
              className={`h-8 rounded-md border px-2.5 text-xs font-semibold ${phase === candidate ? "border-sky-700 bg-sky-50 text-sky-800" : "border-zinc-300 bg-white text-zinc-600"}`}
              onClick={() => setPhase(candidate)}
            >
              {candidate.replaceAll("_", " ")}
            </button>
          ))}
        </div>
      </div>
      <p id="roofline-description" className="text-xs leading-5 text-zinc-500">
        Each point is simulated model work. The diagonal is the selected bandwidth ceiling;
        {roofline.computeRoof === undefined
          ? " this dtype has no defensible compute ceiling, so none is invented."
          : ` the horizontal line is a ${roofline.computeRoof.evidence.replaceAll("_", " ")} ceiling for ${roofline.computeRoof.dtype}.`}
      </p>
      {roofline.computeRoof?.evidence === "vendor_peak"
        && roofline.computeRoof.sourceUrls?.[0] !== undefined ? (
          <a
            className="inline-flex text-xs font-semibold text-sky-700 hover:underline"
            href={roofline.computeRoof.sourceUrls[0]}
            target="_blank"
            rel="noreferrer"
          >
            View official compute specification
          </a>
        ) : null}
      <div className={`grid gap-3 border-l-2 px-3 py-3 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] ${interpretationTone}`} role="status">
        <div>
          <div className="text-[11px] font-bold uppercase text-current opacity-70">
            What this run suggests
          </div>
          <div className="mt-1 text-sm font-bold">{interpretation.verdict}</div>
          <p className="mt-1 text-xs leading-5">{interpretation.explanation}</p>
        </div>
        <div className="border-t border-current/15 pt-3 lg:border-l lg:border-t-0 lg:pl-3 lg:pt-0">
          <div className="text-[11px] font-bold uppercase text-current opacity-70">
            Next useful experiment
          </div>
          <p className="mt-1 text-xs leading-5">{interpretation.nextStep}</p>
        </div>
      </div>
      <div className="flex items-center justify-end gap-1" role="toolbar" aria-label="Roofline view controls">
        <Move className="mr-1 size-3.5 text-zinc-400" aria-hidden="true" />
        <button type="button" className="roofline-tool-button" aria-label="Zoom in" title="Zoom in" onClick={() => setZoom(0.7)}>
          <ZoomIn className="size-4" />
        </button>
        <button type="button" className="roofline-tool-button" aria-label="Zoom out" title="Zoom out" onClick={() => setZoom(1.4)}>
          <ZoomOut className="size-4" />
        </button>
        <button type="button" className="roofline-tool-button" aria-label="Reset view" title="Reset view" disabled={viewport === null} onClick={() => setViewport(null)}>
          <RotateCcw className="size-4" />
        </button>
      </div>
      <div
        ref={frame}
        className={`roofline-frame ${drag.current === null ? "cursor-grab" : "cursor-grabbing"}`}
        role="application"
        tabIndex={0}
        aria-describedby="roofline-description"
        aria-label="Interactive logarithmic hierarchical roofline chart"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerEnd}
        onPointerCancel={handlePointerEnd}
        onWheel={handleWheel}
        onKeyDown={handleKeyDown}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart margin={{ left: 12, right: 34, top: 18, bottom: 30 }}>
            <CartesianGrid stroke="#e4e4e7" />
            <XAxis
              type="number"
              dataKey="intensity"
              scale="log"
              domain={xDomain}
              allowDataOverflow
              tickFormatter={formatLogNumber}
              tick={{ fill: "#71717a", fontSize: 11 }}
              height={54}
              label={{ value: "Arithmetic intensity (FLOP/byte)", position: "insideBottom", offset: -12, fontSize: 11, fill: "#52525b" }}
            />
            <YAxis
              type="number"
              dataKey="rate"
              scale="log"
              domain={yDomain}
              allowDataOverflow
              tickFormatter={formatAxisFlops}
              tick={{ fill: "#71717a", fontSize: 11 }}
              width={84}
            />
            <ChartTooltip content={<RooflineTooltip roof={selectedRoof} />} />
            <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 11 }} />
            {interpretation.knee !== undefined
              ? (
                  <ReferenceLine
                    x={interpretation.knee}
                    stroke="#71717a"
                    strokeDasharray="2 4"
                    label={{ value: "knee", position: "insideTopRight", fontSize: 10, fill: "#71717a" }}
                  />
                )
              : null}
            <Line data={samples} dataKey="bandwidth" name={`${selectedRoof.label} bandwidth`} stroke="#d97706" strokeWidth={2} dot={false} isAnimationActive={false} />
            {roofline.computeRoof
              ? <Line data={samples} dataKey="compute" name={roofline.computeRoof.label} stroke="#047857" strokeWidth={2} strokeDasharray="6 4" dot={false} isAnimationActive={false} />
              : null}
            <Scatter data={data} dataKey="rate" name="Predicted work" fill="#0369a1" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] border-collapse text-left text-xs">
          <thead className="border-y border-zinc-200 text-zinc-500">
            <tr><th className="py-2">Work</th><th>Phase</th><th>Intensity</th><th>Predicted rate</th><th>Predicted tokens</th><th>Interpretation</th></tr>
          </thead>
          <tbody>
            {points.map((point) => (
              <tr key={point.id} className="border-b border-zinc-100">
                <td className="py-2 font-semibold">{point.label}</td>
                <td>{point.phase.replaceAll("_", " ")}</td>
                <td>{formatLogNumber(point.arithmeticIntensity)} FLOP/B</td>
                <td>{formatFlops(point.predictedFlopsPerSecond)}</td>
                <td>{point.predictedTokensPerSecond === undefined ? "n/a" : `${formatLogNumber(point.predictedTokensPerSecond)} tok/s`}</td>
                <td className="max-w-72 py-2 leading-4 text-zinc-600">
                  {interpretation.pointLabels[point.id] ?? "No interpretation available."}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <details className="border-t border-zinc-200 pt-2 text-xs text-zinc-600">
        <summary className="cursor-pointer font-semibold text-zinc-700">
          Terms and evidence limits
        </summary>
        <div className="mt-2 grid gap-2 leading-5 md:grid-cols-2">
          <p><strong>Intensity:</strong> useful arithmetic per byte moved. Farther right means more weight reuse or more computation per byte.</p>
          <p><strong>Knee:</strong> where the selected bandwidth roof meets effective compute. Left is bandwidth-sensitive; right is compute-sensitive.</p>
          <p><strong>Predicted:</strong> produced by the simulator replay. It is not measured hardware throughput.</p>
          <p><strong>Counterfactual tiers:</strong> PCIe, network, and SSD lines show what that tier could sustain if the modeled bytes crossed it.</p>
        </div>
      </details>
    </div>
  );
}

function RooflineTooltip({ active, payload, roof }: {
  readonly active?: boolean;
  readonly payload?: readonly { readonly payload?: Record<string, unknown> }[];
  readonly roof: NonNullable<DashboardResult["roofline"]>["bandwidthRoofs"][number];
}): React.JSX.Element | null {
  const item = payload?.find((entry) => entry.payload?.phase)?.payload;
  if (!active || item === undefined) return null;
  return (
    <div style={chartTooltipStyle} className="bg-white p-2.5">
      <div className="font-bold">{String(item.label)}</div>
      <div className="mt-1 text-zinc-600">{String(item.phase).replaceAll("_", " ")}</div>
      <div>{formatLogNumber(Number(item.arithmeticIntensity))} FLOP/byte</div>
      <div>{formatFlops(Number(item.predictedFlopsPerSecond))} predicted</div>
      <div className="mt-1 text-zinc-500">Compared with {roof.label}</div>
    </div>
  );
}

function logSamples(minimum: number, maximum: number, count: number) {
  const start = Math.log10(minimum);
  const span = Math.log10(maximum) - start;
  return Array.from({ length: count }, (_, index) => (
    10 ** (start + span * index / (count - 1))
  ));
}

function formatFlops(value: number): string {
  if (value >= 1e15) return `${(value / 1e15).toFixed(1)} PFLOP/s`;
  if (value >= 1e12) return `${(value / 1e12).toFixed(1)} TFLOP/s`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)} GFLOP/s`;
  return `${formatLogNumber(value)} FLOP/s`;
}

function formatAxisFlops(value: number): string {
  if (value >= 1e15) return `${formatAxisNumber(value / 1e15)}P`;
  if (value >= 1e12) return `${formatAxisNumber(value / 1e12)}T`;
  if (value >= 1e9) return `${formatAxisNumber(value / 1e9)}G`;
  if (value >= 1e6) return `${formatAxisNumber(value / 1e6)}M`;
  return formatAxisNumber(value);
}

function formatAxisNumber(value: number): string {
  return value >= 100 ? value.toFixed(0) : value.toFixed(1).replace(/\.0$/, "");
}

function formatLogNumber(value: number): string {
  if (value >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (value >= 10) return value.toFixed(0);
  if (value >= 1) return value.toFixed(1);
  return value.toPrecision(2);
}

function formatBytesPerSecond(value: number): string {
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} TB/s`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(0)} GB/s`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(0)} MB/s`;
  return `${formatLogNumber(value)} B/s`;
}

function TopologyComparisonChart({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const data = result.comparison?.map((entry) => ({
    scenario: entry.scenarioId.replaceAll("-", " "),
    duration: entry.totalDurationNs / 1_000_000,
    rank: entry.rank,
  })) ?? [];
  return (
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 14, right: 18 }}>
          <CartesianGrid stroke="#e4e4e7" horizontal={false} />
          <XAxis
            type="number"
            tickFormatter={(value: number) => `${Math.round(value)}ms`}
            tick={{ fill: "#71717a", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="scenario"
            width={112}
            tick={{ fill: "#52525b", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <ChartTooltip
            formatter={(value) => `${Number(value).toFixed(2)} ms`}
            contentStyle={chartTooltipStyle}
          />
          <Bar dataKey="duration" name="Duration" radius={[0, 3, 3, 0]}>
            {data.map((entry) => (
              <Cell
                key={entry.scenario}
                fill={entry.rank === 1 ? "#059669" : "#0369a1"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ServingLatencyChart({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const data = result.serving?.requests.map((request, index) => ({
    request: `R${index + 1}`,
    ttft: request.timeToFirstTokenNs / 1_000_000,
    latency: request.latencyNs / 1_000_000,
  })) ?? [];
  return (
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ left: 0, right: 8 }}>
          <CartesianGrid stroke="#e4e4e7" vertical={false} />
          <XAxis
            dataKey="request"
            interval={Math.max(0, Math.ceil(data.length / 8) - 1)}
            tick={{ fill: "#71717a", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tickFormatter={(value: number) => `${Math.round(value)}ms`}
            tick={{ fill: "#71717a", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <ChartTooltip
            formatter={(value) => `${Number(value).toFixed(2)} ms`}
            contentStyle={chartTooltipStyle}
          />
          <Bar
            dataKey="ttft"
            name="TTFT"
            fill="#d97706"
            radius={[3, 3, 0, 0]}
          />
          <Bar
            dataKey="latency"
            name="Completion"
            fill="#0369a1"
            radius={[3, 3, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ResourceChart({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const data = result.topology.topResources.map((resource) => ({
    name: shortResource(resource.resourceId),
    utilization: Math.round(resource.utilization * 1000) / 10,
    kind: resource.resourceId.startsWith("link:") ? "link" : "compute",
  }));
  return (
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ left: 8, right: 14 }}
        >
          <CartesianGrid stroke="#e4e4e7" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, 100]}
            tickFormatter={(value: number) => `${value}%`}
            tick={{ fill: "#71717a", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={104}
            tick={{ fill: "#52525b", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <ChartTooltip
            formatter={(value) => `${Number(value).toFixed(1)}%`}
            contentStyle={chartTooltipStyle}
          />
          <Bar dataKey="utilization" name="Utilization" radius={[0, 3, 3, 0]}>
            {data.map((entry) => (
              <Cell
                key={entry.name}
                fill={entry.kind === "link" ? "#d97706" : "#0369a1"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * What each allocation purpose means to someone tuning a run, ordered by how
 * directly they control it. Free space is last so the used extent reads as one
 * contiguous block from the axis.
 */
const MEMORY_PURPOSES = [
  { key: "weights", label: "Weights", fill: "#0369a1" },
  { key: "kv", label: "KV cache", fill: "#0891b2" },
  { key: "cache", label: "Expert cache", fill: "#7c3aed" },
  { key: "backing", label: "Expert backing", fill: "#a855f7" },
  { key: "workspace", label: "Workspace", fill: "#f59e0b" },
  { key: "staging", label: "Staging", fill: "#f97316" },
  { key: "checkpoint", label: "Checkpoint", fill: "#65a30d" },
  { key: "sidecar", label: "Sidecar", fill: "#84cc16" },
] as const;

function formatGiB(bytes: number): string {
  const giB = bytes / 1024 ** 3;
  if (giB >= 100) return `${giB.toFixed(0)} GiB`;
  if (giB >= 1) return `${giB.toFixed(1)} GiB`;
  const miB = bytes / 1024 ** 2;
  return miB >= 1 ? `${miB.toFixed(0)} MiB` : `${(bytes / 1024).toFixed(0)} KiB`;
}

/**
 * Where each model's time went. A model that had to be swapped shows its
 * requests waiting for the device to load it again, which is the cost the
 * working set not fitting actually imposes.
 */
function CoResidencyChart({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const metrics = result.coResidency!.metrics;
  const data = metrics.tenants.map((tenant) => ({
    name: tenant.displayName,
    resident: tenant.residentNs / 1e6,
    waiting: tenant.residencyWaitNs / 1e6,
    loading: tenant.loadServiceNs / 1e6,
    footprintBytes: tenant.residencyFootprintBytes,
    loads: tenant.loads,
    evictions: tenant.evictions,
  }));

  return (
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 18 }}>
          <CartesianGrid stroke="#e4e4e7" horizontal={false} />
          <XAxis
            type="number"
            tickFormatter={(value: number) => `${Math.round(value)} ms`}
            tick={{ fill: "#71717a", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={132}
            tick={{ fill: "#52525b", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <ChartTooltip
            contentStyle={chartTooltipStyle}
            formatter={(value, name) => [
              `${Number(value).toFixed(1)} ms`,
              String(name),
            ]}
          />
          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 4 }} iconSize={8} />
          <Bar dataKey="resident" name="Resident" fill="#0369a1" />
          <Bar dataKey="loading" name="Being loaded" fill="#f59e0b" />
          <Bar dataKey="waiting" name="Requests waiting" fill="#dc2626" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

const FAULT_STATUS_FILL = {
  failed: "#dc2626",
  aborted: "#f59e0b",
  succeeded: "#16a34a",
} as const;

function formatDurationMs(ns: number): string {
  if (ns >= 1_000_000) return `${(ns / 1_000_000).toFixed(2)} ms`;
  if (ns >= 1_000) return `${(ns / 1_000).toFixed(1)} us`;
  return `${ns} ns`;
}

/**
 * When each rank stopped, against the fault instant and the abort deadline.
 * Ranks on the failed node stop at the fault; survivors either drain their own
 * work or are aborted when the coordinator closes the epoch.
 */
function FaultRankChart({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const fault = result.fault!;
  const data = fault.rankStates.map((state) => ({
    name: `${state.rankId}${state.onFailedNode ? " (failed node)" : ""}`,
    terminalAtNs: state.terminalAtNs,
    status: state.status,
    nodeId: state.nodeId,
  }));
  const upperNs = Math.max(
    fault.abortDeadlineNs,
    fault.quiescedAtNs,
    ...data.map((row) => row.terminalAtNs),
  ) * 1.05;

  return (
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 18 }}>
          <CartesianGrid stroke="#e4e4e7" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, upperNs]}
            tickFormatter={(value: number) => formatDurationMs(value)}
            tick={{ fill: "#71717a", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={132}
            tick={{ fill: "#52525b", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <ChartTooltip
            contentStyle={chartTooltipStyle}
            formatter={(value, _name, entry) => {
              const row = entry?.payload as {
                readonly status?: string;
                readonly nodeId?: string;
              } | undefined;
              return [
                `${formatDurationMs(Number(value))} · ${row?.status ?? ""}${
                  row?.nodeId ? ` · ${row.nodeId}` : ""
                }`,
                "Terminal",
              ];
            }}
          />
          <ReferenceLine
            x={fault.faultAtNs}
            stroke="#dc2626"
            strokeDasharray="4 3"
            label={{
              value: "fault",
              position: "top",
              fill: "#dc2626",
              fontSize: 11,
            }}
          />
          <ReferenceLine
            x={fault.abortDeadlineNs}
            stroke="#f59e0b"
            strokeDasharray="4 3"
            label={{
              value: "abort deadline",
              position: "top",
              fill: "#b45309",
              fontSize: 11,
            }}
          />
          <Bar dataKey="terminalAtNs" name="Terminal">
            {data.map((row) => (
              <Cell
                key={row.name}
                fill={FAULT_STATUS_FILL[row.status]}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}


function MemoryTimelineSection({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element | null {
  // Sweeping every request's tokens is cheap but not free at the maximum
  // request and output counts, and this rerenders on any prop change.
  const timeline = useMemo(() => memoryTimeline(result), [result]);
  if (timeline === undefined || timeline.samples.length < 2) {
    return null;
  }
  const palette = new Map<string, { key: string; label: string; fill: string }>(
    MEMORY_PURPOSES.map((purpose) => [purpose.key, { ...purpose }] as const),
  );
  const bands = timeline.purposes.map((purpose) => (
    palette.get(purpose) ?? { key: purpose, label: purpose, fill: "#94a3b8" }
  ));
  const data = timeline.samples.map((sample) => ({
    ms: sample.atNs / 1e6,
    ...sample.bytes,
    total: sample.total,
  }));
  // Headroom above the peak rather than the whole domain: on a machine with
  // far more memory than the run uses, scaling to capacity would flatten the
  // series into a line along the axis and show nothing.
  const ceiling = Math.min(
    timeline.capacityBytes,
    Math.max(timeline.peakTotalBytes * 1.25, timeline.residentBytes * 1.05),
  );
  const peakShare = timeline.peakTotalBytes / timeline.capacityBytes * 100;
  return (
    <section className="panel">
      <SectionHeading
        title="Memory over time"
        detail={`${shortDomain(timeline.domainId)}${
          timeline.weightDomainCount > 1
            ? ` · 1 of ${timeline.weightDomainCount} weight shards`
            : ""
        } · peak ${
          formatGiB(timeline.peakTotalBytes)
        } of ${formatGiB(timeline.capacityBytes)} (${
          peakShare.toFixed(1)
        }%)`}
      />
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 8, right: 16, bottom: 4, left: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
            <XAxis
              dataKey="ms"
              type="number"
              domain={[0, "dataMax"]}
              tick={{ fontSize: 11 }}
              tickFormatter={(value: number) => `${value.toFixed(0)} ms`}
            />
            <YAxis
              tick={{ fontSize: 11 }}
              domain={[0, ceiling]}
              tickFormatter={(value: number) => formatGiB(value)}
            />
            <ChartTooltip
              labelFormatter={(value) => `${Number(value).toFixed(2)} ms`}
              formatter={(value) => formatGiB(Number(value))}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {/* Stacked, so the top edge is total occupancy and the bands
                below say what the memory is being held for. */}
            {bands.map((band) => (
              <Area
                key={band.key}
                type="stepAfter"
                dataKey={band.key}
                stackId="memory"
                name={band.label}
                stroke={band.fill}
                fill={band.fill}
                fillOpacity={0.8}
                isAnimationActive={false}
              />
            ))}
            {timeline.capacityBytes <= ceiling
              ? (
                  <ReferenceLine
                    y={timeline.capacityBytes}
                    stroke="#dc2626"
                    strokeDasharray="4 4"
                    label={{
                      value: "allocatable",
                      position: "insideTopRight",
                      fontSize: 10,
                      fill: "#dc2626",
                    }}
                  />
                )
              : null}
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-[11px] leading-4 text-zinc-500">
        Weights and caches do not move once loaded, so KV is the only term that
        varies: it grows while a request generates and is released when the
        request retires. {timeline.caveat}
        {timeline.weightDomainCount > 1
          ? " Weights are sharded, and this is one shard: another may sit"
            + " closer to its limit."
          : ""}
      </p>
    </section>
  );
}

function MemoryChart({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const entries = result.scenario.memoryLedger.filter((entry) => entry.enabled);
  const usedPurposes = MEMORY_PURPOSES.filter((purpose) => (
    entries.some((entry) => (entry.reservedByPurpose[purpose.key] ?? 0) > 0)
  ));
  // Domains differ by orders of magnitude, so a shared byte axis would hide
  // an 80 GiB device next to a 2 TiB SSD. Each bar is normalized to its own
  // allocatable extent and every number the user reads is in bytes.
  const data = entries.map((entry) => {
    const limit = Math.max(1, entry.capacityBytes);
    const bytes = Object.fromEntries(usedPurposes.map((purpose) => [
      purpose.key,
      entry.reservedByPurpose[purpose.key] ?? 0,
    ]));
    return {
      name: `${shortDomain(entry.domainId)} · ${formatGiB(entry.capacityBytes)}`,
      capacityBytes: entry.capacityBytes,
      physicalCapacityBytes: entry.physicalCapacityBytes,
      reservedBytes: entry.reservedBytes,
      freeBytes: entry.freeBytes,
      bytes,
      free: entry.freeBytes / limit * 100,
      ...Object.fromEntries(usedPurposes.map((purpose) => [
        purpose.key,
        (bytes[purpose.key] ?? 0) / limit * 100,
      ])),
    };
  });

  return (
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 14 }}>
          <CartesianGrid stroke="#e4e4e7" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, 100]}
            tickFormatter={(value: number) => `${Math.round(value)}%`}
            tick={{ fill: "#71717a", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={132}
            tick={{ fill: "#52525b", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <ChartTooltip
            content={<MemoryTooltip purposes={usedPurposes} />}
            cursor={{ fill: "rgba(24,24,27,0.04)" }}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, paddingTop: 4 }}
            iconSize={8}
          />
          {usedPurposes.map((purpose) => (
            <Bar
              key={purpose.key}
              dataKey={purpose.key}
              name={purpose.label}
              stackId="memory"
              fill={purpose.fill}
            />
          ))}
          <Bar
            dataKey="free"
            name="Allocatable free"
            stackId="memory"
            fill="#e4e4e7"
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

interface MemoryRow {
  readonly name: string;
  readonly capacityBytes: number;
  readonly physicalCapacityBytes: number;
  readonly reservedBytes: number;
  readonly freeBytes: number;
  readonly bytes: Record<string, number>;
}

function MemoryTooltip({
  active,
  payload,
  purposes,
}: {
  readonly active?: boolean;
  readonly payload?: readonly { readonly payload?: MemoryRow }[];
  readonly purposes: readonly typeof MEMORY_PURPOSES[number][];
}): React.JSX.Element | null {
  const row = payload?.[0]?.payload;
  if (!active || row === undefined) {
    return null;
  }
  const limit = row.capacityBytes;
  const physical = row.physicalCapacityBytes;
  const share = (bytes: number) => (
    limit > 0 ? `${(bytes / limit * 100).toFixed(1)}%` : "n/a"
  );
  return (
    <div style={chartTooltipStyle} className="bg-white p-2.5">
      <div className="mb-1 font-semibold">{row.name}</div>
      <div className="space-y-0.5">
        {purposes
          .filter((purpose) => (row.bytes[purpose.key] ?? 0) > 0)
          .map((purpose) => (
            <div key={purpose.key} className="flex justify-between gap-4">
              <span style={{ color: purpose.fill }}>{purpose.label}</span>
              <span>
                {formatGiB(row.bytes[purpose.key]!)}
                {" · "}
                {share(row.bytes[purpose.key]!)}
              </span>
            </div>
          ))}
        <div className="flex justify-between gap-4 font-medium">
          <span>Reserved</span>
          <span>{formatGiB(row.reservedBytes)} · {share(row.reservedBytes)}</span>
        </div>
        <div className="flex justify-between gap-4 text-zinc-500">
          <span>Allocatable free</span>
          <span>{formatGiB(row.freeBytes)} · {share(row.freeBytes)}</span>
        </div>
      </div>
      <div className="mt-1.5 border-t border-zinc-200 pt-1.5 text-[11px] text-zinc-500">
        <div>Resource limit {formatGiB(limit)}</div>
        {physical > limit
          ? <div>Physical {formatGiB(physical)} · {formatGiB(physical - limit)} withheld</div>
          : <div>Physical {formatGiB(physical)}</div>}
      </div>
    </div>
  );
}

function AcceptanceChart({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const data = result.speculative?.metrics.acceptanceByPosition.map(
    (acceptance, index) => ({
      position: `P${index + 1}`,
      acceptance: Math.round(acceptance * 1000) / 10,
    }),
  ) ?? [];
  return (
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ left: 0, right: 8 }}>
          <CartesianGrid stroke="#e4e4e7" vertical={false} />
          <XAxis
            dataKey="position"
            tick={{ fill: "#71717a", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tickFormatter={(value: number) => `${value}%`}
            tick={{ fill: "#71717a", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <ChartTooltip
            formatter={(value) => `${Number(value).toFixed(1)}%`}
            contentStyle={chartTooltipStyle}
          />
          <Bar
            dataKey="acceptance"
            name="Acceptance"
            fill="#059669"
            radius={[3, 3, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function CacheOutcomeChart({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const metrics = result.expertCache?.metrics;
  const data = [
    { name: "Hot", value: metrics?.hotHits ?? 0, color: "#0369a1" },
    { name: "Warm", value: metrics?.warmMisses ?? 0, color: "#d97706" },
    { name: "Cold", value: metrics?.coldMisses ?? 0, color: "#be123c" },
  ];
  return (
    <div className="cache-chart chart-frame grid items-center md:grid-cols-[minmax(0,1fr)_180px]">
      <div className="h-44 min-w-0 md:h-full">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              innerRadius={58}
              outerRadius={76}
              paddingAngle={2}
            >
              {data.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
            </Pie>
            <ChartTooltip contentStyle={chartTooltipStyle} />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="space-y-3 pr-4">
        {data.map((entry) => (
          <div
            key={entry.name}
            className="flex items-center justify-between gap-4 text-sm"
          >
            <span className="flex items-center gap-2 text-zinc-600">
              <span
                className="size-2.5 rounded-sm"
                style={{ background: entry.color }}
              />
              {entry.name}
            </span>
            <strong className="tabular-nums">{entry.value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function CachePartitionChart({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const data = cachePartitionRows(result.expertCache);
  return (
    <div className="partition-chart chart-frame flex flex-col">
      <div className="min-h-0 flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ left: 8, right: 14 }}
          >
            <CartesianGrid stroke="#e4e4e7" horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 100]}
              tickFormatter={(value: number) => `${Math.round(value)}%`}
              tick={{ fill: "#71717a", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={112}
              tick={{ fill: "#52525b", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <ChartTooltip
              formatter={(value, name, item) => [
                `${Number(value).toFixed(1)}%`,
                `${String(name)} · ${Number(item.payload.capacityMiB).toFixed(0)} MiB`,
              ]}
              contentStyle={chartTooltipStyle}
            />
            <Bar
              dataKey="resident"
              name="Resident"
              stackId="capacity"
            >
              {data.map((entry) => (
                <Cell
                  key={entry.name}
                  fill={entry.tier === "hot" ? "#0369a1" : "#d97706"}
                />
              ))}
            </Bar>
            <Bar
              dataKey="reserved"
              name="Reserved"
              stackId="capacity"
              fill="#be123c"
            />
            <Bar
              dataKey="free"
              name="Free"
              stackId="capacity"
              fill="#d4d4d8"
              radius={[0, 3, 3, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 px-3 pb-2 text-[11px] text-zinc-600">
        {[
          ["Hot resident", "#0369a1"],
          ["Warm resident", "#d97706"],
          ["Reserved", "#be123c"],
          ["Free", "#d4d4d8"],
        ].map(([label, color]) => (
          <span key={label} className="flex items-center gap-1.5">
            <span
              className="size-2.5 shrink-0 rounded-sm"
              style={{ backgroundColor: color }}
            />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

function CachePartitionPressure({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element | null {
  const cache = result.expertCache;
  if (cache === undefined || cache.metrics.evictions === 0) {
    return null;
  }
  const pressured = cachePartitionRows(cache).filter((partition) => (
    partition.resident + partition.reserved >= 90
  ));
  if (pressured.length === 0) {
    return null;
  }
  return (
    <div
      role="status"
      className="mb-3 flex flex-wrap items-center justify-between gap-2 border-l-2 border-amber-600 bg-amber-50 px-3 py-2 text-xs"
    >
      <span className="font-semibold text-amber-900">
        Capacity pressure · {cache.metrics.evictions.toLocaleString()} evictions
      </span>
      <span className="text-amber-800">
        {pressured.map((partition) => (
          `${partition.name} ${Math.round(
            partition.resident + partition.reserved,
          )}%`
        )).join(" · ")}
      </span>
    </div>
  );
}

export function cachePartitionRows(
  cache: DashboardResult["expertCache"],
) {
  if (cache === undefined) {
    return [];
  }
  return [
    ...cache.hotPartitions.map((partition) => ({
      ...partition,
      tier: "hot" as const,
    })),
    ...cache.warmPartitions.map((partition) => ({
      ...partition,
      tier: "warm" as const,
    })),
  ].filter((partition) => partition.capacityBytes > 0)
    .map((partition) => ({
      name: `${partition.tier === "hot" ? "H" : "W"} ${shortPartition(partition.id)}`,
      tier: partition.tier,
      resident: partition.residentBytes / partition.capacityBytes * 100,
      reserved: partition.reservedBytes / partition.capacityBytes * 100,
      free: Math.max(
        0,
        (
          partition.capacityBytes
          - partition.residentBytes
          - partition.reservedBytes
        ) / partition.capacityBytes * 100,
      ),
      capacityMiB: partition.capacityBytes / 1024 ** 2,
    }));
}

function SectionHeading({
  title,
  detail,
}: {
  readonly title: string;
  readonly detail: string;
}): React.JSX.Element {
  return (
    <div className="mb-3 flex items-center justify-between gap-4">
      <h2 className="text-sm font-bold">{title}</h2>
      <span className="truncate text-xs text-zinc-500">{detail}</span>
    </div>
  );
}

function shortPartition(id: string): string {
  return id
    .replace("target-shard-", "owner ")
    .replace("node", "node ");
}

const chartTooltipStyle = {
  border: "1px solid #e4e4e7",
  borderRadius: 6,
  color: "#18181b",
  fontSize: 12,
  boxShadow: "0 8px 24px rgba(24,24,27,0.08)",
};

function shortDomain(id: string): string {
  return id
    .replace(/^node(\d+):/, "n$1/")
    .replaceAll("-", " ");
}

function shortResource(id: string): string {
  return id
    .replace(/^compute:/, "")
    .replace(/^link:/, "")
    .replace(/^network:/, "")
    .replace(/^node(\d+):/, "n$1/")
    .replace(/node(\d+)/g, "n$1")
    .replaceAll("-", " ");
}
