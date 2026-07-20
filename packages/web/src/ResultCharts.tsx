import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DashboardResult } from "./types.js";

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
      <section className="panel">
        <SectionHeading
          title="Memory domains"
          detail={`${result.scenario.deviceCount} devices · ${result.scenario.linkCount} links`}
        />
        <MemoryChart result={result} />
      </section>
      <section className="panel">
        <SectionHeading
          title="Resource utilization"
          detail={`${result.topology.planSteps.toLocaleString()} replay-verified steps`}
        />
        <ResourceChart result={result} />
      </section>
      <section className="panel">
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
      </section>
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

function MemoryChart({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const data = result.scenario.memoryLedger.map((entry) => ({
    name: shortDomain(entry.domainId),
    used: entry.reservedBytes / entry.capacityBytes * 100,
    free: entry.freeBytes / entry.capacityBytes * 100,
  }));
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
            width={92}
            tick={{ fill: "#52525b", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <ChartTooltip
            formatter={(value) => `${Number(value).toFixed(1)}%`}
            contentStyle={chartTooltipStyle}
          />
          <Bar dataKey="used" name="Reserved" stackId="memory" fill="#0369a1" />
          <Bar dataKey="free" name="Free" stackId="memory" fill="#d4d4d8" />
        </BarChart>
      </ResponsiveContainer>
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
  return id.replace(/^node\d+:/, "").replaceAll("-", " ");
}

function shortResource(id: string): string {
  return id
    .replace(/^compute:/, "")
    .replace(/^link:/, "")
    .replace(/^node\d+:/, "")
    .replaceAll("-", " ");
}
