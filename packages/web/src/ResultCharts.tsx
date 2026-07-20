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
      <section className="panel">
        <SectionHeading
          title="Memory domains"
          detail={`${result.scenario.deviceCount} devices · ${result.scenario.linkCount} links`}
        />
        <MemoryChart result={result} />
      </section>
      <section className="panel">
        <SectionHeading
          title={result.mode === "speculative"
            ? "Acceptance profile"
            : "Cache outcomes"}
          detail={result.mode === "speculative"
            ? "Conditional prefix positions"
            : "Routed expert accesses"}
        />
        {result.mode === "speculative"
          ? <AcceptanceChart result={result} />
          : <CacheOutcomeChart result={result} />}
      </section>
    </>
  );
}

function MemoryChart({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const data = result.scenario.memoryLedger.map((entry) => ({
    name: shortDomain(entry.domainId),
    used: entry.reservedBytes / 1024 ** 3,
    free: entry.freeBytes / 1024 ** 3,
  }));
  return (
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 14 }}>
          <CartesianGrid stroke="#e4e4e7" horizontal={false} />
          <XAxis
            type="number"
            tickFormatter={(value: number) => `${Math.round(value)}G`}
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
            formatter={(value) => `${Number(value).toFixed(1)} GiB`}
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
