import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  CheckCircle2,
  Clock3,
  Cpu,
  Database,
  Gauge,
  MemoryStick,
  Network,
  Play,
  RotateCcw,
  Square,
  TriangleAlert,
} from "lucide-react";
import { Badge } from "./components/ui/badge.js";
import { Button } from "./components/ui/button.js";
import { Progress } from "./components/ui/progress.js";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select.js";
import { Slider } from "./components/ui/slider.js";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "./components/ui/tabs.js";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./components/ui/tooltip.js";
import type {
  DashboardResult,
  DashboardRunConfig,
  WorkerResponse,
  WorkloadMode,
} from "./types.js";

const SCENARIOS: ReadonlyArray<{
  readonly value: DashboardRunConfig["scenarioName"];
  readonly label: string;
}> = [
  { value: "cpu-only", label: "CPU only" },
  { value: "single-gpu-cpu", label: "Discrete GPU + CPU" },
  { value: "multi-gpu", label: "Multi-GPU" },
  { value: "gpu-npu", label: "GPU + NPU" },
  { value: "unified-memory", label: "Unified memory" },
  { value: "multi-node", label: "Multi-node" },
];

const DEFAULT_CONFIG: DashboardRunConfig = {
  scenarioName: "multi-gpu",
  mode: "speculative",
  seed: 42,
  speculative: {
    outputTokens: 128,
    draftWidth: 4,
    firstPositionAcceptance: 0.82,
  },
  expertCache: {
    tokenCount: 96,
    topK: 2,
    expertCount: 16,
    hotSlots: 6,
    warmSlots: 8,
  },
};

const ResultCharts = lazy(() => import("./ResultCharts.js"));

type RunStatus = "idle" | "running" | "complete" | "cancelled" | "error";

interface RunState {
  readonly status: RunStatus;
  readonly progress: number;
  readonly phase: string;
  readonly result?: DashboardResult;
  readonly error?: string;
}

export function App(): React.JSX.Element {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [runState, setRunState] = useState<RunState>({
    status: "idle",
    progress: 0,
    phase: "Ready",
  });
  const workerRef = useRef<Worker | undefined>(undefined);
  const runIdRef = useRef(0);
  const initializedRef = useRef(false);
  const changeConfig = useCallback((nextConfig: DashboardRunConfig) => {
    setConfig(nextConfig);
    setRunState((current) => current.status === "running"
      ? current
      : {
          status: "idle",
          progress: 0,
          phase: "Configuration changed",
        });
  }, []);

  const cancel = useCallback(() => {
    workerRef.current?.terminate();
    workerRef.current = undefined;
    setRunState((current) => ({
      ...current,
      status: "cancelled",
      phase: "Cancelled",
    }));
  }, []);

  const run = useCallback((nextConfig: DashboardRunConfig) => {
    workerRef.current?.terminate();
    const worker = new Worker(new URL("./sim-worker.ts", import.meta.url), {
      type: "module",
    });
    const runId = ++runIdRef.current;
    workerRef.current = worker;
    setRunState({
      status: "running",
      progress: 4,
      phase: "Starting worker",
    });
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.runId !== runIdRef.current) {
        return;
      }
      if (message.type === "progress") {
        setRunState((current) => ({
          ...current,
          status: "running",
          progress: message.progress,
          phase: message.phase,
        }));
        return;
      }
      if (message.type === "error") {
        setRunState({
          status: "error",
          progress: 100,
          phase: "Failed",
          error: message.message,
        });
        worker.terminate();
        workerRef.current = undefined;
        return;
      }
      setRunState({
        status: "complete",
        progress: 100,
        phase: "Replay verified",
        result: message.result,
      });
      worker.terminate();
      workerRef.current = undefined;
    };
    worker.onerror = (event) => {
      setRunState({
        status: "error",
        progress: 100,
        phase: "Worker failed",
        error: event.message,
      });
      worker.terminate();
      workerRef.current = undefined;
    };
    worker.postMessage({ type: "run", runId, config: nextConfig });
  }, []);

  useEffect(() => {
    if (!initializedRef.current) {
      initializedRef.current = true;
      run(DEFAULT_CONFIG);
    }
    return () => workerRef.current?.terminate();
  }, [run]);

  const result = runState.result;
  return (
    <TooltipProvider>
      <div className="min-h-screen bg-[#f7f8fa] text-zinc-950">
        <header className="flex h-14 items-center justify-between border-b border-zinc-200 bg-white px-4 sm:px-5">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid size-8 shrink-0 place-items-center rounded-md bg-zinc-950 text-white">
              <Gauge className="size-4" />
            </div>
            <div className="min-w-0">
              <div className="truncate text-sm font-bold">Inference Sim</div>
              <div className="truncate text-xs text-zinc-500">
                Deterministic runtime model
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <RunBadge status={runState.status} />
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Reset configuration"
                  onClick={() => changeConfig(DEFAULT_CONFIG)}
                  disabled={runState.status === "running"}
                >
                  <RotateCcw className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Reset configuration</TooltipContent>
            </Tooltip>
          </div>
        </header>

        <div className="workspace-grid">
          <aside className="border-b border-zinc-200 bg-white p-4 lg:border-b-0 lg:border-r">
            <ConfigurationPanel
              config={config}
              disabled={runState.status === "running"}
              onChange={changeConfig}
              onRun={() => run(config)}
              onCancel={cancel}
              running={runState.status === "running"}
            />
          </aside>

          <main className="min-w-0 overflow-y-auto p-4 sm:p-5">
            <RunProgress state={runState} />
            {result
              ? <Results result={result} />
              : <EmptyState state={runState} />}
          </main>

          <aside className="min-w-0 border-t border-zinc-200 bg-white p-4 xl:border-l xl:border-t-0">
            <Inspector result={result} />
          </aside>
        </div>
      </div>
    </TooltipProvider>
  );
}

function ConfigurationPanel({
  config,
  disabled,
  onChange,
  onRun,
  onCancel,
  running,
}: {
  readonly config: DashboardRunConfig;
  readonly disabled: boolean;
  readonly onChange: (config: DashboardRunConfig) => void;
  readonly onRun: () => void;
  readonly onCancel: () => void;
  readonly running: boolean;
}): React.JSX.Element {
  const setMode = (mode: WorkloadMode) => onChange({ ...config, mode });
  return (
    <div className="mx-auto max-w-md lg:max-w-none">
      <div className="mb-4">
        <h2 className="text-sm font-bold">Run configuration</h2>
        <p className="mt-0.5 text-xs text-zinc-500">Seed {config.seed}</p>
      </div>

      <Field label="Device topology">
        <Select
          value={config.scenarioName}
          disabled={disabled}
          onValueChange={(scenarioName) => onChange({
            ...config,
            scenarioName: scenarioName as DashboardRunConfig["scenarioName"],
          })}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SCENARIOS.map((scenario) => (
              <SelectItem key={scenario.value} value={scenario.value}>
                {scenario.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>

      <Tabs
        value={config.mode}
        onValueChange={(mode) => setMode(mode as WorkloadMode)}
      >
        <TabsList className="mb-4 w-full">
          <TabsTrigger value="speculative">Speculative</TabsTrigger>
          <TabsTrigger value="expert-cache">Expert cache</TabsTrigger>
        </TabsList>
        <TabsContent value="speculative" className="space-y-5">
          <SliderField
            label="Output tokens"
            value={config.speculative.outputTokens}
            minimum={32}
            maximum={512}
            step={16}
            disabled={disabled}
            onChange={(outputTokens) => onChange({
              ...config,
              speculative: { ...config.speculative, outputTokens },
            })}
          />
          <SliderField
            label="Draft width"
            value={config.speculative.draftWidth}
            minimum={1}
            maximum={8}
            step={1}
            disabled={disabled}
            onChange={(draftWidth) => onChange({
              ...config,
              speculative: { ...config.speculative, draftWidth },
            })}
          />
          <SliderField
            label="Position 1 acceptance"
            value={Math.round(
              config.speculative.firstPositionAcceptance * 100,
            )}
            suffix="%"
            minimum={20}
            maximum={98}
            step={1}
            disabled={disabled}
            onChange={(acceptance) => onChange({
              ...config,
              speculative: {
                ...config.speculative,
                firstPositionAcceptance: acceptance / 100,
              },
            })}
          />
        </TabsContent>
        <TabsContent value="expert-cache" className="space-y-5">
          <SliderField
            label="Token routes"
            value={config.expertCache.tokenCount}
            minimum={16}
            maximum={512}
            step={16}
            disabled={disabled}
            onChange={(tokenCount) => onChange({
              ...config,
              expertCache: { ...config.expertCache, tokenCount },
            })}
          />
          <SliderField
            label="Experts per token"
            value={config.expertCache.topK}
            minimum={1}
            maximum={4}
            step={1}
            disabled={disabled}
            onChange={(topK) => onChange({
              ...config,
              expertCache: {
                ...config.expertCache,
                topK,
                hotSlots: Math.max(topK, config.expertCache.hotSlots),
              },
            })}
          />
          <SliderField
            label="Hot cache slots"
            value={config.expertCache.hotSlots}
            minimum={config.expertCache.topK}
            maximum={config.expertCache.expertCount}
            step={1}
            disabled={disabled}
            onChange={(hotSlots) => onChange({
              ...config,
              expertCache: { ...config.expertCache, hotSlots },
            })}
          />
          <SliderField
            label="Warm cache slots"
            value={config.expertCache.warmSlots}
            minimum={0}
            maximum={config.expertCache.expertCount}
            step={1}
            disabled={disabled}
            onChange={(warmSlots) => onChange({
              ...config,
              expertCache: { ...config.expertCache, warmSlots },
            })}
          />
        </TabsContent>
      </Tabs>

      <div className="mt-6 flex gap-2 border-t border-zinc-200 pt-4">
        {running
          ? (
            <Button className="w-full" variant="destructive" onClick={onCancel}>
              <Square className="size-4 fill-current" />
              Cancel
            </Button>
          )
          : (
            <Button className="w-full" onClick={onRun}>
              <Play className="size-4 fill-current" />
              Run simulation
            </Button>
          )}
      </div>
    </div>
  );
}

function Results({ result }: { readonly result: DashboardResult }): React.JSX.Element {
  const metrics = result.mode === "speculative"
    ? speculativeMetrics(result)
    : expertMetrics(result);
  return (
    <div className="space-y-5">
      <section className="metric-grid" aria-label="Run metrics">
        {metrics.map((metric) => (
          <div key={metric.label} className="metric-card">
            <div className="flex items-center justify-between gap-2 text-zinc-500">
              <span className="text-xs font-semibold">{metric.label}</span>
              {metric.icon}
            </div>
            <div className="mt-2 text-xl font-bold tabular-nums">
              {metric.value}
            </div>
            <div className="mt-1 truncate text-xs text-zinc-500">
              {metric.detail}
            </div>
          </div>
        ))}
      </section>

      <Suspense fallback={<ChartSkeleton />}>
        <ResultCharts result={result} />
      </Suspense>
    </div>
  );
}

function Inspector({ result }: { readonly result?: DashboardResult }): React.JSX.Element {
  const rows = useMemo(() => {
    if (result?.speculative) {
      return result.speculative.iterations.slice(-12).reverse().map((iteration) => ({
        id: `Iteration ${iteration.iteration + 1}`,
        primary: `${iteration.acceptedDraftTokens}/${iteration.proposedDraftTokens} accepted`,
        secondary: iteration.outcome.replace("_", " "),
        value: `+${iteration.committedTokens}`,
      }));
    }
    if (result?.expertCache) {
      return result.expertCache.routes.slice(-12).reverse().map((route, index) => ({
        id: `Route ${result.expertCache!.routes.length - index}`,
        primary: route.expertIds.join(", "),
        secondary: route.sourceTiers.join(" · "),
        value: formatDuration(route.stallNs),
      }));
    }
    return [];
  }, [result]);
  return (
    <div className="mx-auto max-w-md xl:max-w-none">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-bold">Run inspector</h2>
        <Badge variant="neutral">{rows.length} recent</Badge>
      </div>
      {rows.length === 0
        ? (
          <div className="grid min-h-40 place-items-center border-y border-zinc-200 text-sm text-zinc-500">
            No events
          </div>
        )
        : (
          <div className="divide-y divide-zinc-200 border-y border-zinc-200">
            {rows.map((row) => (
              <div key={row.id} className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 py-3">
                <div className="min-w-0">
                  <div className="text-xs font-bold text-zinc-800">{row.id}</div>
                  <div className="mt-1 truncate text-xs text-zinc-600">{row.primary}</div>
                  <div className="mt-0.5 truncate text-xs capitalize text-zinc-400">
                    {row.secondary}
                  </div>
                </div>
                <div className="self-center text-xs font-bold tabular-nums text-zinc-700">
                  {row.value}
                </div>
              </div>
            ))}
          </div>
        )}
    </div>
  );
}

function RunProgress({ state }: { readonly state: RunState }): React.JSX.Element {
  if (state.status !== "running") {
    return <div className="h-0" />;
  }
  return (
    <div className="mb-4 border border-sky-200 bg-sky-50 p-3">
      <div className="mb-2 flex items-center justify-between gap-3 text-xs">
        <span className="font-semibold text-sky-900">{state.phase}</span>
        <span className="tabular-nums text-sky-700">{state.progress}%</span>
      </div>
      <Progress value={state.progress} />
    </div>
  );
}

function EmptyState({ state }: { readonly state: RunState }): React.JSX.Element {
  const failed = state.status === "error";
  return (
    <div className="grid min-h-[55vh] place-items-center border border-dashed border-zinc-300 bg-white p-8 text-center">
      <div>
        {failed
          ? <TriangleAlert className="mx-auto size-7 text-rose-700" />
          : <Clock3 className="mx-auto size-7 text-zinc-400" />}
        <h2 className="mt-3 text-sm font-bold">
          {failed ? "Simulation failed" : state.phase}
        </h2>
        <p className="mt-1 max-w-md text-xs text-zinc-500">
          {state.error ?? "Run results will appear here."}
        </p>
      </div>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  readonly label: string;
  readonly children: React.ReactNode;
}): React.JSX.Element {
  return (
    <label className="mb-4 block">
      <span className="mb-1.5 block text-xs font-semibold text-zinc-600">
        {label}
      </span>
      {children}
    </label>
  );
}

function SliderField({
  label,
  value,
  suffix = "",
  minimum,
  maximum,
  step,
  disabled,
  onChange,
}: {
  readonly label: string;
  readonly value: number;
  readonly suffix?: string;
  readonly minimum: number;
  readonly maximum: number;
  readonly step: number;
  readonly disabled: boolean;
  readonly onChange: (value: number) => void;
}): React.JSX.Element {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-3 text-xs">
        <span className="font-semibold text-zinc-600">{label}</span>
        <span className="font-bold tabular-nums text-zinc-900">
          {value}{suffix}
        </span>
      </div>
      <Slider
        value={[value]}
        min={minimum}
        max={maximum}
        step={step}
        disabled={disabled}
        onValueChange={([next]) => onChange(next)}
        aria-label={label}
      />
    </div>
  );
}

function ChartSkeleton(): React.JSX.Element {
  return (
    <div className="panel">
      <div className="h-4 w-36 animate-pulse rounded bg-zinc-200" />
      <div className="mt-4 h-64 animate-pulse rounded bg-zinc-100" />
    </div>
  );
}

function RunBadge({ status }: { readonly status: RunStatus }): React.JSX.Element {
  if (status === "complete") {
    return (
      <Badge variant="success">
        <CheckCircle2 className="mr-1 size-3.5" />
        Verified
      </Badge>
    );
  }
  if (status === "running") {
    return <Badge>Running</Badge>;
  }
  if (status === "error") {
    return <Badge variant="warning">Failed</Badge>;
  }
  return <Badge variant="neutral">{status === "cancelled" ? "Cancelled" : "Ready"}</Badge>;
}

function speculativeMetrics(result: DashboardResult) {
  const metrics = result.speculative!.metrics;
  return [
    {
      label: "Efficiency",
      value: metrics.committedTokensPerTargetForward.toFixed(2),
      detail: "tokens / target forward",
      icon: <Gauge className="size-4" />,
    },
    {
      label: "Accepted drafts",
      value: metrics.acceptedDraftTokens.toLocaleString(),
      detail: `${metrics.rejectedDraftTokens} rejected`,
      icon: <CheckCircle2 className="size-4 text-emerald-700" />,
    },
    {
      label: "KV high water",
      value: formatBytes(metrics.kvHighWaterReservedBytes),
      detail: `${metrics.kvPagesAllocated} pages allocated`,
      icon: <Database className="size-4 text-sky-700" />,
    },
    {
      label: "Runtime",
      value: `${result.durationMs.toFixed(1)} ms`,
      detail: `${metrics.iterations} iterations`,
      icon: <Clock3 className="size-4 text-amber-700" />,
    },
  ];
}

function expertMetrics(result: DashboardResult) {
  const metrics = result.expertCache!.metrics;
  return [
    {
      label: "Hot hit rate",
      value: `${(metrics.hotHitRate * 100).toFixed(1)}%`,
      detail: `${metrics.hotHits} hot accesses`,
      icon: <MemoryStick className="size-4 text-sky-700" />,
    },
    {
      label: "Bytes moved",
      value: formatBytes(metrics.bytesMoved),
      detail: `${metrics.evictions} evictions`,
      icon: <Network className="size-4 text-amber-700" />,
    },
    {
      label: "Cache stall",
      value: formatDuration(metrics.stallNs),
      detail: `${metrics.demandLoads} demand loads`,
      icon: <Clock3 className="size-4 text-rose-700" />,
    },
    {
      label: "Runtime",
      value: `${result.durationMs.toFixed(1)} ms`,
      detail: `${metrics.routes} token routes`,
      icon: <Cpu className="size-4 text-emerald-700" />,
    },
  ];
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) {
    return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
  }
  return `${(bytes / 1024 ** 2).toFixed(0)} MiB`;
}

function formatDuration(nanoseconds: number): string {
  if (nanoseconds >= 1_000_000) {
    return `${(nanoseconds / 1_000_000).toFixed(1)} ms`;
  }
  return `${Math.round(nanoseconds / 1_000)} us`;
}
