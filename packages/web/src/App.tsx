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
  FileDiff,
  FileCheck2,
  Gauge,
  MemoryStick,
  Network,
  Play,
  RotateCcw,
  Square,
  TriangleAlert,
  Upload,
  X,
} from "lucide-react";
import {
  MAX_CALIBRATION_FILE_BYTES,
  parseCalibrationFileText,
  type ParsedCalibrationFile,
} from "./calibration-import.js";
import {
  MAX_TOKEN_TRACE_FILE_BYTES,
  parseTokenTraceFileText,
  type ParsedTokenTraceFile,
} from "./token-trace-import.js";
import {
  MAX_RUNTIME_CAPTURE_FILE_BYTES,
  parseRuntimeCapturePairFileTexts,
  type ParsedRuntimeCapturePair,
} from "./runtime-capture-import.js";
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

const SPECULATIVE_FAMILIES: ReadonlyArray<{
  readonly value: DashboardRunConfig["speculative"]["family"];
  readonly label: string;
}> = [
  { value: "prompt_lookup", label: "Prompt lookup" },
  { value: "draft_model", label: "Draft model" },
  { value: "mtp", label: "MTP" },
  { value: "eagle3", label: "EAGLE-3" },
  { value: "shared_kv", label: "Shared KV" },
  { value: "self_speculative", label: "Self speculative (design)" },
];

const DEFAULT_CONFIG: DashboardRunConfig = {
  scenarioName: "multi-gpu",
  mode: "speculative",
  seed: 42,
  speculative: {
    family: "mtp",
    outputTokens: 128,
    draftWidth: 4,
    firstPositionAcceptance: 0.82,
  },
  serving: {
    compareTopologies: false,
    decodeMode: "mtp",
    draftWidth: 4,
    firstPositionAcceptance: 0.82,
    requestCount: 12,
    arrivalGapUs: 250,
    promptTokens: 512,
    outputTokens: 64,
    maxBatchSize: 8,
    maxBatchTokens: 128,
    prefillChunkTokens: 64,
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

type RunStatus =
  | "idle"
  | "running"
  | "complete"
  | "mismatch"
  | "cancelled"
  | "error";

interface RunState {
  readonly status: RunStatus;
  readonly progress: number;
  readonly phase: string;
  readonly result?: DashboardResult;
  readonly error?: string;
}

interface CalibrationSelection {
  readonly fileName?: string;
  readonly parsed?: ParsedCalibrationFile;
  readonly error?: string;
}

interface TokenTraceSelection {
  readonly fileName?: string;
  readonly parsed?: ParsedTokenTraceFile;
  readonly runtimePair?: ParsedRuntimeCapturePair;
  readonly error?: string;
}

export function App(): React.JSX.Element {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [runState, setRunState] = useState<RunState>({
    status: "idle",
    progress: 0,
    phase: "Ready",
  });
  const [calibration, setCalibration] = useState<CalibrationSelection>({});
  const [tokenTrace, setTokenTrace] = useState<TokenTraceSelection>({});
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

  const importCalibration = useCallback(async (file: File) => {
    try {
      if (file.size > MAX_CALIBRATION_FILE_BYTES) {
        throw new Error("calibration file exceeds the 1 MiB limit");
      }
      const parsed = await parseCalibrationFileText(
        await file.text(),
        file.name,
      );
      changeConfig({ ...config, calibration: parsed.dataset });
      setCalibration({ fileName: file.name, parsed });
    } catch (error) {
      setCalibration((current) => ({
        ...current,
        ...(current.parsed ? {} : { fileName: file.name }),
        error: error instanceof Error ? error.message : String(error),
      }));
    }
  }, [changeConfig, config]);

  const clearCalibration = useCallback(() => {
    const { calibration: _calibration, ...withoutCalibration } = config;
    changeConfig(withoutCalibration);
    setCalibration({});
  }, [changeConfig, config]);

  const importTokenTrace = useCallback(async (file: File) => {
    try {
      if (file.size > MAX_TOKEN_TRACE_FILE_BYTES) {
        throw new Error("token trace file exceeds the 1 MiB limit");
      }
      const parsed = await parseTokenTraceFileText(
        await file.text(),
        file.name,
      );
      changeConfig({
        ...config,
        mode: "speculative",
        speculative: {
          ...config.speculative,
          family: parsed.trace.family,
          outputTokens: parsed.trace.expectedOutputTokenIds.length,
          draftWidth: parsed.trace.maxAdditionalTokens,
          trace: parsed.trace,
        },
      });
      setTokenTrace({ fileName: file.name, parsed });
    } catch (error) {
      setTokenTrace((current) => ({
        ...current,
        ...(current.parsed ? {} : { fileName: file.name }),
        error: error instanceof Error ? error.message : String(error),
      }));
    }
  }, [changeConfig, config]);

  const importRuntimeCaptures = useCallback(async (
    files: readonly File[],
  ) => {
    try {
      if (files.length !== 2) {
        throw new Error("runtime evidence import requires exactly two files");
      }
      if (files.some((file) => file.size > MAX_RUNTIME_CAPTURE_FILE_BYTES)) {
        throw new Error("runtime capture file exceeds the 1 MiB limit");
      }
      const runtimePair = await parseRuntimeCapturePairFileTexts(
        await Promise.all(files.map(async (file) => ({
          fileName: file.name,
          text: await file.text(),
        }))),
      );
      const parsed: ParsedTokenTraceFile = {
        trace: runtimePair.trace,
        preview: runtimePair.preview,
      };
      changeConfig({
        ...config,
        mode: "speculative",
        speculative: {
          ...config.speculative,
          family: parsed.trace.family,
          outputTokens: parsed.trace.expectedOutputTokenIds.length,
          draftWidth: parsed.trace.maxAdditionalTokens,
          trace: parsed.trace,
        },
      });
      setTokenTrace({
        fileName:
          `${runtimePair.targetOnly.fileName} + ${runtimePair.speculative.fileName}`,
        parsed,
        runtimePair,
      });
    } catch (error) {
      setTokenTrace((current) => ({
        ...current,
        error: error instanceof Error ? error.message : String(error),
      }));
    }
  }, [changeConfig, config]);

  const clearTokenTrace = useCallback(() => {
    const { trace: _trace, ...withoutTrace } = config.speculative;
    changeConfig({ ...config, speculative: withoutTrace });
    setTokenTrace({});
  }, [changeConfig, config]);

  const reset = useCallback(() => {
    changeConfig(DEFAULT_CONFIG);
    setCalibration({});
    setTokenTrace({});
  }, [changeConfig]);

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
      const tokenMismatch =
        message.result.speculative?.tokenTrace?.matchesTargetOnly === false;
      setRunState({
        status: tokenMismatch ? "mismatch" : "complete",
        progress: 100,
        phase: tokenMismatch ? "Token mismatch" : "Replay verified",
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
                  onClick={reset}
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
              calibration={calibration}
              onCalibrationFile={importCalibration}
              onClearCalibration={clearCalibration}
              tokenTrace={tokenTrace}
              onTokenTraceFile={importTokenTrace}
              onRuntimeCaptureFiles={importRuntimeCaptures}
              onClearTokenTrace={clearTokenTrace}
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
  calibration,
  onCalibrationFile,
  onClearCalibration,
  tokenTrace,
  onTokenTraceFile,
  onRuntimeCaptureFiles,
  onClearTokenTrace,
  onRun,
  onCancel,
  running,
}: {
  readonly config: DashboardRunConfig;
  readonly disabled: boolean;
  readonly onChange: (config: DashboardRunConfig) => void;
  readonly calibration: CalibrationSelection;
  readonly onCalibrationFile: (file: File) => void;
  readonly onClearCalibration: () => void;
  readonly tokenTrace: TokenTraceSelection;
  readonly onTokenTraceFile: (file: File) => void;
  readonly onRuntimeCaptureFiles: (files: readonly File[]) => void;
  readonly onClearTokenTrace: () => void;
  readonly onRun: () => void;
  readonly onCancel: () => void;
  readonly running: boolean;
}): React.JSX.Element {
  const calibrationInput = useRef<HTMLInputElement | null>(null);
  const tokenTraceInput = useRef<HTMLInputElement | null>(null);
  const runtimeCaptureInput = useRef<HTMLInputElement | null>(null);
  const setMode = (mode: WorkloadMode) => onChange({ ...config, mode });
  return (
    <div className="configuration-panel mx-auto max-w-md lg:max-w-none">
      <div className="configuration-scroll">
        <div className="mb-4">
          <h2 className="text-sm font-bold">Run configuration</h2>
          <p className="mt-0.5 text-xs text-zinc-500">Seed {config.seed}</p>
        </div>

        {config.mode === "serving" && config.serving.compareTopologies
          ? null
          : (
              <Field label="Device topology">
                <Select
                  value={config.scenarioName}
                  disabled={disabled}
                  onValueChange={(scenarioName) => onChange({
                    ...config,
                    scenarioName:
                      scenarioName as DashboardRunConfig["scenarioName"],
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
            )}

        <div className="mb-4 border-y border-zinc-200 py-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="text-xs font-semibold text-zinc-600">
              Timing evidence
            </span>
            <Badge
              variant={calibration.parsed?.fit.confidence === "calibrated"
                ? "success"
                : "warning"}
            >
              {calibration.parsed
                ? calibration.parsed.dataset.provenance.kind
                : "bundled heuristic"}
            </Badge>
          </div>
          <input
            ref={calibrationInput}
            type="file"
            accept=".yaml,.yml,.json,application/json,text/yaml"
            className="sr-only"
            disabled={disabled}
            onChange={(event) => {
              const file = event.currentTarget.files?.[0];
              event.currentTarget.value = "";
              if (file) {
                onCalibrationFile(file);
              }
            }}
          />
          {calibration.parsed
            ? (
                <div className="flex min-w-0 items-center gap-2">
                  <FileCheck2 className="size-4 shrink-0 text-emerald-700" />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-xs font-semibold text-zinc-800">
                      {calibration.fileName}
                    </div>
                    <div className="truncate text-[11px] text-zinc-500">
                      {calibration.parsed.fit.datasetFingerprint}
                    </div>
                  </div>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="size-8"
                        aria-label="Replace calibration"
                        disabled={disabled}
                        onClick={() => calibrationInput.current?.click()}
                      >
                        <Upload className="size-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Replace calibration</TooltipContent>
                  </Tooltip>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="size-8"
                        aria-label="Remove calibration"
                        disabled={disabled}
                        onClick={onClearCalibration}
                      >
                        <X className="size-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Remove calibration</TooltipContent>
                  </Tooltip>
                </div>
              )
            : (
                <Button
                  type="button"
                  variant="secondary"
                  className="w-full"
                  disabled={disabled}
                  onClick={() => calibrationInput.current?.click()}
                >
                  <Upload className="size-4" />
                  Import YAML or JSON
                </Button>
              )}
          {calibration.error
            ? (
                <div className="mt-2 text-xs leading-4 text-rose-700">
                  {calibration.error}
                </div>
              )
            : null}
        </div>

        <Tabs
          value={config.mode}
          onValueChange={(mode) => setMode(mode as WorkloadMode)}
        >
        <TabsList className="mb-4 w-full grid-cols-3">
          <TabsTrigger value="serving">Serving</TabsTrigger>
          <TabsTrigger value="speculative" aria-label="Speculative">
            Spec
          </TabsTrigger>
          <TabsTrigger value="expert-cache">Experts</TabsTrigger>
        </TabsList>
        <TabsContent value="serving" className="space-y-5">
          <Field label="Topology scope">
            <Select
              value={config.serving.compareTopologies ? "all" : "single"}
              disabled={disabled}
              onValueChange={(scope) => onChange({
                ...config,
                serving: {
                  ...config.serving,
                  compareTopologies: scope === "all",
                },
              })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="single">Selected topology</SelectItem>
                <SelectItem value="all">Compare all six</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <Field label="Decode mode">
            <Select
              value={config.serving.decodeMode}
              disabled={disabled}
              onValueChange={(decodeMode) => onChange({
                ...config,
                serving: {
                  ...config.serving,
                  decodeMode:
                    decodeMode as DashboardRunConfig["serving"]["decodeMode"],
                },
              })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="target_only">Target only</SelectItem>
                {SPECULATIVE_FAMILIES.map((family) => (
                  <SelectItem key={family.value} value={family.value}>
                    {family.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          {config.serving.decodeMode === "target_only"
            ? null
            : (
                <>
                  <SliderField
                    label="Draft width"
                    value={config.serving.draftWidth}
                    minimum={1}
                    maximum={8}
                    step={1}
                    disabled={disabled}
                    onChange={(draftWidth) => onChange({
                      ...config,
                      serving: { ...config.serving, draftWidth },
                    })}
                  />
                  <SliderField
                    label="Position 1 acceptance"
                    value={Math.round(
                      config.serving.firstPositionAcceptance * 100,
                    )}
                    suffix="%"
                    minimum={20}
                    maximum={98}
                    step={1}
                    disabled={disabled}
                    onChange={(acceptance) => onChange({
                      ...config,
                      serving: {
                        ...config.serving,
                        firstPositionAcceptance: acceptance / 100,
                      },
                    })}
                  />
                </>
              )}
          <SliderField
            label="Requests"
            value={config.serving.requestCount}
            minimum={1}
            maximum={32}
            step={1}
            disabled={disabled}
            onChange={(requestCount) => onChange({
              ...config,
              serving: { ...config.serving, requestCount },
            })}
          />
          <SliderField
            label="Arrival gap"
            value={config.serving.arrivalGapUs}
            suffix=" us"
            minimum={0}
            maximum={5000}
            step={50}
            disabled={disabled}
            onChange={(arrivalGapUs) => onChange({
              ...config,
              serving: { ...config.serving, arrivalGapUs },
            })}
          />
          <SliderField
            label="Prompt tokens"
            value={config.serving.promptTokens}
            minimum={64}
            maximum={2048}
            step={64}
            disabled={disabled}
            onChange={(promptTokens) => onChange({
              ...config,
              serving: { ...config.serving, promptTokens },
            })}
          />
          <SliderField
            label="Output tokens"
            value={config.serving.outputTokens}
            minimum={8}
            maximum={256}
            step={8}
            disabled={disabled}
            onChange={(outputTokens) => onChange({
              ...config,
              serving: { ...config.serving, outputTokens },
            })}
          />
          <SliderField
            label="Batch sequences"
            value={config.serving.maxBatchSize}
            minimum={1}
            maximum={16}
            step={1}
            disabled={disabled}
            onChange={(maxBatchSize) => onChange({
              ...config,
              serving: { ...config.serving, maxBatchSize },
            })}
          />
          <SliderField
            label="Batch token budget"
            value={config.serving.maxBatchTokens}
            minimum={16}
            maximum={512}
            step={16}
            disabled={disabled}
            onChange={(maxBatchTokens) => onChange({
              ...config,
              serving: { ...config.serving, maxBatchTokens },
            })}
          />
          <SliderField
            label="Prefill chunk"
            value={config.serving.prefillChunkTokens}
            minimum={16}
            maximum={512}
            step={16}
            disabled={disabled}
            onChange={(prefillChunkTokens) => onChange({
              ...config,
              serving: { ...config.serving, prefillChunkTokens },
            })}
          />
        </TabsContent>
        <TabsContent value="speculative" className="space-y-5">
          <div className="border-y border-zinc-200 py-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="text-xs font-semibold text-zinc-600">
                Token evidence
              </span>
              <Badge variant={tokenTrace.parsed
                ? tokenTrace.parsed.preview.differential.matchesTargetOnly
                  ? "success"
                  : "danger"
                : "neutral"}
              >
                {tokenTrace.parsed
                  ? tokenTrace.parsed.preview.differential.matchesTargetOnly
                    ? "parity"
                    : "mismatch"
                  : "seeded heuristic"}
              </Badge>
            </div>
            <input
              ref={tokenTraceInput}
              type="file"
              accept=".yaml,.yml,.json,application/json,text/yaml"
              className="sr-only"
              disabled={disabled}
              onChange={(event) => {
                const file = event.currentTarget.files?.[0];
                event.currentTarget.value = "";
                if (file) {
                  onTokenTraceFile(file);
                }
              }}
            />
            <input
              ref={runtimeCaptureInput}
              type="file"
              multiple
              accept=".yaml,.yml,.json,application/json,text/yaml"
              className="sr-only"
              disabled={disabled}
              onChange={(event) => {
                const files = Array.from(event.currentTarget.files ?? []);
                event.currentTarget.value = "";
                if (files.length > 0) {
                  onRuntimeCaptureFiles(files);
                }
              }}
            />
            {tokenTrace.parsed
              ? (
                  <div className="flex min-w-0 items-center gap-2">
                    <FileDiff className="size-4 shrink-0 text-sky-700" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-xs font-semibold text-zinc-800">
                        {tokenTrace.fileName}
                      </div>
                      <div className="truncate text-[11px] text-zinc-500">
                        {tokenTrace.runtimePair
                          ? `${tokenTrace.runtimePair.targetOnly.capture.id} + ${tokenTrace.runtimePair.speculative.capture.id}`
                          : `${tokenTrace.parsed.trace.id} · ${tokenTrace.parsed.trace.provenance.source}`}
                      </div>
                    </div>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="size-8"
                          aria-label={tokenTrace.runtimePair
                            ? "Replace runtime captures"
                            : "Replace token trace"}
                          disabled={disabled}
                          onClick={() => (
                            tokenTrace.runtimePair
                              ? runtimeCaptureInput.current?.click()
                              : tokenTraceInput.current?.click()
                          )}
                        >
                          <Upload className="size-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        {tokenTrace.runtimePair
                          ? "Replace runtime captures"
                          : "Replace token trace"}
                      </TooltipContent>
                    </Tooltip>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="size-8"
                          aria-label="Remove token trace"
                          disabled={disabled}
                          onClick={onClearTokenTrace}
                        >
                          <X className="size-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Remove token trace</TooltipContent>
                    </Tooltip>
                  </div>
                )
              : (
                  <div className="grid gap-2">
                    <Button
                      type="button"
                      variant="secondary"
                      className="w-full"
                      disabled={disabled}
                      onClick={() => runtimeCaptureInput.current?.click()}
                    >
                      <FileCheck2 className="size-4" />
                      Import runtime captures
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      className="w-full"
                      disabled={disabled}
                      onClick={() => tokenTraceInput.current?.click()}
                    >
                      <Upload className="size-4" />
                      Import assembled trace
                    </Button>
                  </div>
                )}
            {tokenTrace.error
              ? (
                  <div className="mt-2 text-xs leading-4 text-rose-700">
                    {tokenTrace.error}
                  </div>
                )
              : null}
          </div>
          {tokenTrace.parsed
            ? (
                <div className="grid grid-cols-3 gap-3 border-b border-zinc-200 pb-4">
                  <DiagnosticValue
                    label="Family"
                    value={tokenTrace.parsed.trace.family.replaceAll("_", " ")}
                  />
                  <DiagnosticValue
                    label="Output"
                    value={`${tokenTrace.parsed.trace.expectedOutputTokenIds.length} tok`}
                  />
                  <DiagnosticValue
                    label="Iterations"
                    value={String(tokenTrace.parsed.trace.iterations.length)}
                  />
                </div>
              )
            : (
                <>
                  <Field label="Proposer family">
                    <Select
                      value={config.speculative.family}
                      disabled={disabled}
                      onValueChange={(family) => onChange({
                        ...config,
                        speculative: {
                          ...config.speculative,
                          family:
                            family as DashboardRunConfig["speculative"]["family"],
                        },
                      })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {SPECULATIVE_FAMILIES.map((family) => (
                          <SelectItem key={family.value} value={family.value}>
                            {family.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
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
                </>
              )}
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
      </div>

      <div className="configuration-action mt-6 flex gap-2 border-t border-zinc-200 bg-white pt-4">
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
    : result.mode === "serving"
      ? servingMetrics(result)
      : expertMetrics(result);
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-200 pb-3">
        <div className="text-xs text-zinc-500">
          {result.topology.planSteps.toLocaleString()} frozen-plan steps ·{" "}
          {result.topology.operationCounts.compute.toLocaleString()} compute ·{" "}
          {result.topology.operationCounts.transfer.toLocaleString()} transfer ·{" "}
          {result.topology.operationCounts.collective.toLocaleString()} collective
        </div>
        <div className="flex items-center gap-2">
          {result.comparison
            ? (
                <>
                  <Badge variant="neutral">
                    {result.comparison.length} topologies
                  </Badge>
                  <Badge variant="success">
                    fastest · {result.scenario.id}
                  </Badge>
                </>
              )
            : null}
          {result.speculative
            ? (
                <Badge variant={result.speculative.support === "design_only"
                  ? "warning"
                  : "neutral"}
                >
                  {result.speculative.family.replaceAll("_", " ")}
                  {result.speculative.support === "design_only"
                    ? " · design only"
                    : ""}
                </Badge>
              )
            : null}
          {result.speculative?.tokenTrace
            ? (
                <Badge variant={result.speculative.tokenTrace.matchesTargetOnly
                  ? "success"
                  : "danger"}
                >
                  {result.speculative.tokenTrace.matchesTargetOnly
                    ? "token parity"
                    : "token mismatch"}
                </Badge>
              )
            : null}
          {result.serving
            ? <Badge variant="neutral">continuous batch</Badge>
            : null}
          {result.serving && result.serving.decodeMode !== "target_only"
            ? (
                <Badge variant={result.serving.support === "design_only"
                  ? "warning"
                  : "neutral"}
                >
                  {result.serving.decodeMode.replaceAll("_", " ")}
                  {result.serving.support === "design_only"
                    ? " · design only"
                    : ""}
                </Badge>
              )
            : null}
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant={result.topology.confidence === "heuristic"
                ? "warning"
                : "success"}
              >
                {result.topology.confidence} timing evidence
              </Badge>
            </TooltipTrigger>
            <TooltipContent className="max-w-72">
              {result.topology.assumptions[1] ?? result.topology.assumptions[0]}
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
      {result.speculative?.tokenTrace
        ? <TokenTraceSummary result={result} />
        : null}
      {result.calibration ? <CalibrationSummary result={result} /> : null}
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

function TokenTraceSummary({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const trace = result.speculative!.tokenTrace!;
  return (
    <section className="grid gap-3 border-y border-zinc-200 bg-white px-3 py-3 sm:grid-cols-[minmax(0,1fr)_auto_auto] sm:items-center">
      <div className="flex min-w-0 items-center gap-3">
        {trace.matchesTargetOnly
          ? <FileCheck2 className="size-4 shrink-0 text-emerald-700" />
          : <TriangleAlert className="size-4 shrink-0 text-rose-700" />}
        <div className="min-w-0">
          <div className="truncate text-xs font-bold text-zinc-800">
            {trace.traceId}
          </div>
          <div className="truncate text-[11px] text-zinc-500">
            {trace.source} · {trace.runtimeRevision} · {trace.modelFingerprint}
          </div>
        </div>
      </div>
      <DiagnosticValue
        label="Differential"
        value={trace.matchesTargetOnly
          ? `${trace.comparedTokenCount} matched`
          : `Token ${(trace.firstMismatch?.outputIndex ?? 0) + 1}`}
      />
      <DiagnosticValue
        label="Run binding"
        value={trace.matchesTargetOnly
          ? "distinct IDs"
          : `${trace.firstMismatch?.expectedTokenId} -> ${trace.firstMismatch?.actualTokenId}`}
      />
    </section>
  );
}

function CalibrationSummary({
  result,
}: {
  readonly result: DashboardResult;
}): React.JSX.Element {
  const calibration = result.calibration!;
  const maxNormalizedRmse = Math.max(
    ...calibration.diagnostics.map((diagnostic) => diagnostic.normalizedRmse),
    ...calibration.transportDiagnostics.map(
      (diagnostic) => diagnostic.normalizedRmse,
    ),
  );
  const maxP95RelativeError = Math.max(
    ...calibration.diagnostics.map((diagnostic) => diagnostic.p95RelativeError),
    ...calibration.transportDiagnostics.map(
      (diagnostic) => diagnostic.p95RelativeError,
    ),
  );
  return (
    <section className="grid gap-3 border-y border-zinc-200 bg-white px-3 py-3 sm:grid-cols-[minmax(0,1fr)_auto_auto] sm:items-center">
      <div className="flex min-w-0 items-center gap-3">
        <FileCheck2 className="size-4 shrink-0 text-emerald-700" />
        <div className="min-w-0">
          <div className="truncate text-xs font-bold text-zinc-800">
            {calibration.datasetId}
          </div>
          <div className="truncate text-[11px] text-zinc-500">
            {calibration.datasetFingerprint} · {calibration.evidenceKind} ·{" "}
            {calibration.fitConfidence} fit ·{" "}
            {calibration.transportDiagnostics.length} transport curves
          </div>
        </div>
      </div>
      <DiagnosticValue
        label="Max NRMSE"
        value={`${(maxNormalizedRmse * 100).toFixed(2)}%`}
      />
      <DiagnosticValue
        label="Max P95 error"
        value={`${(maxP95RelativeError * 100).toFixed(2)}%`}
      />
    </section>
  );
}

function DiagnosticValue({
  label,
  value,
}: {
  readonly label: string;
  readonly value: string;
}): React.JSX.Element {
  return (
    <div className="min-w-24">
      <div className="text-[11px] text-zinc-500">{label}</div>
      <div className="text-xs font-bold tabular-nums text-zinc-800">{value}</div>
    </div>
  );
}

function Inspector({ result }: { readonly result?: DashboardResult }): React.JSX.Element {
  const rows = useMemo(() => {
    if (result?.comparison) {
      return result.comparison.map((entry) => ({
        id: `#${entry.rank} ${entry.scenarioId}`,
        primary: `P95 TTFT ${formatDuration(entry.p95TimeToFirstTokenNs)}`,
        secondary:
          `${entry.relativeToFastest.toFixed(2)}x · ${entry.batches} batches`,
        value: formatRate(entry.throughputTokensPerSecond),
      }));
    }
    if (result?.speculative) {
      return result.speculative.iterations.slice(-12).reverse().map((iteration) => ({
        id: `Iteration ${iteration.iteration + 1}`,
        primary:
          `${iteration.acceptedDraftTokens}/${iteration.proposedDraftTokens} proposal tokens`,
        secondary: [
          iteration.outcome.replace("_", " "),
          iteration.guaranteedTargetTokens > 0 ? "guaranteed prefix" : undefined,
        ].filter(Boolean).join(" · "),
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
    if (result?.serving) {
      return result.serving.requests.slice(-12).reverse().map((request) => ({
        id: request.id,
        primary: `TTFT ${formatDuration(request.timeToFirstTokenNs)}`,
        secondary: `${request.tokenTimestampsNs.length} tokens`,
        value: formatDuration(request.latencyNs),
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
  if (status === "mismatch") {
    return (
      <Badge variant="danger">
        <TriangleAlert className="mr-1 size-3.5" />
        Mismatch
      </Badge>
    );
  }
  if (status === "running") {
    return <Badge>Running</Badge>;
  }
  if (status === "error") {
    return <Badge variant="danger">Failed</Badge>;
  }
  return <Badge variant="neutral">{status === "cancelled" ? "Cancelled" : "Ready"}</Badge>;
}

function speculativeMetrics(result: DashboardResult) {
  const metrics = result.speculative!.metrics;
  const topology = result.topology.metrics;
  return [
    {
      label: "Efficiency",
      value: metrics.committedTokensPerTargetForward.toFixed(2),
      detail: "tokens / target forward",
      icon: <Gauge className="size-4" />,
    },
    {
      label: "Modeled latency",
      value: formatDuration(topology.totalDurationNs),
      detail: `${result.topology.confidence} device + link time`,
      icon: <Clock3 className="size-4 text-amber-700" />,
    },
    {
      label: "KV high water",
      value: formatBytes(metrics.kvHighWaterReservedBytes),
      detail: `${metrics.kvPagesAllocated} pages allocated`,
      icon: <Database className="size-4 text-sky-700" />,
    },
    {
      label: "Throughput",
      value: formatRate(topology.tokensPerSecond),
      detail:
        `${metrics.acceptedAdditionalTokens}/${metrics.proposedAdditionalTokens} additional drafts accepted`,
      icon: <Cpu className="size-4 text-emerald-700" />,
    },
  ];
}

function expertMetrics(result: DashboardResult) {
  const metrics = result.expertCache!.metrics;
  const topology = result.topology.metrics;
  return [
    {
      label: "Hot hit rate",
      value: `${(metrics.hotHitRate * 100).toFixed(1)}%`,
      detail: `${metrics.hotHits} hot accesses`,
      icon: <MemoryStick className="size-4 text-sky-700" />,
    },
    {
      label: "Modeled latency",
      value: formatDuration(topology.totalDurationNs),
      detail: `${result.topology.confidence} device + link time`,
      icon: <Clock3 className="size-4 text-amber-700" />,
    },
    {
      label: "Throughput",
      value: formatRate(topology.tokensPerSecond),
      detail: `${metrics.demandLoads} demand loads`,
      icon: <Cpu className="size-4 text-emerald-700" />,
    },
    {
      label: "Bytes moved",
      value: formatBytes(metrics.bytesMoved),
      detail: `${metrics.evictions} evictions · ${formatDuration(metrics.stallNs)} cache stall`,
      icon: <Network className="size-4 text-amber-700" />,
    },
  ];
}

function servingMetrics(result: DashboardResult) {
  const serving = result.serving!;
  const metrics = serving.metrics;
  const speculative = serving.decodeMode !== "target_only";
  return [
    {
      label: "P95 TTFT",
      value: formatDuration(metrics.p95TimeToFirstTokenNs),
      detail: result.comparison
        ? `${result.scenario.id} · fastest of ${result.comparison.length}`
        : `${metrics.requests} requests`,
      icon: <Clock3 className="size-4 text-amber-700" />,
    },
    {
      label: "P95 ITL",
      value: formatDuration(metrics.p95InterTokenLatencyNs),
      detail: speculative
        ? `${metrics.targetForwards} target verifications`
        : `${metrics.batches} continuous batches`,
      icon: <Gauge className="size-4 text-sky-700" />,
    },
    {
      label: "Throughput",
      value: formatRate(metrics.throughputTokensPerSecond),
      detail: speculative
        ? `${metrics.committedTokensPerTargetForward.toFixed(2)} tokens / target`
        : `${(metrics.tokenBatchUtilization * 100).toFixed(1)}% token slots`,
      icon: <Cpu className="size-4 text-emerald-700" />,
    },
    {
      label: "KV high water",
      value: `${metrics.kvHighWaterTokens.toLocaleString()} tok`,
      detail: speculative
        ? `${metrics.acceptedAdditionalTokens}/${metrics.proposedAdditionalTokens} additional drafts accepted`
        : `${(metrics.sequenceBatchUtilization * 100).toFixed(1)}% sequence slots`,
      icon: <Database className="size-4 text-sky-700" />,
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

function formatRate(tokensPerSecond: number): string {
  const value = tokensPerSecond >= 1000
    ? tokensPerSecond.toLocaleString(undefined, { maximumFractionDigits: 0 })
    : tokensPerSecond.toFixed(1);
  return `${value} tok/s`;
}
