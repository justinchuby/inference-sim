import {
  buildMultiGpuRingScenario,
  buildScenarioPreset,
  buildTopology,
  type ScenarioPresetName,
  type SimulationScenario,
} from "@inference-sim/core";
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
  Download,
  FileDiff,
  FileCheck2,
  FilePlay,
  FolderOpen,
  Gauge,
  History,
  MemoryStick,
  Network,
  Play,
  RotateCcw,
  SlidersHorizontal,
  Square,
  TriangleAlert,
  Trash2,
  Upload,
  Workflow,
  X,
} from "lucide-react";
import {
  deleteArtifactFromHistory,
  listArtifactHistory,
  readArtifactFromHistory,
  saveArtifactToHistory,
  type ArtifactHistoryEntry,
} from "./artifact-history.js";
import {
  MAX_CALIBRATION_FILE_BYTES,
  parseCalibrationFileText,
  type ParsedCalibrationFile,
} from "./calibration-import.js";
import {
  MAX_DASHBOARD_ARTIFACT_FILE_BYTES,
  parseDashboardArtifactFileText,
} from "./dashboard-artifact.js";
import { MAX_FROZEN_PLAN_FILE_BYTES } from "./frozen-plan-import.js";
import {
  MAX_ONNX_MANIFEST_FILE_BYTES,
  parseOnnxManifestFileText,
} from "./onnx-manifest-import.js";
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
import {
  MAX_SCENARIO_FILE_BYTES,
  parseScenarioFileText,
} from "./scenario-import.js";
import { importModelPackage } from "./model-import-client.js";
import type { ImportedModelPackage } from "./model-package-import.js";
import {
  createBuiltinModelBinding,
  createImportedModelBinding,
  DASHBOARD_MODEL_PRESETS,
  type DashboardModelPreset,
} from "./model-binding.js";
import {
  calculateIdealRoofline,
  summarizeModelPackage,
} from "./model-metrics.js";
import { Badge } from "./components/ui/badge.js";
import { Button } from "./components/ui/button.js";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "./components/ui/dialog.js";
import { Progress } from "./components/ui/progress.js";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select.js";
import { Slider } from "./components/ui/slider.js";
import { Switch } from "./components/ui/switch.js";
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
  DashboardArtifactDownload,
  DashboardArtifactExpectation,
  DashboardArtifactReplay,
  DashboardModelBinding,
  DashboardResult,
  DashboardRunConfig,
  FrozenPlanBrowserResult,
  OnnxSearchBrowserConfig,
  OnnxSearchBrowserResult,
  OnnxStaticBrowserConfig,
  OnnxStaticBrowserResult,
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
  { value: "custom", label: "Custom scenario" },
];

const MULTI_GPU_RANKS: readonly DashboardRunConfig["multiGpuRanks"][] = [
  2,
  4,
  8,
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
  multiGpuRanks: 2,
  modelBinding: createBuiltinModelBinding("llama-3-8b"),
  mode: "serving",
  seed: 42,
  speculative: {
    family: "mtp",
    outputTokens: 128,
    draftWidth: 4,
    firstPositionAcceptance: 0.82,
  },
  serving: {
    compareTopologies: false,
    useExpertCache: false,
    decodeMode: "target_only",
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
    placementStrategy: "contiguous",
    tokenCount: 96,
    topK: 2,
    expertCount: 16,
    hotSlots: 6,
    warmSlots: 8,
    adaptivePrefetch: true,
  },
};

const DEFAULT_ONNX_CONFIG: OnnxStaticBrowserConfig = {
  hardwarePreset: "dgx-h100",
  kvCacheQuantization: "fp16",
  activationQuantization: "fp16",
  batchSize: 1,
  inputSeqLen: 2048,
  outputSeqLen: 512,
  parallelism: {
    tensorParallel: 1,
    pipelineParallel: 1,
    expertParallel: 1,
    dataParallel: 1,
  },
  memory: {
    kvCacheBudgetFraction: 0.35,
    expertCacheBudgetFraction: 0.25,
    pinnedPoolFraction: 0.05,
    offloadStrategy: "none",
    prefetchAhead: 0,
    pressureThreshold: 0.9,
    reclaimBatchSize: 4,
  },
};

const DEFAULT_ONNX_SEARCH_CONFIG: OnnxSearchBrowserConfig = {
  objective: "decode_throughput",
  topologyScope: "all",
  kvCacheScope: "fp16_fp8",
  batchScope: "common",
  parallelismScope: "common",
  offloadScope: "none_partial",
  maximumDeviceUsedFraction: 0.9,
  topK: 10,
  maxCandidates: 10_000,
};

const ONNX_HARDWARE: ReadonlyArray<{
  readonly value: OnnxStaticBrowserConfig["hardwarePreset"];
  readonly label: string;
}> = [
  { value: "dgx-h100", label: "8x H100 SXM" },
  { value: "dgx-h200", label: "8x H200 SXM" },
  { value: "2x-dgx-h100", label: "16x H100 / 2 nodes" },
  { value: "a100-4x", label: "4x A100 80G" },
  { value: "rtx-4090-2x", label: "2x RTX 4090" },
  { value: "4x-mac-studio-m4", label: "4x M4 Max" },
];

const ResultCharts = lazy(() => import("./ResultCharts.js"));
const TopologyEditorDialog = lazy(
  () => import("./TopologyEditorDialog.js"),
);
const TopologyGraph = lazy(() => import("./TopologyGraph.js"));

type RunStatus =
  | "idle"
  | "running"
  | "complete"
  | "mismatch"
  | "artifact-mismatch"
  | "cancelled"
  | "error";

interface RunState {
  readonly status: RunStatus;
  readonly progress: number;
  readonly phase: string;
  readonly result?: DashboardResult;
  readonly artifact?: DashboardArtifactDownload;
  readonly artifactReplay?: DashboardArtifactReplay;
  readonly frozenPlan?: FrozenPlanBrowserResult;
  readonly onnxStatic?: OnnxStaticBrowserResult;
  readonly onnxSearch?: OnnxSearchBrowserResult;
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

interface OnnxManifestSelection {
  readonly fileName: string;
  readonly artifactText: string;
}

interface ScenarioSelection {
  readonly fileName?: string;
  readonly error?: string;
}

interface ModelPackageSelection {
  readonly label?: string;
  readonly result?: ImportedModelPackage;
  readonly embeddedBinding?: DashboardModelBinding;
  readonly importing?: boolean;
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
  const [customScenario, setCustomScenario] = useState<ScenarioSelection>({});
  const [modelPackage, setModelPackage] =
    useState<ModelPackageSelection>({});
  const [onnxManifest, setOnnxManifest] =
    useState<OnnxManifestSelection | undefined>(undefined);
  const [onnxConfig, setOnnxConfig] = useState(DEFAULT_ONNX_CONFIG);
  const [onnxSearchConfig, setOnnxSearchConfig] =
    useState(DEFAULT_ONNX_SEARCH_CONFIG);
  const [onnxMode, setOnnxMode] = useState<"analyze" | "search">("analyze");
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyEntries, setHistoryEntries] =
    useState<readonly ArtifactHistoryEntry[]>([]);
  const [historyError, setHistoryError] = useState<string | undefined>(
    undefined,
  );
  const workerRef = useRef<Worker | undefined>(undefined);
  const artifactInputRef = useRef<HTMLInputElement | null>(null);
  const frozenPlanInputRef = useRef<HTMLInputElement | null>(null);
  const onnxManifestInputRef = useRef<HTMLInputElement | null>(null);
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
      if (
        config.modelBinding !== undefined
        && !config.modelBinding.speculativeFamilies.includes(parsed.trace.family)
      ) {
        throw new Error(
          `model package does not declare ${parsed.trace.family}`,
        );
      }
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
      if (
        config.modelBinding !== undefined
        && !config.modelBinding.speculativeFamilies.includes(parsed.trace.family)
      ) {
        throw new Error(
          `model package does not declare ${parsed.trace.family}`,
        );
      }
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

  const importCustomScenario = useCallback(async (file: File) => {
    try {
      if (file.size > MAX_SCENARIO_FILE_BYTES) {
        throw new Error("scenario file exceeds the 4 MiB limit");
      }
      const parsed = await parseScenarioFileText(await file.text(), file.name);
      changeConfig({
        ...config,
        scenarioName: "custom",
        customScenario: parsed.scenario,
        serving: {
          ...config.serving,
          compareTopologies: false,
        },
      });
      setCustomScenario({ fileName: file.name });
    } catch (error) {
      setCustomScenario((current) => ({
        ...current,
        error: error instanceof Error ? error.message : String(error),
      }));
    }
  }, [changeConfig, config]);

  const clearCustomScenario = useCallback((
    scenarioName: Exclude<
      DashboardRunConfig["scenarioName"],
      "custom"
    > = "single-gpu-cpu",
  ) => {
    const { customScenario: _customScenario, ...withoutCustomScenario } = config;
    changeConfig({
      ...withoutCustomScenario,
      scenarioName,
    });
    setCustomScenario({});
  }, [changeConfig, config]);

  const applyEditedTopology = useCallback((
    scenario: SimulationScenario,
  ) => {
    changeConfig({
      ...config,
      scenarioName: "custom",
      customScenario: scenario,
      serving: {
        ...config.serving,
        compareTopologies: false,
      },
    });
    setCustomScenario({ fileName: "Web topology editor" });
  }, [changeConfig, config]);

  const importLocalModelPackage = useCallback(async (
    files: readonly File[],
    label: string,
  ) => {
    setModelPackage((current) => ({
      ...current,
      importing: true,
      error: undefined,
    }));
    try {
      const result = await importModelPackage(files);
      const families = result.metadata.speculative.availableFamilies;
      const selectedFamily = families.includes(config.speculative.family)
        ? config.speculative.family
        : families[0];
      const maximumDraftTokens = selectedFamily === undefined
        ? undefined
        : result.metadata.speculative.evidence.find(
            (item) => item.family === selectedFamily,
          )?.maximumDraftTokens;
      const { trace: _trace, ...speculative } = config.speculative;
      const modelBinding = createImportedModelBinding(result);
      changeConfig({
        ...config,
        modelBinding,
        mode: config.mode === "speculative" && selectedFamily === undefined
          ? "serving"
          : config.mode,
        speculative: {
          ...speculative,
          ...(selectedFamily === undefined ? {} : { family: selectedFamily }),
          ...(maximumDraftTokens === undefined
            ? {}
            : { draftWidth: Math.min(8, maximumDraftTokens) }),
        },
        serving: {
          ...config.serving,
          decodeMode: config.serving.decodeMode !== "target_only"
            && families.includes(config.serving.decodeMode)
            ? config.serving.decodeMode
            : "target_only",
        },
      });
      setTokenTrace({});
      setModelPackage({ label, result });
    } catch (error) {
      setModelPackage((current) => ({
        ...current,
        importing: false,
        error: error instanceof Error ? error.message : String(error),
      }));
    }
  }, [changeConfig, config]);

  const selectBuiltinModel = useCallback((
    preset: DashboardModelPreset,
  ) => {
    changeConfig({
      ...config,
      modelBinding: createBuiltinModelBinding(preset),
      mode: "serving",
      serving: {
        ...config.serving,
        decodeMode: "target_only",
        useExpertCache: false,
      },
    });
    setModelPackage({});
    setTokenTrace({});
  }, [changeConfig, config]);

  const reset = useCallback(() => {
    changeConfig(DEFAULT_CONFIG);
    setCalibration({});
    setTokenTrace({});
    setCustomScenario({});
    setModelPackage({});
    setOnnxManifest(undefined);
    setOnnxConfig(DEFAULT_ONNX_CONFIG);
    setOnnxSearchConfig(DEFAULT_ONNX_SEARCH_CONFIG);
    setOnnxMode("analyze");
  }, [changeConfig]);

  const run = useCallback((
    nextConfig: DashboardRunConfig,
    expectedArtifact?: DashboardArtifactExpectation,
  ) => {
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
      if (message.type !== "result") {
        setRunState({
          status: "error",
          progress: 100,
          phase: "Worker protocol failed",
          error: "dashboard worker returned an unexpected result",
        });
        worker.terminate();
        workerRef.current = undefined;
        return;
      }
      const result: DashboardResult = {
        ...message.summary,
        durationMs: message.durationMs,
      };
      const tokenMismatch =
        result.speculative?.tokenTrace?.matchesTargetOnly === false;
      const artifactMismatch =
        message.artifactReplay?.matches === false;
      setRunState({
        status: artifactMismatch
          ? "artifact-mismatch"
          : tokenMismatch
            ? "mismatch"
            : "complete",
        progress: 100,
        phase: artifactMismatch
          ? "Artifact mismatch"
          : tokenMismatch
            ? "Token mismatch"
            : message.artifactReplay
              ? "Artifact replay verified"
              : "Replay verified",
        result,
        artifact: message.artifact,
        ...(message.artifactReplay === undefined
          ? {}
          : { artifactReplay: message.artifactReplay }),
      });
      void saveArtifactToHistory(message.artifact, result)
        .then((entries) => {
          setHistoryEntries(entries);
          setHistoryError(undefined);
        })
        .catch((error) => {
          setHistoryError(
            error instanceof Error ? error.message : String(error),
          );
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
    worker.postMessage({
      type: "run",
      runId,
      config: nextConfig,
      ...(expectedArtifact === undefined ? {} : { expectedArtifact }),
    });
  }, []);

  const replayArtifactText = useCallback((
    text: string,
    fileName: string,
  ) => {
    try {
      const parsed = parseDashboardArtifactFileText(
        text,
        fileName,
      );
      setConfig(parsed.config);
      setCalibration(parsed.calibration
        ? {
            fileName: `${fileName} / embedded calibration`,
            parsed: parsed.calibration,
          }
        : {});
      setTokenTrace(parsed.tokenTrace
        ? {
            fileName: `${fileName} / embedded token trace`,
            parsed: parsed.tokenTrace,
          }
        : {});
      setCustomScenario(parsed.config.customScenario
        ? { fileName: `${fileName} / embedded scenario` }
        : {});
      setModelPackage(parsed.config.modelBinding
        ? {
            label: `${fileName} / embedded model binding`,
            embeddedBinding: parsed.config.modelBinding,
          }
        : {});
      setOnnxManifest(undefined);
      run(parsed.config, parsed.expectation);
    } catch (error) {
      setRunState({
        status: "error",
        progress: 100,
        phase: "Artifact import failed",
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }, [run]);

  const importArtifact = useCallback(async (file: File) => {
    if (file.size > MAX_DASHBOARD_ARTIFACT_FILE_BYTES) {
      setRunState({
        status: "error",
        progress: 100,
        phase: "Artifact import failed",
        error: "result artifact exceeds the 128 MiB limit",
      });
      return;
    }
    replayArtifactText(await file.text(), file.name);
  }, [replayArtifactText]);

  const replayHistoryEntry = useCallback(async (
    entry: ArtifactHistoryEntry,
  ) => {
    try {
      const stored = await readArtifactFromHistory(entry.fingerprint);
      setHistoryOpen(false);
      replayArtifactText(stored.text, stored.fileName);
      setHistoryEntries(await listArtifactHistory());
      setHistoryError(undefined);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : String(error));
    }
  }, [replayArtifactText]);

  const downloadHistoryEntry = useCallback(async (
    entry: ArtifactHistoryEntry,
  ) => {
    try {
      const stored = await readArtifactFromHistory(entry.fingerprint);
      downloadArtifact({
        blob: new Blob([stored.text], { type: "application/json" }),
        fileName: stored.fileName,
        artifactFingerprint: entry.fingerprint,
      });
      setHistoryEntries(await listArtifactHistory());
      setHistoryError(undefined);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const deleteHistoryEntry = useCallback(async (
    fingerprint: string,
  ) => {
    try {
      setHistoryEntries(await deleteArtifactFromHistory(fingerprint));
      setHistoryError(undefined);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const runFrozenPlan = useCallback(async (file: File) => {
    try {
      if (file.size > MAX_FROZEN_PLAN_FILE_BYTES) {
        throw new Error("FrozenPlan artifact exceeds the 128 MiB limit");
      }
      if (!file.name.toLowerCase().endsWith(".json")) {
        throw new Error("FrozenPlan artifact file must use .json");
      }
      const artifactText = await file.text();
      workerRef.current?.terminate();
      const worker = new Worker(new URL("./sim-worker.ts", import.meta.url), {
        type: "module",
      });
      const runId = ++runIdRef.current;
      workerRef.current = worker;
      setRunState({
        status: "running",
        progress: 4,
        phase: "Starting FrozenPlan worker",
      });
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const message = event.data;
        if (message.runId !== runIdRef.current) {
          return;
        }
        if (message.type === "progress") {
          setRunState((current) => ({
            ...current,
            progress: message.progress,
            phase: message.phase,
          }));
          return;
        }
        if (message.type === "error") {
          setRunState({
            status: "error",
            progress: 100,
            phase: "FrozenPlan execution failed",
            error: message.message,
          });
        } else if (message.type === "frozen-plan-result") {
          setRunState({
            status: "complete",
            progress: 100,
            phase: "FrozenPlan replay verified",
            frozenPlan: message.result,
          });
        } else {
          setRunState({
            status: "error",
            progress: 100,
            phase: "FrozenPlan execution failed",
            error: "FrozenPlan worker returned an unexpected result",
          });
        }
        worker.terminate();
        workerRef.current = undefined;
      };
      worker.onerror = (event) => {
        setRunState({
          status: "error",
          progress: 100,
          phase: "FrozenPlan worker failed",
          error: event.message,
        });
        worker.terminate();
        workerRef.current = undefined;
      };
      worker.postMessage({
        type: "run-frozen-plan",
        runId,
        sourceFileName: file.name,
        artifactText,
      });
    } catch (error) {
      setRunState({
        status: "error",
        progress: 100,
        phase: "FrozenPlan import failed",
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }, []);

  const runOnnxStatic = useCallback((
    selection: OnnxManifestSelection,
    nextConfig: OnnxStaticBrowserConfig,
  ) => {
    workerRef.current?.terminate();
    const worker = new Worker(new URL("./sim-worker.ts", import.meta.url), {
      type: "module",
    });
    const runId = ++runIdRef.current;
    workerRef.current = worker;
    setRunState({
      status: "running",
      progress: 4,
      phase: "Starting ONNX analysis worker",
    });
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.runId !== runIdRef.current) {
        return;
      }
      if (message.type === "progress") {
        setRunState((current) => ({
          ...current,
          progress: message.progress,
          phase: message.phase,
        }));
        return;
      }
      if (message.type === "error") {
        setRunState({
          status: "error",
          progress: 100,
          phase: "ONNX analysis failed",
          error: message.message,
        });
      } else if (message.type === "onnx-static-result") {
        setRunState({
          status: "complete",
          progress: 100,
          phase: message.result.analysis.feasible
            ? "Capacity check passed"
            : "Capacity check failed",
          onnxStatic: message.result,
        });
      } else {
        setRunState({
          status: "error",
          progress: 100,
          phase: "ONNX analysis failed",
          error: "ONNX worker returned an unexpected result",
        });
      }
      worker.terminate();
      workerRef.current = undefined;
    };
    worker.onerror = (event) => {
      setRunState({
        status: "error",
        progress: 100,
        phase: "ONNX worker failed",
        error: event.message,
      });
      worker.terminate();
      workerRef.current = undefined;
    };
    worker.postMessage({
      type: "run-onnx-static",
      runId,
      sourceFileName: selection.fileName,
      artifactText: selection.artifactText,
      config: nextConfig,
    });
  }, []);

  const runOnnxSearch = useCallback((
    selection: OnnxManifestSelection,
    baseConfig: OnnxStaticBrowserConfig,
    searchConfig: OnnxSearchBrowserConfig,
  ) => {
    workerRef.current?.terminate();
    const worker = new Worker(new URL("./sim-worker.ts", import.meta.url), {
      type: "module",
    });
    const runId = ++runIdRef.current;
    workerRef.current = worker;
    setRunState({
      status: "running",
      progress: 4,
      phase: "Starting configuration search",
    });
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.runId !== runIdRef.current) {
        return;
      }
      if (message.type === "progress") {
        setRunState((current) => ({
          ...current,
          progress: message.progress,
          phase: message.phase,
        }));
        return;
      }
      if (message.type === "error") {
        setRunState({
          status: "error",
          progress: 100,
          phase: "Configuration search failed",
          error: message.message,
        });
      } else if (message.type === "onnx-search-result") {
        setRunState({
          status: "complete",
          progress: 100,
          phase: "Candidate ranking complete",
          onnxSearch: message.result,
        });
      } else {
        setRunState({
          status: "error",
          progress: 100,
          phase: "Configuration search failed",
          error: "search worker returned an unexpected result",
        });
      }
      worker.terminate();
      workerRef.current = undefined;
    };
    worker.onerror = (event) => {
      setRunState({
        status: "error",
        progress: 100,
        phase: "Configuration search worker failed",
        error: event.message,
      });
      worker.terminate();
      workerRef.current = undefined;
    };
    worker.postMessage({
      type: "run-onnx-search",
      runId,
      sourceFileName: selection.fileName,
      artifactText: selection.artifactText,
      baseConfig,
      searchConfig,
    });
  }, []);

  const importOnnxManifest = useCallback(async (file: File) => {
    try {
      if (file.size > MAX_ONNX_MANIFEST_FILE_BYTES) {
        throw new Error("ONNX manifest exceeds the 64 MiB limit");
      }
      const artifactText = await file.text();
      parseOnnxManifestFileText(artifactText, file.name);
      const selection = { fileName: file.name, artifactText };
      setOnnxManifest(selection);
      setOnnxConfig(DEFAULT_ONNX_CONFIG);
      setOnnxMode("analyze");
      runOnnxStatic(selection, DEFAULT_ONNX_CONFIG);
    } catch (error) {
      setRunState({
        status: "error",
        progress: 100,
        phase: "ONNX manifest import failed",
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }, [runOnnxStatic]);

  useEffect(() => {
    void listArtifactHistory()
      .then(setHistoryEntries)
      .catch((error) => {
        setHistoryError(error instanceof Error ? error.message : String(error));
      });
  }, []);

  useEffect(() => {
    if (!initializedRef.current) {
      initializedRef.current = true;
      run(DEFAULT_CONFIG);
    }
    return () => workerRef.current?.terminate();
  }, [run]);

  const result = runState.result;
  const selectedScenario = useMemo(
    () => resolveSelectedScenario(config),
    [config.customScenario, config.multiGpuRanks, config.scenarioName],
  );
  const displayedScenario = result?.comparison
    ? buildScenarioPreset(result.scenario.id as ScenarioPresetName)
    : selectedScenario;
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
                  aria-label="Open artifact history"
                  disabled={runState.status === "running"}
                  onClick={() => {
                    setHistoryOpen(true);
                    void listArtifactHistory()
                      .then(setHistoryEntries)
                      .catch((error) => {
                        setHistoryError(
                          error instanceof Error
                            ? error.message
                            : String(error),
                        );
                      });
                  }}
                >
                  <History className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                Artifact history · {historyEntries.length}
              </TooltipContent>
            </Tooltip>
            <input
              ref={onnxManifestInputRef}
              type="file"
              accept=".json,application/json"
              hidden
              disabled={runState.status === "running"}
              onChange={(event) => {
                const file = event.currentTarget.files?.[0];
                event.currentTarget.value = "";
                if (file) {
                  void importOnnxManifest(file);
                }
              }}
            />
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Import ONNX model manifest"
                  onClick={() => onnxManifestInputRef.current?.click()}
                  disabled={runState.status === "running"}
                >
                  <Database className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Import ONNX model manifest</TooltipContent>
            </Tooltip>
            <input
              ref={frozenPlanInputRef}
              type="file"
              accept=".json,application/json"
              hidden
              disabled={runState.status === "running"}
              onChange={(event) => {
                const file = event.currentTarget.files?.[0];
                event.currentTarget.value = "";
                if (file) {
                  void runFrozenPlan(file);
                }
              }}
            />
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Import and execute FrozenPlan"
                  onClick={() => frozenPlanInputRef.current?.click()}
                  disabled={runState.status === "running"}
                >
                  <FilePlay className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Import and execute FrozenPlan</TooltipContent>
            </Tooltip>
            <input
              ref={artifactInputRef}
              type="file"
              accept=".json,application/json"
              hidden
              disabled={runState.status === "running"}
              onChange={(event) => {
                const file = event.currentTarget.files?.[0];
                event.currentTarget.value = "";
                if (file) {
                  void importArtifact(file);
                }
              }}
            />
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Import and replay result"
                  onClick={() => artifactInputRef.current?.click()}
                  disabled={runState.status === "running"}
                >
                  <Upload className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Import and replay result</TooltipContent>
            </Tooltip>
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
            {onnxManifest
              ? (
                  <OnnxConfigurationPanel
                    selection={onnxManifest}
                    config={onnxConfig}
                    mode={onnxMode}
                    searchConfig={onnxSearchConfig}
                    disabled={runState.status === "running"}
                    running={runState.status === "running"}
                    onChange={(nextConfig) => {
                      setOnnxConfig(nextConfig);
                      setRunState((current) => current.status === "running"
                        ? current
                        : {
                            status: "idle",
                            progress: 0,
                            phase: "Configuration changed",
                          });
                    }}
                    onModeChange={(mode) => {
                      setOnnxMode(mode);
                      setRunState({
                        status: "idle",
                        progress: 0,
                        phase: mode === "search"
                          ? "Search configured"
                          : "Analysis configured",
                      });
                    }}
                    onSearchChange={(nextSearchConfig) => {
                      setOnnxSearchConfig(nextSearchConfig);
                      setRunState((current) => current.status === "running"
                        ? current
                        : {
                            status: "idle",
                            progress: 0,
                            phase: "Search configuration changed",
                          });
                    }}
                    onRun={() => onnxMode === "search"
                      ? runOnnxSearch(
                          onnxManifest,
                          onnxConfig,
                          onnxSearchConfig,
                        )
                      : runOnnxStatic(onnxManifest, onnxConfig)}
                    onCancel={cancel}
                    onClose={() => {
                      setOnnxManifest(undefined);
                      setRunState({
                        status: "idle",
                        progress: 0,
                        phase: "ONNX manifest closed",
                      });
                    }}
                  />
                )
              : (
                  <ConfigurationPanel
                    config={config}
                    modelPackage={modelPackage}
                    onModelPackageFiles={importLocalModelPackage}
                    onBuiltinModel={selectBuiltinModel}
                    customScenario={customScenario}
                    disabled={runState.status === "running"}
                    onChange={changeConfig}
                    onCustomScenarioFile={importCustomScenario}
                    onEditedTopology={applyEditedTopology}
                    onClearCustomScenario={clearCustomScenario}
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
                )}
          </aside>

          <main className="min-w-0 overflow-y-auto p-4 sm:p-5">
            <RunProgress state={runState} />
            {runState.onnxSearch
              ? <OnnxSearchResults result={runState.onnxSearch} />
              : runState.onnxStatic
              ? <OnnxStaticResults result={runState.onnxStatic} />
              : runState.frozenPlan
              ? <FrozenPlanResults result={runState.frozenPlan} />
              : result
              ? (
                  <Results
                    result={result}
                    topologyScenario={displayedScenario}
                    artifact={runState.artifact}
                    artifactReplay={runState.artifactReplay}
                  />
                )
              : modelPackage.result
              ? (
                  <ModelPackageOverview
                    modelPackage={modelPackage.result}
                    scenario={selectedScenario}
                  />
                )
              : runState.status === "idle" && selectedScenario
              ? <TopologyConfigurationPreview scenario={selectedScenario} />
              : <EmptyState state={runState} />}
          </main>

          <aside className="min-w-0 border-t border-zinc-200 bg-white p-4 xl:border-l xl:border-t-0">
            {runState.onnxSearch
              ? <OnnxSearchInspector result={runState.onnxSearch} />
              : runState.onnxStatic
              ? <OnnxInspector result={runState.onnxStatic} />
              : modelPackage.result
              ? <ModelPackageInspector modelPackage={modelPackage.result} />
              : <Inspector result={result} />}
          </aside>
        </div>
        <ArtifactHistoryDialog
          open={historyOpen}
          entries={historyEntries}
          error={historyError}
          onOpenChange={setHistoryOpen}
          onReplay={replayHistoryEntry}
          onDownload={downloadHistoryEntry}
          onDelete={deleteHistoryEntry}
        />
      </div>
    </TooltipProvider>
  );
}

function ArtifactHistoryDialog({
  open,
  entries,
  error,
  onOpenChange,
  onReplay,
  onDownload,
  onDelete,
}: {
  readonly open: boolean;
  readonly entries: readonly ArtifactHistoryEntry[];
  readonly error?: string;
  readonly onOpenChange: (open: boolean) => void;
  readonly onReplay: (entry: ArtifactHistoryEntry) => void;
  readonly onDownload: (entry: ArtifactHistoryEntry) => void;
  readonly onDelete: (fingerprint: string) => void;
}): React.JSX.Element {
  const totalBytes = entries.reduce(
    (sum, entry) => sum + entry.byteLength,
    0,
  );
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <div className="border-b border-zinc-200 px-5 py-4 pr-14">
          <DialogTitle className="text-sm font-bold">
            Artifact history
          </DialogTitle>
          <DialogDescription className="mt-1 text-xs text-zinc-500">
            {entries.length} artifacts · {formatBytes(totalBytes)}
          </DialogDescription>
        </div>
        {error
          ? (
              <div className="border-b border-rose-200 bg-rose-50 px-5 py-3 text-xs text-rose-800">
                {error}
              </div>
            )
          : null}
        <div className="max-h-[min(65vh,600px)] overflow-y-auto">
          {entries.length === 0
            ? (
                <div className="grid min-h-48 place-items-center px-6 text-center">
                  <div>
                    <History className="mx-auto size-6 text-zinc-400" />
                    <div className="mt-2 text-sm font-semibold">
                      No saved artifacts
                    </div>
                  </div>
                </div>
              )
            : (
                <div className="divide-y divide-zinc-200">
                  {entries.map((entry) => (
                    <div
                      key={entry.fingerprint}
                      className="grid gap-3 px-5 py-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center"
                    >
                      <div className="min-w-0">
                        <div className="flex min-w-0 items-center gap-2">
                          <span className="truncate text-xs font-bold text-zinc-800">
                            {entry.scenarioId}
                          </span>
                          <Badge variant="neutral">{entry.mode}</Badge>
                        </div>
                        <div className="mt-1 truncate font-mono text-[10px] text-zinc-500">
                          {entry.fingerprint}
                        </div>
                        <div className="mt-1 text-[11px] text-zinc-400">
                          {formatHistoryTimestamp(entry.savedAtMs)} ·{" "}
                          {formatBytes(entry.byteLength)}
                        </div>
                      </div>
                      <div className="flex items-center justify-end gap-1">
                        <Button
                          variant="secondary"
                          className="h-8"
                          onClick={() => onReplay(entry)}
                        >
                          <Play className="size-3.5 fill-current" />
                          Replay
                        </Button>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="size-8"
                              aria-label={`Download ${entry.fileName}`}
                              onClick={() => onDownload(entry)}
                            >
                              <Download className="size-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Download artifact</TooltipContent>
                        </Tooltip>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="size-8 text-rose-700"
                              aria-label={`Delete ${entry.fileName}`}
                              onClick={() => onDelete(entry.fingerprint)}
                            >
                              <Trash2 className="size-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Delete artifact</TooltipContent>
                        </Tooltip>
                      </div>
                    </div>
                  ))}
                </div>
              )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function formatHistoryTimestamp(milliseconds: number): string {
  return new Date(milliseconds).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function OnnxConfigurationPanel({
  selection,
  config,
  mode,
  searchConfig,
  disabled,
  running,
  onChange,
  onModeChange,
  onSearchChange,
  onRun,
  onCancel,
  onClose,
}: {
  readonly selection: OnnxManifestSelection;
  readonly config: OnnxStaticBrowserConfig;
  readonly mode: "analyze" | "search";
  readonly searchConfig: OnnxSearchBrowserConfig;
  readonly disabled: boolean;
  readonly running: boolean;
  readonly onChange: (config: OnnxStaticBrowserConfig) => void;
  readonly onModeChange: (mode: "analyze" | "search") => void;
  readonly onSearchChange: (config: OnnxSearchBrowserConfig) => void;
  readonly onRun: () => void;
  readonly onCancel: () => void;
  readonly onClose: () => void;
}): React.JSX.Element {
  const setParallelism = (
    field: keyof OnnxStaticBrowserConfig["parallelism"],
    value: number,
  ) => onChange({
    ...config,
    parallelism: { ...config.parallelism, [field]: value },
  });
  return (
    <div className="configuration-panel mx-auto max-w-md lg:max-w-none">
      <div className="configuration-scroll">
        <div className="mb-4 flex items-start gap-3 border-b border-zinc-200 pb-4">
          <Database className="mt-0.5 size-4 shrink-0 text-sky-700" />
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-sm font-bold">{selection.fileName}</h2>
            <p className="mt-0.5 text-xs text-zinc-500">ONNX model manifest</p>
          </div>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                aria-label="Close ONNX model manifest"
                disabled={disabled}
                onClick={onClose}
              >
                <X className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Close ONNX model manifest</TooltipContent>
          </Tooltip>
        </div>

        <div className="mb-4 grid grid-cols-2 gap-1 rounded-md border border-zinc-200 bg-zinc-50 p-1">
          {(["analyze", "search"] as const).map((candidateMode) => (
            <Button
              key={candidateMode}
              type="button"
              size="sm"
              className="capitalize"
              variant={mode === candidateMode ? "default" : "ghost"}
              disabled={disabled}
              aria-pressed={mode === candidateMode}
              onClick={() => onModeChange(candidateMode)}
            >
              {candidateMode}
            </Button>
          ))}
        </div>

        <Field label="Hardware">
          <Select
            value={config.hardwarePreset}
            disabled={disabled}
            onValueChange={(hardwarePreset) => onChange({
              ...config,
              hardwarePreset:
                hardwarePreset as OnnxStaticBrowserConfig["hardwarePreset"],
            })}
          >
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              {ONNX_HARDWARE.map((hardware) => (
                <SelectItem key={hardware.value} value={hardware.value}>
                  {hardware.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="KV cache">
            <QuantizationSelect
              value={config.kvCacheQuantization}
              disabled={disabled}
              onChange={(kvCacheQuantization) => onChange({
                ...config,
                kvCacheQuantization,
              })}
            />
          </Field>
          <Field label="Activations">
            <QuantizationSelect
              value={config.activationQuantization}
              disabled={disabled}
              onChange={(activationQuantization) => onChange({
                ...config,
                activationQuantization,
              })}
            />
          </Field>
        </div>
        <SliderField
          label="Batch size"
          value={config.batchSize}
          minimum={1}
          maximum={64}
          step={1}
          disabled={disabled}
          onChange={(batchSize) => onChange({ ...config, batchSize })}
        />
        <SliderField
          label="Input sequence"
          value={config.inputSeqLen}
          minimum={128}
          maximum={32768}
          step={128}
          disabled={disabled}
          onChange={(inputSeqLen) => onChange({ ...config, inputSeqLen })}
        />
        <SliderField
          label="Output sequence"
          value={config.outputSeqLen}
          minimum={16}
          maximum={4096}
          step={16}
          disabled={disabled}
          onChange={(outputSeqLen) => onChange({ ...config, outputSeqLen })}
        />
        {mode === "analyze"
          ? (
              <>
                <div className="grid grid-cols-3 gap-3 border-y border-zinc-200 py-4">
                  <CompactParallelismSelect
                    label="TP"
                    value={config.parallelism.tensorParallel}
                    disabled={disabled}
                    onChange={(value) => setParallelism("tensorParallel", value)}
                  />
                  <CompactParallelismSelect
                    label="PP"
                    value={config.parallelism.pipelineParallel}
                    disabled={disabled}
                    onChange={(value) => setParallelism("pipelineParallel", value)}
                  />
                  <CompactParallelismSelect
                    label="EP"
                    value={config.parallelism.expertParallel}
                    disabled={disabled}
                    onChange={(value) => setParallelism("expertParallel", value)}
                  />
                </div>
                <Field label="Offload">
                  <Select
                    value={config.memory.offloadStrategy}
                    disabled={disabled}
                    onValueChange={(offloadStrategy) => onChange({
                      ...config,
                      memory: {
                        ...config.memory,
                        offloadStrategy:
                          offloadStrategy as OnnxStaticBrowserConfig[
                            "memory"
                          ]["offloadStrategy"],
                      },
                    })}
                  >
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="none">None</SelectItem>
                      <SelectItem value="partial">Partial</SelectItem>
                      <SelectItem value="full">Full</SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
              </>
            )
          : (
              <div className="border-y border-zinc-200 py-4">
                <Field label="Ranking objective">
                  <Select
                    value={searchConfig.objective}
                    disabled={disabled}
                    onValueChange={(objective) => onSearchChange({
                      ...searchConfig,
                      objective:
                        objective as OnnxSearchBrowserConfig["objective"],
                    })}
                  >
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="decode_throughput">Decode throughput</SelectItem>
                      <SelectItem value="prefill_throughput">Prefill throughput</SelectItem>
                      <SelectItem value="time_to_first_token">Lowest TTFT</SelectItem>
                      <SelectItem value="inter_token_latency">Lowest ITL</SelectItem>
                      <SelectItem value="device_headroom">Device headroom</SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
                <SearchScopeField
                  label="Topologies"
                  value={searchConfig.topologyScope}
                  expandedValue="all"
                  selectedLabel="Selected"
                  expandedLabel="All six"
                  disabled={disabled}
                  onChange={(topologyScope) => onSearchChange({
                    ...searchConfig,
                    topologyScope: topologyScope as OnnxSearchBrowserConfig[
                      "topologyScope"
                    ],
                  })}
                />
                <SearchScopeField
                  label="KV cache axis"
                  value={searchConfig.kvCacheScope}
                  expandedValue="fp16_fp8"
                  selectedLabel="Selected"
                  expandedLabel="FP16 + FP8"
                  disabled={disabled}
                  onChange={(kvCacheScope) => onSearchChange({
                    ...searchConfig,
                    kvCacheScope: kvCacheScope as OnnxSearchBrowserConfig[
                      "kvCacheScope"
                    ],
                  })}
                />
                <SearchScopeField
                  label="Batch axis"
                  value={searchConfig.batchScope}
                  expandedValue="common"
                  selectedLabel="Selected"
                  expandedLabel="1, 4, 16"
                  disabled={disabled}
                  onChange={(batchScope) => onSearchChange({
                    ...searchConfig,
                    batchScope: batchScope as OnnxSearchBrowserConfig[
                      "batchScope"
                    ],
                  })}
                />
                <SearchScopeField
                  label="Parallelism axes"
                  value={searchConfig.parallelismScope}
                  expandedValue="common"
                  selectedLabel="Selected"
                  expandedLabel="Common powers"
                  disabled={disabled}
                  onChange={(parallelismScope) => onSearchChange({
                    ...searchConfig,
                    parallelismScope:
                      parallelismScope as OnnxSearchBrowserConfig[
                        "parallelismScope"
                      ],
                  })}
                />
                <SearchScopeField
                  label="Offload axis"
                  value={searchConfig.offloadScope}
                  expandedValue="none_partial"
                  selectedLabel="Selected"
                  expandedLabel="None + partial"
                  disabled={disabled}
                  onChange={(offloadScope) => onSearchChange({
                    ...searchConfig,
                    offloadScope: offloadScope as OnnxSearchBrowserConfig[
                      "offloadScope"
                    ],
                  })}
                />
                <SliderField
                  label="Maximum device use"
                  value={Math.round(
                    searchConfig.maximumDeviceUsedFraction * 100,
                  )}
                  suffix="%"
                  minimum={50}
                  maximum={100}
                  step={1}
                  disabled={disabled}
                  onChange={(maximumDeviceUsedFraction) => onSearchChange({
                    ...searchConfig,
                    maximumDeviceUsedFraction:
                      maximumDeviceUsedFraction / 100,
                  })}
                />
                <div className="flex items-center justify-between gap-3 bg-zinc-50 px-3 py-2 text-xs">
                  <span className="font-semibold text-zinc-600">
                    Declared candidates
                  </span>
                  <span className="font-bold tabular-nums text-zinc-900">
                    {declaredOnnxSearchCandidates(searchConfig).toLocaleString()}
                  </span>
                </div>
              </div>
            )}
      </div>
      <div className="configuration-action mt-3 flex gap-2 border-t border-zinc-200 bg-white pt-3">
        {running
          ? (
              <Button
                className="w-full"
                size="sm"
                variant="destructive"
                onClick={onCancel}
              >
                <Square className="size-4 fill-current" />
                Cancel
              </Button>
            )
          : (
              <Button className="w-full" size="sm" onClick={onRun}>
                <Play className="size-4 fill-current" />
                {mode === "search" ? "Search configurations" : "Analyze model"}
              </Button>
            )}
      </div>
    </div>
  );
}

function QuantizationSelect({
  value,
  disabled,
  onChange,
}: {
  readonly value: OnnxStaticBrowserConfig["kvCacheQuantization"];
  readonly disabled: boolean;
  readonly onChange: (
    value: OnnxStaticBrowserConfig["kvCacheQuantization"],
  ) => void;
}): React.JSX.Element {
  return (
    <Select
      value={value}
      disabled={disabled}
      onValueChange={(next) => onChange(
        next as OnnxStaticBrowserConfig["kvCacheQuantization"],
      )}
    >
      <SelectTrigger><SelectValue /></SelectTrigger>
      <SelectContent>
        {["fp32", "fp16", "bf16", "fp8", "int8", "int4"].map((quant) => (
          <SelectItem key={quant} value={quant}>{quant.toUpperCase()}</SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function SearchScopeField({
  label,
  value,
  expandedValue,
  selectedLabel,
  expandedLabel,
  disabled,
  onChange,
}: {
  readonly label: string;
  readonly value: string;
  readonly expandedValue: string;
  readonly selectedLabel: string;
  readonly expandedLabel: string;
  readonly disabled: boolean;
  readonly onChange: (value: string) => void;
}): React.JSX.Element {
  return (
    <Field label={label}>
      <Select
        value={value}
        disabled={disabled}
        onValueChange={onChange}
      >
        <SelectTrigger><SelectValue /></SelectTrigger>
        <SelectContent>
          <SelectItem value="selected">{selectedLabel}</SelectItem>
          <SelectItem value={expandedValue}>{expandedLabel}</SelectItem>
        </SelectContent>
      </Select>
    </Field>
  );
}

function declaredOnnxSearchCandidates(
  config: OnnxSearchBrowserConfig,
): number {
  return (config.topologyScope === "all" ? 6 : 1)
    * (config.kvCacheScope === "fp16_fp8" ? 2 : 1)
    * (config.batchScope === "common" ? 3 : 1)
    * (config.parallelismScope === "common" ? 4 * 3 * 4 : 1)
    * (config.offloadScope === "none_partial" ? 2 : 1);
}

function CompactParallelismSelect({
  label,
  value,
  disabled,
  onChange,
}: {
  readonly label: string;
  readonly value: number;
  readonly disabled: boolean;
  readonly onChange: (value: number) => void;
}): React.JSX.Element {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-semibold text-zinc-600">
        {label}
      </span>
      <Select
        value={String(value)}
        disabled={disabled}
        onValueChange={(next) => onChange(Number(next))}
      >
        <SelectTrigger><SelectValue /></SelectTrigger>
        <SelectContent>
          {[1, 2, 4, 8, 16].map((count) => (
            <SelectItem key={count} value={String(count)}>{count}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    </label>
  );
}

function resolveSelectedScenario(
  config: DashboardRunConfig,
): SimulationScenario | undefined {
  if (config.scenarioName === "custom") {
    return config.customScenario;
  }
  if (
    config.scenarioName === "multi-gpu"
    && config.multiGpuRanks !== 2
  ) {
    return buildMultiGpuRingScenario(config.multiGpuRanks);
  }
  return buildScenarioPreset(config.scenarioName as ScenarioPresetName);
}

function ConfigurationPanel({
  config,
  modelPackage,
  onModelPackageFiles,
  onBuiltinModel,
  customScenario,
  disabled,
  onChange,
  onCustomScenarioFile,
  onEditedTopology,
  onClearCustomScenario,
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
  readonly modelPackage: ModelPackageSelection;
  readonly onModelPackageFiles: (
    files: readonly File[],
    label: string,
  ) => void;
  readonly onBuiltinModel: (preset: DashboardModelPreset) => void;
  readonly customScenario: ScenarioSelection;
  readonly disabled: boolean;
  readonly onChange: (config: DashboardRunConfig) => void;
  readonly onCustomScenarioFile: (file: File) => void;
  readonly onEditedTopology: (scenario: SimulationScenario) => void;
  readonly onClearCustomScenario: (
    scenarioName?: Exclude<
      DashboardRunConfig["scenarioName"],
      "custom"
    >,
  ) => void;
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
  const [topologyEditorOpen, setTopologyEditorOpen] = useState(false);
  const modelFilesInput = useRef<HTMLInputElement | null>(null);
  const modelDirectoryInput = useRef<HTMLInputElement | null>(null);
  const calibrationInput = useRef<HTMLInputElement | null>(null);
  const scenarioInput = useRef<HTMLInputElement | null>(null);
  const tokenTraceInput = useRef<HTMLInputElement | null>(null);
  const runtimeCaptureInput = useRef<HTMLInputElement | null>(null);
  const availableSpeculativeFamilies = config.modelBinding
    ?.speculativeFamilies ?? SPECULATIVE_FAMILIES.map((family) => family.value);
  const speculativeOptions = SPECULATIVE_FAMILIES.filter((family) => (
    availableSpeculativeFamilies.includes(family.value)
  ));
  const selectedScenario = useMemo(() => resolveSelectedScenario(config), [
    config.customScenario,
    config.multiGpuRanks,
    config.scenarioName,
  ]);
  const setMode = (mode: WorkloadMode) => onChange({ ...config, mode });
  const selectedModelValue = config.modelBinding?.source === "builtin_model"
    ? config.modelBinding.targetModelFingerprint.replace(/^builtin:/, "")
    : "local";
  return (
    <div className="configuration-panel mx-auto max-w-md lg:max-w-none">
      <div className="configuration-scroll">
        <div className="mb-4">
          <h2 className="text-sm font-bold">Run configuration</h2>
          <p className="mt-0.5 text-xs text-zinc-500">Seed {config.seed}</p>
        </div>

        <div className="mb-4 border-y border-zinc-200 py-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="text-xs font-semibold text-zinc-700">
              Model
            </span>
            <Badge variant={config.modelBinding ? "success" : "danger"}>
              {modelPackage.importing
                ? "analyzing"
                : config.modelBinding?.source === "local_model_package"
                  ? "local ONNX"
                  : config.modelBinding
                    ? "built-in"
                    : "synthetic"}
            </Badge>
          </div>
          <input
            ref={modelFilesInput}
            type="file"
            multiple
            accept=".onnx,.data,.bin,.json,.yaml,.yml,application/json,text/yaml"
            className="sr-only"
            disabled={disabled || modelPackage.importing}
            onChange={(event) => {
              const files = Array.from(event.currentTarget.files ?? []);
              event.currentTarget.value = "";
              if (files.length > 0) {
                onModelPackageFiles(files, `${files.length} selected files`);
              }
            }}
          />
          <input
            ref={modelDirectoryInput}
            type="file"
            multiple
            {...{ webkitdirectory: "" }}
            className="sr-only"
            disabled={disabled || modelPackage.importing}
            onChange={(event) => {
              const files = Array.from(event.currentTarget.files ?? []);
              event.currentTarget.value = "";
              if (files.length > 0) {
                const root = files[0]?.webkitRelativePath.split("/")[0];
                onModelPackageFiles(files, root || "Local model folder");
              }
            }}
          />
          <Select
            value={selectedModelValue}
            disabled={disabled || modelPackage.importing}
            onValueChange={(value) => {
              if (value !== "local") {
                onBuiltinModel(value as DashboardModelPreset);
              }
            }}
          >
            <SelectTrigger aria-label="Model">
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {DASHBOARD_MODEL_PRESETS.map((preset) => (
                <SelectItem key={preset} value={preset}>
                  {createBuiltinModelBinding(preset).displayName}
                </SelectItem>
              ))}
              {config.modelBinding?.source === "local_model_package"
                ? (
                    <SelectItem value="local">
                      {config.modelBinding.displayName} (local)
                    </SelectItem>
                  )
                : null}
            </SelectContent>
          </Select>
          <div className="mt-1.5 grid grid-cols-2 gap-1.5">
            <Button
              type="button"
              size="sm"
              variant="secondary"
              className="gap-1 whitespace-nowrap px-2 text-xs"
              disabled={disabled || modelPackage.importing}
              onClick={() => modelDirectoryInput.current?.click()}
            >
              <FolderOpen className="size-4" />
              Import folder
            </Button>
            <Button
              type="button"
              size="sm"
              variant="secondary"
              className="gap-1 whitespace-nowrap px-2 text-xs"
              disabled={disabled || modelPackage.importing}
              onClick={() => modelFilesInput.current?.click()}
            >
              <Upload className="size-4" />
              Import files
            </Button>
          </div>
          {config.modelBinding
            ? (
                <div className="mt-2 flex min-w-0 items-start gap-2 text-[11px] text-zinc-500">
                  <Workflow className="mt-0.5 size-3.5 shrink-0 text-emerald-700" />
                  <div className="min-w-0">
                    <div className="truncate font-semibold text-zinc-800">
                      {config.modelBinding.displayName} ·{" "}
                      {formatLargeCount(config.modelBinding.totalParameters)} params
                    </div>
                    <div className="truncate">
                      {formatBytes(config.modelBinding.weightBytes)} weights ·{" "}
                      {formatFlops(
                        config.modelBinding.executionProfile.forwardFlopsPerToken,
                      )} / token
                    </div>
                  </div>
                </div>
              )
            : null}
          {modelPackage.result
            && (
              modelPackage.result.metadata.edges.length > 0
              || modelPackage.result.metadata.warnings.length > 0
            )
            ? (
                <div className="mt-2 text-[11px] leading-4 text-zinc-500">
                  {modelPackage.result.metadata.edges.length} pipeline edges
                  {modelPackage.result.metadata.warnings.length > 0
                    ? ` · ${modelPackage.result.metadata.warnings.join("; ")}`
                    : ""}
                </div>
              )
            : null}
          {modelPackage.error
            ? (
                <div className="mt-2 text-xs leading-4 text-rose-700">
                  {modelPackage.error}
                </div>
              )
            : null}
        </div>

        {config.mode === "serving" && config.serving.compareTopologies
          ? null
          : (
              <>
                <input
                  ref={scenarioInput}
                  type="file"
                  accept=".yaml,.yml,.json,application/json,text/yaml"
                  className="sr-only"
                  disabled={disabled}
                  onChange={(event) => {
                    const file = event.currentTarget.files?.[0];
                    event.currentTarget.value = "";
                    if (file) {
                      onCustomScenarioFile(file);
                    }
                  }}
                />
                <div className="mb-4">
                  <span className="mb-1.5 block text-xs font-semibold text-zinc-600">
                    Device topology
                  </span>
                  <div className="flex items-center gap-1.5">
                    <Select
                      value={config.scenarioName}
                      disabled={disabled}
                      onValueChange={(scenarioName) => {
                        if (
                          scenarioName === "custom"
                          && config.customScenario === undefined
                        ) {
                          scenarioInput.current?.click();
                          return;
                        }
                        if (scenarioName === "custom") {
                          onChange(config);
                        } else {
                          onClearCustomScenario(scenarioName as Exclude<
                            DashboardRunConfig["scenarioName"],
                            "custom"
                          >);
                        }
                      }}
                    >
                      <SelectTrigger
                        className="min-w-0 flex-1"
                        aria-label="Device topology"
                      >
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
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          type="button"
                          variant="secondary"
                          size="icon-sm"
                          className="shrink-0"
                          aria-label="Import custom scenario"
                          disabled={disabled}
                          onClick={() => scenarioInput.current?.click()}
                        >
                          <Upload className="size-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Import custom scenario</TooltipContent>
                    </Tooltip>
                    {selectedScenario
                      ? (
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                type="button"
                                variant="secondary"
                                size="icon-sm"
                                className="shrink-0"
                                aria-label="Edit device topology"
                                disabled={disabled}
                                onClick={() => setTopologyEditorOpen(true)}
                              >
                                <SlidersHorizontal className="size-4" />
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>Edit device topology</TooltipContent>
                          </Tooltip>
                        )
                      : null}
                  </div>
                </div>
                {config.scenarioName === "custom" && config.customScenario
                  ? (
                      <div className="mb-4 flex min-w-0 items-center gap-2 border-y border-zinc-200 bg-zinc-50 px-2 py-2">
                        <FileCheck2 className="size-4 shrink-0 text-emerald-700" />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-xs font-semibold text-zinc-800">
                            {config.customScenario.id}
                          </div>
                          <div className="truncate text-[11px] text-zinc-500">
                            {customScenario.fileName ?? "Embedded scenario"} ·{" "}
                            {config.customScenario.devices.length} devices ·{" "}
                            {config.customScenario.links.length} links
                          </div>
                        </div>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon-sm"
                              aria-label="Remove custom scenario"
                              disabled={disabled}
                              onClick={() => onClearCustomScenario()}
                            >
                              <X className="size-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Remove custom scenario</TooltipContent>
                        </Tooltip>
                      </div>
                    )
                  : null}
                {customScenario.error
                  ? (
                      <div className="mb-4 text-xs leading-4 text-rose-700">
                        {customScenario.error}
                      </div>
                    )
                  : null}
                {config.scenarioName === "multi-gpu"
                  ? (
                      <Field label="GPU ranks">
                        <Select
                          value={String(config.multiGpuRanks)}
                          disabled={disabled}
                          onValueChange={(rankCount) => {
                            const multiGpuRanks = MULTI_GPU_RANKS.find(
                              (candidate) => String(candidate) === rankCount,
                            );
                            if (multiGpuRanks !== undefined) {
                              onChange({ ...config, multiGpuRanks });
                            }
                          }}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {MULTI_GPU_RANKS.map((rankCount) => (
                              <SelectItem
                                key={rankCount}
                                value={String(rankCount)}
                              >
                                {rankCount} GPUs
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </Field>
                    )
                  : null}
                {selectedScenario
                  ? (
                      <div className="mb-4 border-y border-zinc-200 py-2.5">
                        <div className="mb-2 min-w-0">
                          <div className="min-w-0">
                            <div className="truncate text-xs font-semibold text-zinc-700">
                              {selectedScenario.devices.length} devices ·{" "}
                              {selectedScenario.memoryDomains.length} memory domains
                            </div>
                            <div className="truncate text-[11px] text-zinc-500">
                              {selectedScenario.links.length} directed links · epoch{" "}
                              {selectedScenario.execution.topologyEpoch}
                            </div>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                          {selectedScenario.devices.map((device) => {
                            const capacity = selectedScenario.memoryDomains
                              .filter((domain) => (
                                device.memoryDomainIds.includes(domain.id)
                                && domain.kind !== "storage"
                                && (
                                  domain.kind === "unified"
                                  || (
                                    device.kind === "cpu"
                                      ? domain.kind === "host"
                                      : domain.kind === "device"
                                  )
                                )
                              ))
                              .reduce(
                                (sum, domain) => sum + domain.capacityBytes,
                                0,
                              );
                            return (
                              <div
                                key={device.id}
                                className="min-w-0 text-[11px]"
                              >
                                <div className="truncate font-medium text-zinc-700">
                                  {device.id}
                                </div>
                                <div className="truncate text-zinc-500">
                                  {device.kind.toUpperCase()} · {formatBytes(capacity)}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )
                  : null}
              </>
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
                        size="icon-sm"
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
                        size="icon-sm"
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
                  size="sm"
                  variant="secondary"
                  className="w-full"
                  disabled={disabled}
                  onClick={() => calibrationInput.current?.click()}
                >
                  <Upload className="size-4" />
                  Import calibration
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
          <TabsTrigger
            value="speculative"
            aria-label="Speculative"
            disabled={config.modelBinding !== undefined
              && speculativeOptions.length === 0}
          >
            Spec
          </TabsTrigger>
          <TabsTrigger value="expert-cache">Experts</TabsTrigger>
        </TabsList>
        <TabsContent value="serving" className="space-y-4">
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
                {speculativeOptions.map((family) => (
                  <SelectItem key={family.value} value={family.value}>
                    {family.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <div className="flex min-h-9 items-center justify-between gap-3">
            <label
              className="text-sm font-medium text-zinc-700"
              htmlFor="serving-expert-cache"
            >
              Stateful expert cache
            </label>
            <Switch
              id="serving-expert-cache"
              checked={config.serving.useExpertCache}
              disabled={disabled}
              onCheckedChange={(useExpertCache) => onChange({
                ...config,
                serving: {
                  ...config.serving,
                  useExpertCache,
                },
              })}
            />
          </div>
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
        <TabsContent value="speculative" className="space-y-4">
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
                          size="icon-sm"
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
                          size="icon-sm"
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
                  <div className="grid grid-cols-2 gap-1.5">
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      disabled={disabled}
                      onClick={() => runtimeCaptureInput.current?.click()}
                    >
                      <FileCheck2 className="size-4" />
                      Captures
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      disabled={disabled}
                      onClick={() => tokenTraceInput.current?.click()}
                    >
                      <Upload className="size-4" />
                      Trace
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
                        {speculativeOptions.map((family) => (
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
        <TabsContent value="expert-cache" className="space-y-4">
          <label className="block">
            <span className="mb-1.5 block text-xs font-semibold text-zinc-600">
              Expert placement
            </span>
            <Select
              value={config.expertCache.placementStrategy}
              disabled={disabled}
              onValueChange={(placementStrategy) => onChange({
                ...config,
                expertCache: {
                  ...config.expertCache,
                  placementStrategy:
                    placementStrategy as DashboardRunConfig[
                      "expertCache"
                    ]["placementStrategy"],
                },
              })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="contiguous">Contiguous</SelectItem>
                <SelectItem value="round_robin">Round robin</SelectItem>
              </SelectContent>
            </Select>
          </label>
          <div className="mb-4">
            <span className="mb-1.5 block text-xs font-semibold text-zinc-600">
              Warm prefetch
            </span>
            <div className="grid grid-cols-2 gap-1 rounded-md border border-zinc-200 bg-zinc-50 p-1">
              {([
                [false, "Off"],
                [true, "Adaptive"],
              ] as const).map(([value, label]) => (
                <Button
                  key={label}
                  type="button"
                  size="sm"
                  variant={config.expertCache.adaptivePrefetch
                    === value ? "default" : "ghost"}
                  disabled={disabled || (value && config.expertCache.warmSlots === 0)}
                  aria-pressed={config.expertCache.adaptivePrefetch === value}
                  onClick={() => onChange({
                    ...config,
                    expertCache: {
                      ...config.expertCache,
                      adaptivePrefetch: value,
                    },
                  })}
                >
                  {label}
                </Button>
              ))}
            </div>
          </div>
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
            label="Hot slots / owner"
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
            label="Warm slots / node"
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
      {topologyEditorOpen && selectedScenario
        ? (
            <Suspense fallback={null}>
              <TopologyEditorDialog
                open={topologyEditorOpen}
                scenario={selectedScenario}
                onOpenChange={setTopologyEditorOpen}
                onSave={onEditedTopology}
              />
            </Suspense>
          )
        : null}

      <div className="configuration-action mt-3 flex gap-2 border-t border-zinc-200 bg-white pt-3">
        {running
          ? (
            <Button
              className="w-full"
              size="sm"
              variant="destructive"
              onClick={onCancel}
            >
              <Square className="size-4 fill-current" />
              Cancel
            </Button>
          )
          : (
            <Button className="w-full" size="sm" onClick={onRun}>
              <Play className="size-4 fill-current" />
              Run simulation
            </Button>
          )}
      </div>
    </div>
  );
}

function OnnxSearchResults({
  result,
}: {
  readonly result: OnnxSearchBrowserResult;
}): React.JSX.Element {
  const search = result.result;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 pb-3">
        <div className="min-w-0">
          <h1 className="truncate text-base font-bold">{result.modelName}</h1>
          <div className="mt-1 truncate text-xs text-zinc-500">
            {result.manifest.modelFileName} · {result.manifest.fingerprint}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="success">exhaustive declared space</Badge>
          <Badge variant="warning">{search.evidence.confidence} evidence</Badge>
        </div>
      </div>

      <div className="grid gap-px overflow-hidden border border-zinc-200 bg-zinc-200 sm:grid-cols-2 xl:grid-cols-4">
        <OnnxMetric
          label="Declared"
          value={search.declaredCandidateCount.toLocaleString()}
          detail="finite candidates"
        />
        <OnnxMetric
          label="Evaluated"
          value={search.evaluatedCandidateCount.toLocaleString()}
          detail="structurally valid"
        />
        <OnnxMetric
          label="Eligible"
          value={search.eligibleCandidateCount.toLocaleString()}
          detail="constraints passed"
        />
        <OnnxMetric
          label="Returned"
          value={search.returnedCandidateCount.toLocaleString()}
          detail="ranked candidates"
        />
      </div>

      <section>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-bold">Candidate ranking</h2>
          <Badge variant="neutral">
            {searchObjectiveLabel(search.objective)}
          </Badge>
        </div>
        <div className="overflow-x-auto border-y border-zinc-200">
          <table className="w-full min-w-[760px] text-left text-xs">
            <thead className="bg-zinc-50 text-zinc-500">
              <tr>
                {["Rank", "Topology", "Runtime", "Batch", "TP / PP / EP", "Device use", "Objective"].map((label) => (
                  <th key={label} className="px-3 py-2 font-semibold">{label}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 bg-white">
              {search.candidates.map((candidate) => (
                <tr key={candidate.candidateId}>
                  <td className="px-3 py-2 font-bold tabular-nums">
                    #{candidate.rank}
                  </td>
                  <td className="px-3 py-2 font-semibold">
                    {candidate.topologyId}
                  </td>
                  <td className="px-3 py-2 uppercase">
                    {candidate.kvCacheQuantization} KV ·{" "}
                    {candidate.memory.offloadStrategy}
                  </td>
                  <td className="px-3 py-2 tabular-nums">
                    {candidate.batchSize}
                  </td>
                  <td className="px-3 py-2 tabular-nums">
                    {candidate.parallelism.tensorParallel} /{" "}
                    {candidate.parallelism.pipelineParallel} /{" "}
                    {candidate.parallelism.expertParallel}
                  </td>
                  <td className="px-3 py-2 tabular-nums">
                    {(candidate.deviceUsedFraction * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-2 font-bold tabular-nums text-sky-800">
                    {formatSearchObjective(candidate, search.objective)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="border-y border-zinc-200 py-3">
        <h2 className="mb-2 text-sm font-bold">Search evidence</h2>
        <div className="space-y-2">
          {search.evidence.assumptions.map((assumption) => (
            <div key={assumption} className="flex gap-2 text-xs leading-5 text-zinc-600">
              <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-amber-600" />
              <span>{assumption}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function searchObjectiveLabel(
  objective: OnnxSearchBrowserResult["result"]["objective"],
): string {
  return {
    decode_throughput: "decode throughput",
    prefill_throughput: "prefill throughput",
    time_to_first_token: "lowest TTFT",
    inter_token_latency: "lowest ITL",
    device_headroom: "device headroom",
  }[objective];
}

function formatSearchObjective(
  candidate: OnnxSearchBrowserResult["result"]["candidates"][number],
  objective: OnnxSearchBrowserResult["result"]["objective"],
): string {
  switch (objective) {
    case "decode_throughput":
      return formatRate(candidate.analysis.estimatedThroughput.decodeToksPerSec);
    case "prefill_throughput":
      return formatRate(candidate.analysis.estimatedThroughput.prefillToksPerSec);
    case "time_to_first_token":
      return `${candidate.analysis.estimatedThroughput.timeToFirstTokenMs.toFixed(2)} ms`;
    case "inter_token_latency":
      return `${candidate.analysis.estimatedThroughput.interTokenLatencyMs.toFixed(3)} ms`;
    case "device_headroom":
      return `${((1 - candidate.deviceUsedFraction) * 100).toFixed(1)}% free`;
  }
}

function OnnxStaticResults({
  result,
}: {
  readonly result: OnnxStaticBrowserResult;
}): React.JSX.Element {
  const idealRoofline = useMemo(() => calculateIdealRoofline(
    result.model,
    buildTopology(result.config.hardwarePreset),
    result.config.parallelism.tensorParallel
      * result.config.parallelism.pipelineParallel
      * result.config.parallelism.expertParallel
      * result.config.parallelism.dataParallel,
  ), [result]);
  const firstDevice = result.analysis.memoryBreakdown[0];
  const used = firstDevice === undefined
    ? 0
    : firstDevice.totalBytes - firstDevice.free;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 pb-3">
        <div className="min-w-0">
          <h1 className="truncate text-base font-bold">{result.model.name}</h1>
          <div className="mt-1 truncate text-xs text-zinc-500">
            {result.manifest.modelFileName} · {result.manifest.fingerprint}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={result.analysis.feasible ? "success" : "danger"}>
            {result.analysis.feasible ? "capacity fit" : "capacity exceeded"}
          </Badge>
          <Badge variant="warning">
            {result.model.provenance.evidence} profile
          </Badge>
        </div>
      </div>

      <div className="grid gap-px overflow-hidden border border-zinc-200 bg-zinc-200 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
        <OnnxMetric
          label="Weight inventory"
          value={formatBytes(result.manifest.initializerLogicalBytes)}
          detail={`${result.model.totalParams.toLocaleString()} elements`}
        />
        <OnnxMetric
          label="Forward work"
          value={formatFlops(idealRoofline.forwardFlopsPerToken)}
          detail={`${formatBytes(idealRoofline.activeWeightBytesPerToken)} active weights / token`}
        />
        <OnnxMetric
          label="Ideal roofline"
          value={formatRate(idealRoofline.rooflineCeilingTokensPerSec)}
          detail={`${idealRoofline.limitingResource.replace("_", " ")} ceiling`}
        />
        <OnnxMetric
          label="Device used"
          value={firstDevice ? formatBytes(used) : "N/A"}
          detail={firstDevice
            ? `${formatBytes(firstDevice.totalBytes)} capacity`
            : "No devices"}
        />
        <OnnxMetric
          label="Decode"
          value={formatRate(
            result.analysis.estimatedThroughput.decodeToksPerSec,
          )}
          detail={`${result.analysis.estimatedThroughput.interTokenLatencyMs.toFixed(3)} ms ITL`}
        />
        <OnnxMetric
          label="Prefill"
          value={formatRate(
            result.analysis.estimatedThroughput.prefillToksPerSec,
          )}
          detail={`${result.analysis.estimatedThroughput.timeToFirstTokenMs.toFixed(2)} ms TTFT`}
        />
      </div>

      <section>
        <div className="mb-2 flex items-center justify-between gap-2">
          <h2 className="text-sm font-bold">Theoretical hardware ceilings</h2>
          <Badge variant="warning">ideal upper bound</Badge>
        </div>
        <div className="grid gap-px border border-zinc-200 bg-zinc-200 sm:grid-cols-2">
          <OnnxMetric
            label="Peak compute ceiling"
            value={idealRoofline.computeCeilingTokensPerSec === undefined
              ? "Unavailable"
              : formatRate(idealRoofline.computeCeilingTokensPerSec)}
            detail={idealRoofline.aggregatePeakComputeFlops === 0
              ? "Selected dtype has no declared peak"
              : `${formatFlops(idealRoofline.aggregatePeakComputeFlops)}/s aggregate`}
          />
          <OnnxMetric
            label="Weight-bandwidth ceiling"
            value={formatRate(idealRoofline.bandwidthCeilingTokensPerSec)}
            detail={`${formatBandwidth(idealRoofline.aggregateMemoryBandwidthBytesPerSec)} aggregate`}
          />
        </div>
        <p className="mt-2 text-[11px] leading-5 text-zinc-500">
          Ideal roofline assumes perfect device scaling and no KV, activation,
          communication, scheduling, or kernel overhead. Modeled throughput
          above applies utilization assumptions and is the planning estimate.
        </p>
      </section>

      <section>
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-sm font-bold">Per-device memory</h2>
          <Badge variant="neutral">
            {result.analysis.memoryBreakdown.length} devices
          </Badge>
        </div>
        <div className="overflow-x-auto border-y border-zinc-200">
          <table className="w-full min-w-[640px] text-left text-xs">
            <thead className="bg-zinc-50 text-zinc-500">
              <tr>
                {["Device", "Weights", "Experts", "KV cache", "Activations", "Free"].map((label) => (
                  <th key={label} className="px-3 py-2 font-semibold">{label}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 bg-white">
              {result.analysis.memoryBreakdown.map((memory) => (
                <tr key={memory.deviceId}>
                  <td className="px-3 py-2 font-semibold">{memory.deviceId}</td>
                  <td className="px-3 py-2 tabular-nums">{formatBytes(memory.weights)}</td>
                  <td className="px-3 py-2 tabular-nums">{formatBytes(memory.expertCache)}</td>
                  <td className="px-3 py-2 tabular-nums">{formatBytes(memory.kvCache)}</td>
                  <td className="px-3 py-2 tabular-nums">{formatBytes(memory.activations)}</td>
                  <td className={`px-3 py-2 font-semibold tabular-nums ${
                    memory.free < 0 ? "text-rose-700" : "text-emerald-700"
                  }`}>
                    {formatSignedBytes(memory.free)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="border-y border-zinc-200 py-3">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h2 className="text-sm font-bold">Model evidence</h2>
          <Badge variant="neutral">
            {result.manifest.architectureSource.replaceAll("_", " ")}
          </Badge>
        </div>
        <div className="space-y-2">
          {result.model.provenance.assumptions.map((assumption) => (
            <div key={assumption} className="flex gap-2 text-xs leading-5 text-zinc-600">
              <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-amber-600" />
              <span>{assumption}</span>
            </div>
          ))}
        </div>
      </section>

      {result.analysis.recommendations.length > 0
        ? (
            <section>
              <h2 className="mb-2 text-sm font-bold">Recommendations</h2>
              <div className="divide-y divide-zinc-200 border-y border-zinc-200">
                {result.analysis.recommendations.map((recommendation) => (
                  <div key={recommendation} className="py-2 text-xs text-zinc-700">
                    {recommendation}
                  </div>
                ))}
              </div>
            </section>
          )
        : null}
    </div>
  );
}

function OnnxMetric({
  label,
  value,
  detail,
}: {
  readonly label: string;
  readonly value: string;
  readonly detail: string;
}): React.JSX.Element {
  return (
    <div className="min-w-0 bg-white p-3">
      <div className="text-[11px] font-semibold text-zinc-500">{label}</div>
      <div className="mt-1 truncate text-lg font-bold tabular-nums">{value}</div>
      <div className="mt-0.5 truncate text-[11px] text-zinc-500">{detail}</div>
    </div>
  );
}

function TopologyConfigurationPreview({
  scenario,
}: {
  readonly scenario: SimulationScenario;
}): React.JSX.Element {
  return (
    <div className="space-y-5">
      <div className="border-b border-zinc-200 pb-3">
        <h1 className="text-base font-bold">Topology configuration</h1>
        <p className="mt-1 text-xs text-zinc-500">
          {scenario.id} · epoch {scenario.execution.topologyEpoch}
        </p>
      </div>
      <TopologyVisualizationSection scenario={scenario} title="Device map" />
    </div>
  );
}

function TopologyVisualizationSection({
  scenario,
  title,
}: {
  readonly scenario: SimulationScenario;
  readonly title: string;
}): React.JSX.Element {
  return (
    <section>
      <div className="mb-2 flex items-center justify-between gap-3">
        <h2 className="text-sm font-bold">{title}</h2>
        <Badge variant="neutral">
          {scenario.devices.length} devices · {scenario.links.length} links
        </Badge>
      </div>
      <Suspense
        fallback={(
          <div className="grid h-[430px] place-items-center border border-zinc-200 bg-white text-xs text-zinc-500">
            Loading topology map
          </div>
        )}
      >
        <TopologyGraph scenario={scenario} />
      </Suspense>
    </section>
  );
}

function Results({
  result,
  topologyScenario,
  artifact,
  artifactReplay,
}: {
  readonly result: DashboardResult;
  readonly topologyScenario?: SimulationScenario;
  readonly artifact?: DashboardArtifactDownload;
  readonly artifactReplay?: DashboardArtifactReplay;
}): React.JSX.Element {
  const metrics = result.mode === "speculative"
    ? speculativeMetrics(result)
    : result.mode === "serving"
      ? servingMetrics(result)
      : expertMetrics(result);
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-200 pb-3">
        <div className="min-w-0">
          <h1 className="truncate text-base font-bold">
            {result.model?.name ?? "Synthetic expert-cache workload"}
          </h1>
          <div className="mt-1 text-xs text-zinc-500">
            {result.model
              ? (
                  <>
                    {formatLargeCount(result.model.totalParameters)} params ·{" "}
                    {formatBytes(result.model.weightBytes)} weights ·{" "}
                    {result.scenario.id}
                  </>
                )
              : `${result.scenario.id} · no model timing bound`}
          </div>
          <div className="mt-0.5 text-[11px] text-zinc-400">
            {result.topology.planSteps.toLocaleString()} plan steps ·{" "}
            {result.topology.operationCounts.compute.toLocaleString()} compute ·{" "}
            {result.topology.operationCounts.transfer.toLocaleString()} transfer ·{" "}
            {result.topology.operationCounts.collective.toLocaleString()} collective
          </div>
        </div>
        <div className="flex items-center gap-2">
          {artifact
            ? (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label="Export verified result"
                      onClick={() => downloadArtifact(artifact)}
                    >
                      <Download className="size-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Export verified result</TooltipContent>
                </Tooltip>
              )
            : null}
          {artifactReplay
            ? (
                <Badge variant={artifactReplay.matches ? "success" : "danger"}>
                  {artifactReplay.matches
                    ? "artifact parity"
                    : "artifact mismatch"}
                </Badge>
              )
            : null}
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
          {result.serving && result.expertCache
            ? <Badge variant="neutral">stateful expert cache</Badge>
            : null}
          {result.serving?.physicalReplayEvents
            ? <Badge variant="neutral">global resource replay</Badge>
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
      {artifactReplay
        ? <ArtifactReplaySummary replay={artifactReplay} />
        : null}
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
            <div className="mt-1 min-h-8 text-xs leading-4 text-zinc-500">
              {metric.detail}
            </div>
          </div>
        ))}
      </section>
      {topologyScenario
        ? (
            <TopologyVisualizationSection
              scenario={topologyScenario}
              title="Execution topology"
            />
          )
        : null}

      <Suspense fallback={<ChartSkeleton />}>
        <ResultCharts result={result} />
      </Suspense>
    </div>
  );
}

function FrozenPlanResults({
  result,
}: {
  readonly result: FrozenPlanBrowserResult;
}): React.JSX.Element {
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 pb-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-bold text-zinc-900">
            {result.plan.id}
          </div>
          <div className="truncate text-xs text-zinc-500">
            {result.sourceFileName} · {result.scenario.id} · epoch{" "}
            {result.plan.topologyEpoch}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="success">
            <CheckCircle2 className="mr-1 size-3.5" />
            execution replay parity
          </Badge>
          <Badge variant="neutral">FrozenPlan artifact v{result.artifact.revision}</Badge>
        </div>
      </div>

      <section className="grid gap-3 border-y border-zinc-200 bg-white px-3 py-3 sm:grid-cols-[minmax(0,1fr)_auto_auto_auto] sm:items-center">
        <div className="flex min-w-0 items-center gap-3">
          <FileCheck2 className="size-4 shrink-0 text-emerald-700" />
          <div className="min-w-0">
            <div className="truncate text-xs font-bold text-zinc-800">
              {result.artifact.artifactFingerprint}
            </div>
            <div className="truncate text-[11px] text-zinc-500">
              self-contained scenario and plan
            </div>
          </div>
        </div>
        <DiagnosticValue
          label="Scenario"
          value={result.artifact.scenarioFingerprint}
        />
        <DiagnosticValue
          label="Plan"
          value={result.artifact.planFingerprint}
        />
        <DiagnosticValue
          label="Replay"
          value={`${result.replay.appliedEvents.toLocaleString()} events`}
        />
      </section>

      <section className="metric-grid" aria-label="FrozenPlan run metrics">
        <PlanMetric
          label="Modeled duration"
          value={formatDuration(result.execution.completedAtNs)}
          detail={`${result.execution.operationCount.toLocaleString()} submitted operations`}
          icon={<Clock3 className="size-4 text-amber-700" />}
        />
        <PlanMetric
          label="Plan steps"
          value={result.plan.steps.toLocaleString()}
          detail={`${result.plan.operationCounts.compute.toLocaleString()} compute · ${result.plan.operationCounts.transfer.toLocaleString()} transfer`}
          icon={<Cpu className="size-4 text-sky-700" />}
        />
        <PlanMetric
          label="Topology"
          value={`${result.scenario.devices} devices`}
          detail={`${result.scenario.ranks} ranks · ${result.scenario.links} links`}
          icon={<Network className="size-4 text-emerald-700" />}
        />
        <PlanMetric
          label="Rank terminals"
          value={result.execution.rankStates.length.toLocaleString()}
          detail={`${result.execution.rankStates.filter((rank) => rank.status === "succeeded").length} succeeded`}
          icon={<CheckCircle2 className="size-4 text-emerald-700" />}
        />
      </section>

      <section className="panel">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <h2 className="m-0 text-sm font-bold text-zinc-900">
              Operation trace
            </h2>
            <p className="mt-1 text-xs text-zinc-500">
              {result.execution.operationPreview.length.toLocaleString()} of{" "}
              {result.execution.operationCount.toLocaleString()} operations
            </p>
          </div>
          <Badge variant="neutral">{result.execution.status}</Badge>
        </div>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[620px] border-collapse text-left text-xs">
            <thead className="border-b border-zinc-200 text-zinc-500">
              <tr>
                <th className="px-2 py-2 font-semibold">Step</th>
                <th className="px-2 py-2 font-semibold">Kind</th>
                <th className="px-2 py-2 font-semibold">Start</th>
                <th className="px-2 py-2 font-semibold">Finish</th>
                <th className="px-2 py-2 font-semibold">Resources</th>
                <th className="px-2 py-2 font-semibold">Ranks</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {result.execution.operationPreview.map((operation) => (
                <tr key={operation.sourceSequence}>
                  <td className="px-2 py-2 font-semibold tabular-nums">
                    {operation.stepId}
                  </td>
                  <td className="px-2 py-2">{operation.kind}</td>
                  <td className="px-2 py-2 tabular-nums text-zinc-600">
                    {formatDuration(operation.startNs)}
                  </td>
                  <td className="px-2 py-2 tabular-nums text-zinc-600">
                    {formatDuration(operation.finishNs)}
                  </td>
                  <td className="max-w-64 truncate px-2 py-2 text-zinc-600">
                    {operation.resources.map((resource) => (
                      `${resource.resourceId}:${resource.resourceLane}`
                    )).join(", ")}
                  </td>
                  <td className="px-2 py-2 text-zinc-600">
                    {operation.participants.join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function PlanMetric({
  label,
  value,
  detail,
  icon,
}: {
  readonly label: string;
  readonly value: string;
  readonly detail: string;
  readonly icon: React.ReactNode;
}): React.JSX.Element {
  return (
    <div className="metric-card">
      <div className="flex items-center justify-between gap-2 text-zinc-500">
        <span className="text-xs font-semibold">{label}</span>
        {icon}
      </div>
      <div className="mt-2 text-xl font-bold tabular-nums">{value}</div>
      <div className="mt-1 min-h-8 text-xs leading-4 text-zinc-500">
        {detail}
      </div>
    </div>
  );
}

function ArtifactReplaySummary({
  replay,
}: {
  readonly replay: DashboardArtifactReplay;
}): React.JSX.Element {
  return (
    <section className="grid gap-3 border-y border-zinc-200 bg-white px-3 py-3 sm:grid-cols-[minmax(0,1fr)_auto_auto_auto] sm:items-center">
      <div className="flex min-w-0 items-center gap-3">
        {replay.matches
          ? <FileCheck2 className="size-4 shrink-0 text-emerald-700" />
          : <FileDiff className="size-4 shrink-0 text-rose-700" />}
        <div className="min-w-0">
          <div className="truncate text-xs font-bold text-zinc-800">
            {replay.sourceFileName}
          </div>
          <div className="truncate text-[11px] text-zinc-500">
            deterministic re-execution · current contracts
          </div>
        </div>
      </div>
      <DiagnosticValue
        label="Input"
        value={fingerprintComparison(
          replay.expectedInputFingerprint,
          replay.actualInputFingerprint,
        )}
      />
      <DiagnosticValue
        label="Output"
        value={fingerprintComparison(
          replay.expectedOutputFingerprint,
          replay.actualOutputFingerprint,
        )}
      />
      <DiagnosticValue
        label="Envelope"
        value={fingerprintComparison(
          replay.expectedArtifactFingerprint,
          replay.actualArtifactFingerprint,
        )}
      />
    </section>
  );
}

function fingerprintComparison(expected: string, actual: string): string {
  const expectedShort = expected.slice("fnv1a32:".length);
  const actualShort = actual.slice("fnv1a32:".length);
  return expected === actual
    ? `${actualShort} matched`
    : `${expectedShort} -> ${actualShort}`;
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

function downloadArtifact(artifact: DashboardArtifactDownload): void {
  const url = URL.createObjectURL(artifact.blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = artifact.fileName;
  anchor.hidden = true;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function OnnxSearchInspector({
  result,
}: {
  readonly result: OnnxSearchBrowserResult;
}): React.JSX.Element {
  const search = result.result;
  const rejected = Object.entries(search.rejectionCounts);
  const rejectedTotal = rejected.reduce(
    (sum, [, count]) => sum + count,
    0,
  );
  return (
    <div className="mx-auto max-w-md xl:max-w-none">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-bold">Search inspector</h2>
        <Badge variant="neutral">{rejectedTotal.toLocaleString()} rejected</Badge>
      </div>
      <div className="divide-y divide-zinc-200 border-y border-zinc-200">
        {rejected.map(([reason, count]) => (
          <div
            key={reason}
            className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 py-2.5 text-xs"
          >
            <span className="min-w-0 break-words capitalize text-zinc-500">
              {reason.replaceAll("_", " ")}
            </span>
            <span className="font-semibold tabular-nums text-zinc-800">
              {count.toLocaleString()}
            </span>
          </div>
        ))}
        {rejected.length === 0
          ? (
              <div className="py-4 text-center text-xs text-zinc-500">
                No rejected candidates
              </div>
            )
          : null}
      </div>
      <div className="mt-4 divide-y divide-zinc-200 border-y border-zinc-200">
        <div className="flex justify-between gap-3 py-2.5 text-xs">
          <span className="text-zinc-500">Objective</span>
          <span className="font-semibold capitalize">
            {searchObjectiveLabel(search.objective)}
          </span>
        </div>
        <div className="flex justify-between gap-3 py-2.5 text-xs">
          <span className="text-zinc-500">Candidate limit</span>
          <span className="font-semibold tabular-nums">
            {result.searchConfig.maxCandidates.toLocaleString()}
          </span>
        </div>
        <div className="flex justify-between gap-3 py-2.5 text-xs">
          <span className="text-zinc-500">Device use ceiling</span>
          <span className="font-semibold tabular-nums">
            {(result.searchConfig.maximumDeviceUsedFraction * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  );
}

function OnnxInspector({
  result,
}: {
  readonly result: OnnxStaticBrowserResult;
}): React.JSX.Element {
  const architecture = result.model.architecture;
  const rows = [
    ["Revision", String(result.manifest.revision)],
    ["Graph", result.manifest.graphName || "unnamed"],
    ["Nodes", result.manifest.nodeCount.toLocaleString()],
    ["Initializers", result.manifest.initializerCount.toLocaleString()],
    ["External files", result.manifest.externalDataFiles.toLocaleString()],
    ["Architecture", architecture.kind.toUpperCase()],
    ["Layers", architecture.numLayers.toLocaleString()],
    ["Hidden", architecture.hiddenDim.toLocaleString()],
    ["Attention heads", architecture.numHeads.toLocaleString()],
    ["KV heads", architecture.numKVHeads.toLocaleString()],
    ["Weight dtype", result.model.quantization.weights.toUpperCase()],
    ["Bottleneck", result.analysis.bottleneck.replaceAll("_", " ")],
  ] as const;
  return (
    <div className="mx-auto max-w-md xl:max-w-none">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-bold">Model inspector</h2>
        <Badge variant={result.manifest.profileReadiness.ready
          ? "success"
          : "danger"}
        >
          {result.manifest.profileReadiness.ready ? "ready" : "incomplete"}
        </Badge>
      </div>
      <div className="divide-y divide-zinc-200 border-y border-zinc-200">
        {rows.map(([label, value]) => (
          <div
            key={label}
            className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 py-2.5 text-xs"
          >
            <span className="text-zinc-500">{label}</span>
            <span className="max-w-40 truncate text-right font-semibold capitalize text-zinc-800">
              {value}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-4">
        <div className="mb-1 text-[11px] font-semibold text-zinc-500">
          Model SHA-256
        </div>
        <div className="break-all font-mono text-[10px] leading-4 text-zinc-600">
          {result.manifest.modelSha256}
        </div>
      </div>
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

function ModelPackageOverview({
  modelPackage,
  scenario,
}: {
  readonly modelPackage: ImportedModelPackage;
  readonly scenario?: SimulationScenario;
}): React.JSX.Element {
  const metrics = useMemo(
    () => summarizeModelPackage(modelPackage, scenario),
    [modelPackage, scenario],
  );
  const components = modelPackage.metadata.components.length > 0
    ? modelPackage.metadata.components
    : modelPackage.models.map((model) => ({
        id: model.fileName,
        filename: model.fileName,
        type: "model",
      }));
  return (
    <div className="space-y-5">
      <div className="border-b border-zinc-200 pb-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-lg font-bold">Model pipeline</h1>
            <p className="mt-1 text-xs text-zinc-500">
              {modelPackage.metadataFileName ?? "ONNX package without inference metadata"}
            </p>
          </div>
          <Badge variant={modelPackage.metadata.pipelineStrategy
            ? "success"
            : "neutral"}
          >
            {modelPackage.metadata.pipelineStrategy ?? "single model"}
          </Badge>
        </div>
      </div>
      <div className="grid gap-px overflow-hidden border border-zinc-200 bg-zinc-200 sm:grid-cols-2 xl:grid-cols-5">
        <OnnxMetric
          label="Package size"
          value={formatBytes(metrics.packageBytes)}
          detail={`${modelPackage.fileCount} local files`}
        />
        <OnnxMetric
          label="Weight inventory"
          value={formatBytes(metrics.weightBytes)}
          detail={`${formatLargeCount(metrics.parameterCount)} parameters`}
        />
        <OnnxMetric
          label="Graph"
          value={formatLargeCount(metrics.graphNodes)}
          detail={`${metrics.components.length} ONNX model${metrics.components.length === 1 ? "" : "s"}`}
        />
        <OnnxMetric
          label="Forward work"
          value={metrics.forwardFlopsPerToken === undefined
            ? "Unavailable"
            : formatFlops(metrics.forwardFlopsPerToken)}
          detail={metrics.components.length === 1
            ? `${metrics.completeComputeProfiles}/1 complete profile`
            : "pipeline schedule required"}
        />
        <OnnxMetric
          label="Ideal decode ceiling"
          value={metrics.bandwidthCeilingTokensPerSec === undefined
            ? "Unavailable"
            : formatRate(metrics.bandwidthCeilingTokensPerSec)}
          detail="weight-bandwidth upper bound"
        />
      </div>
      <section>
        <h2 className="mb-3 text-sm font-bold">Components</h2>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {components.map((component) => {
            const model = modelPackage.models.find((candidate) => (
              candidate.fileName === component.filename
            ));
            const componentMetrics = model === undefined
              ? undefined
              : metrics.components.find(
                (entry) => entry.fileName === model.fileName,
              );
            return (
              <div
                key={component.id}
                className="metric-card"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold">
                      {component.id}
                    </div>
                    <div className="truncate text-xs text-zinc-500">
                      {component.filename}
                    </div>
                  </div>
                  <Badge variant="neutral">{component.type}</Badge>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
                  <DiagnosticValue
                    label="Parameters"
                    value={model === undefined
                      ? "N/A"
                      : formatLargeCount(
                        model.manifest.totals.initializerElements,
                      )}
                  />
                  <DiagnosticValue
                    label="Weights"
                    value={formatBytes(
                      model?.manifest.totals.initializerLogicalBytes ?? 0,
                    )}
                  />
                  <DiagnosticValue
                    label="Forward work"
                    value={componentMetrics?.forwardFlopsPerToken === undefined
                      ? "Unavailable"
                      : formatFlops(componentMetrics.forwardFlopsPerToken)}
                  />
                  <DiagnosticValue
                    label="ONNX graph"
                    value={model === undefined
                      ? "N/A"
                      : `${model.manifest.graph.nodeCount.toLocaleString()} nodes`}
                  />
                  <DiagnosticValue
                    label="BW ceiling"
                    value={componentMetrics?.activeWeightBytesPerToken === undefined
                      || metrics.hotMemoryBandwidthBytesPerSec === 0
                      ? "Unavailable"
                      : formatRate(
                        metrics.hotMemoryBandwidthBytesPerSec
                          / componentMetrics.activeWeightBytesPerToken,
                      )}
                  />
                  <DiagnosticValue
                    label="Operator kinds"
                    value={componentMetrics === undefined
                      ? "N/A"
                      : componentMetrics.operatorKinds.toLocaleString()}
                  />
                </div>
                {model === undefined
                  ? null
                  : <ComponentArchitecture metrics={componentMetrics} />}
              </div>
            );
          })}
        </div>
      </section>
      <section className="border-y border-zinc-200 py-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-bold">Theoretical speed boundary</h2>
          <Badge variant="warning">ideal upper bound</Badge>
        </div>
        <div className="mt-3 grid gap-4 sm:grid-cols-3">
          <DiagnosticValue
            label="Hot-memory bandwidth"
            value={metrics.hotMemoryBandwidthBytesPerSec === 0
              ? "Unavailable"
              : formatBandwidth(metrics.hotMemoryBandwidthBytesPerSec)}
          />
          <DiagnosticValue
            label="Active weight stream"
            value={metrics.activeWeightBytesPerToken === undefined
              ? metrics.components.length > 1
                ? "Pipeline schedule required"
                : "Unavailable"
              : `${formatBytes(metrics.activeWeightBytesPerToken)} / token`}
          />
          <DiagnosticValue
            label="Weight-streaming ceiling"
            value={metrics.bandwidthCeilingTokensPerSec === undefined
              ? "Unavailable"
              : formatRate(metrics.bandwidthCeilingTokensPerSec)}
          />
        </div>
        <p className="mt-3 text-[11px] leading-5 text-zinc-500">
          This is a batch-1 algebraic ceiling from declared hot-memory
          bandwidth divided by active model weights. It assumes perfect
          sharding and ignores compute, KV, activations, communication,
          scheduling, and kernel overhead. Runtime timing binds the
          architecture-derived active attention and FFN weight streams, but
          does not claim per-operator kernel calibration.
          Multi-model packages report ceilings per component until an execution
          schedule defines component invocation rates.
        </p>
      </section>
      <section className="border-t border-zinc-200 pt-4">
        <h2 className="mb-3 text-sm font-bold">Dataflow</h2>
        {modelPackage.metadata.edges.length === 0
          ? (
              <div className="text-xs text-zinc-500">
                No inter-component edges declared.
              </div>
            )
          : (
              <div className="divide-y divide-zinc-200 border-y border-zinc-200">
                {modelPackage.metadata.edges.map((edge) => (
                  <div
                    key={`${edge.from}->${edge.to}`}
                    className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-3 py-3 text-xs"
                  >
                    <span className="truncate font-medium">{edge.from}</span>
                    <span className="text-zinc-400">→</span>
                    <span className="truncate font-medium">{edge.to}</span>
                    <span className="col-span-3 text-[11px] text-zinc-500">
                      {edge.dtype ?? "dtype unspecified"}
                      {edge.deviceTransfer === true
                        ? " · device transfer"
                        : edge.deviceTransfer === false
                          ? " · same-device preferred"
                          : ""}
                    </span>
                  </div>
                ))}
              </div>
            )}
      </section>
    </div>
  );
}

function ComponentArchitecture({
  metrics,
}: {
  readonly metrics: ReturnType<
    typeof summarizeModelPackage
  >["components"][number] | undefined;
}): React.JSX.Element | null {
  if (metrics === undefined) {
    return null;
  }
  const architecture = metrics.architecture;
  const details = architecture === undefined
    ? ["Architecture metadata incomplete"]
    : [
        `${architecture.layers} layers`,
        `hidden ${architecture.hiddenSize.toLocaleString()}`,
        `${architecture.attentionHeads} attention / ${architecture.kvHeads} KV heads`,
        ...(architecture.experts === undefined
          ? []
          : [`${architecture.activeExperts}/${architecture.experts} active experts`]),
      ];
  return (
    <div className="mt-3 border-t border-zinc-200 pt-2">
      <div className="text-[11px] leading-5 text-zinc-500">
        {details.join(" · ")}
      </div>
      <div className="mt-1 truncate text-[11px] text-zinc-500">
        {metrics.topOperators.length === 0
          ? "No graph operators"
          : `Top operators: ${metrics.topOperators.join(" · ")}`}
      </div>
    </div>
  );
}

function ModelPackageInspector({
  modelPackage,
}: {
  readonly modelPackage: ImportedModelPackage;
}): React.JSX.Element {
  const families = modelPackage.metadata.speculative.availableFamilies;
  return (
    <div>
      <h2 className="text-sm font-bold">Package evidence</h2>
      <div className="mt-4 space-y-4">
        <InspectorRow label="Files" value={String(modelPackage.fileCount)} />
        <InspectorRow
          label="Package size"
          value={formatBytes(modelPackage.packageByteLength)}
        />
        <InspectorRow
          label="ONNX models"
          value={String(modelPackage.models.length)}
        />
        <InspectorRow
          label="Speculative"
          value={families.length > 0 ? families.join(", ") : "target only"}
        />
        {modelPackage.models.map((model) => (
          <div key={model.fileName} className="border-t border-zinc-200 pt-3">
            <div className="truncate text-xs font-semibold">
              {model.fileName}
            </div>
            <div className="mt-1 truncate text-[11px] text-zinc-500">
              {model.manifest.manifestFingerprint}
            </div>
            <div className="mt-1 text-[11px] text-zinc-500">
              {model.manifest.externalDataFiles.length} external files
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function InspectorRow({
  label,
  value,
}: {
  readonly label: string;
  readonly value: string;
}): React.JSX.Element {
  return (
    <div className="flex items-start justify-between gap-3 text-xs">
      <span className="text-zinc-500">{label}</span>
      <span className="min-w-0 text-right font-medium break-words">{value}</span>
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
  if (status === "artifact-mismatch") {
    return (
      <Badge variant="danger">
        <FileDiff className="mr-1 size-3.5" />
        Artifact mismatch
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
      detail: `${metrics.demandLoads} demand loads · ${metrics.adaptivePrefetchSelections} adaptive prefetches`,
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
      label: "Aggregate throughput",
      value: formatRate(metrics.throughputTokensPerSecond),
      detail: speculative
        ? `${metrics.committedTokensPerTargetForward.toFixed(2)} tokens / target`
        : `${metrics.requests} requests · ${(metrics.tokenBatchUtilization * 100).toFixed(1)}% token slots`,
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
  const magnitude = Math.abs(bytes);
  if (magnitude >= 1024 ** 3) {
    return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
  }
  if (magnitude >= 1024 ** 2) {
    return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  }
  if (magnitude >= 1024) {
    return `${(bytes / 1024).toFixed(1)} KiB`;
  }
  return `${bytes.toFixed(0)} B`;
}

function formatSignedBytes(bytes: number): string {
  return bytes < 0 ? `-${formatBytes(-bytes)}` : formatBytes(bytes);
}

function formatDuration(nanoseconds: number): string {
  if (nanoseconds >= 1_000_000) {
    return `${(nanoseconds / 1_000_000).toFixed(1)} ms`;
  }
  return `${Math.round(nanoseconds / 1_000)} us`;
}

function formatRate(tokensPerSecond: number): string {
  const value = tokensPerSecond >= 1_000_000_000
    ? `${(tokensPerSecond / 1_000_000_000).toFixed(1)}B`
    : tokensPerSecond >= 1_000_000
      ? `${(tokensPerSecond / 1_000_000).toFixed(1)}M`
      : tokensPerSecond >= 1000
        ? tokensPerSecond.toLocaleString(undefined, {
            maximumFractionDigits: 0,
          })
        : tokensPerSecond.toFixed(1);
  return `${value} tok/s`;
}

function formatLargeCount(value: number): string {
  if (value >= 1_000_000_000_000) {
    return `${(value / 1_000_000_000_000).toFixed(2)}T`;
  }
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(2)}B`;
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toLocaleString();
}

function formatFlops(flops: number): string {
  if (flops >= 1e15) {
    return `${(flops / 1e15).toFixed(2)} PFLOP`;
  }
  if (flops >= 1e12) {
    return `${(flops / 1e12).toFixed(2)} TFLOP`;
  }
  if (flops >= 1e9) {
    return `${(flops / 1e9).toFixed(2)} GFLOP`;
  }
  if (flops >= 1e6) {
    return `${(flops / 1e6).toFixed(2)} MFLOP`;
  }
  return `${flops.toLocaleString(undefined, {
    maximumFractionDigits: 0,
  })} FLOP`;
}

function formatBandwidth(bytesPerSecond: number): string {
  return `${(bytesPerSecond / 1e12).toLocaleString(undefined, {
    maximumFractionDigits: 2,
  })} TB/s`;
}
