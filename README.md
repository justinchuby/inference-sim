# inference-sim

A deterministic simulator for LLM inference placement, memory protocols, and
resource scheduling.

The project is intentionally split between exact protocol checks and calibrated
performance estimates. It can prove ledger and state-transition properties; it
does not claim hardware-accurate latency without calibration data.

## Current Status

Phase 1 is complete in `@inference-sim/core`:

- heuristic static memory and roofline analysis;
- deterministic discrete-event scheduling;
- exact host-pressure ticket and allocation accounting;
- contract-revisioned pressure traces;
- independent trace replay with invariant checks;
- composite speculative checkpoint/restore transaction semantics.

Phase 2 is complete:

- capability-based scenarios for all six required topology families;
- exact physical memory-domain ledgers and shared-allocation aliases;
- FrozenPlan DAG validation and deterministic resource scheduling;
- collective ordering, transport-link contention, and allocation leases; and
- independent plan-trace replay with rank-local success/failure/abort
  quiescence; and
- workload-to-FrozenPlan compilation with device compute, directed transfers,
  algorithm-labeled bidirectional tensor collectives, capability-overlap
  expert dispatch/gather, and topology-aware heuristic timing; and
- time-anchored device/link/topology-epoch fault injection, submission closure,
  quiescence, independent replay, and deterministic per-resource campaigns.
- seeded multi-execution FrozenPlan campaigns with arrival-ordered admission,
  shared compute/link/collective lanes, communicator ownership, physical
  allocation leases, and independent global replay.
- structured node failures and explicit quiesce-before-admit failover to a
  failed-node-free scenario and replanned workload at a newer topology epoch.
- atomic node-fault fanout across all active old-epoch concurrent executions,
  with one shared pre-fault schedule prefix and independent global replay.

Phase 3 has an initial speculative workload slice:

- revisioned family contracts for prompt lookup, draft model, MTP, EAGLE-3,
  shared-KV, and design-only self-speculative execution;
- fail-closed eligibility for proposer availability, target KV, grammar,
  decoding mode, hidden outputs, shared-KV groups, and early-exit layers;
- committed-prefix, proposal-local, and borrowed state lifetimes instead of
  generic proposer KV;
- accepted-prefix replay or seeded conditional first-mismatch acceptance;
- exact output-budget and target-only final-length parity;
- family-specific guaranteed-prefix, additional-draft, accepted-tail,
  correction/bonus, rejection, and efficiency metrics; and
- provenance-bound token-value differential traces with independent
  token-decision/state-decision replay;
- independently bound target-only/speculative runtime capture artifacts with
  terminal-count, provenance, offset, and runtime-commit verification;
- composite target/proposer state rollback on every iteration;
- exact-capacity paged KV allocation with checkpoint-relative rollback,
  non-reused physical page identities, and independent trace replay; and
- speculative KV high-water, allocation, release, and final-reservation
  metrics.
- variable-size, byte-capacity hot/warm expert caches with weighted routing
  without replacement, deterministic LRU eviction, asynchronous initial plus
  history-driven adaptive warm prefetch, EP-owner hot partitions, node-local
  warm partitions, physical cold-storage link contention, and independent
  trace replay; and
- routed expert execution with TP attention, `all_reduce_ring`,
  `all_to_all_v` dispatch/gather, explicit contiguous/round-robin expert
  ownership, owner-only demand/prefetch transfers, and route-skewed owner-local
  FFN work on arbitrary-rank topologies, with phase-aware ring/pairwise
  collective timing, explicit round-robin token-source to expert-owner traffic,
  and deterministic message-size-aware physical route selection; and
- speculative and expert-cache traces compiled onto all six topology families
  with replay-verified resource utilization and relative comparisons; and
- a six-proposer by six-device-topology execution/replay matrix with
  target-only final-state differential checks;
- arrival-driven, decode-first continuous batching with same-time dispatch
  barriers, chunked prefill, priority ordering, and exact KV admission; and
- independent serving-trace replay plus TTFT/ITL/latency/utilization metrics
  across all six topology presets; and
- speculative serving with per-request acceptance streams, transactional
  target/proposer state, transient candidate-KV admission, burst emission, and
  a six-proposer by six-topology execution matrix; and
- deterministic serving comparison across all six device configurations with
  ranked latency, throughput, TTFT/ITL, KV, and replay evidence; and
- revisioned topology-cost calibration import with repeated observations,
  provenance, quality diagnostics, scoped applicability, stable fingerprints,
  exact-path transfer/collective curves, and fail-closed interpolation ranges.

The initial CLI and browser dashboard are implemented. The dashboard runs core
simulation in a cancellable Web Worker and exposes topology selection,
continuous-serving, speculative, and expert-cache controls, including
target-only versus proposer-family serving, modeled latency/throughput, request
TTFT/ITL, memory, resource-utilization and caching charts, and a recent event
inspector. Serving can run one selected topology or compare all six in one
replay-verified view.
The CLI also exports a compiled workload as a self-contained, revisioned
FrozenPlan artifact and executes that artifact without consulting mutable
preset definitions. Scenario, plan, and whole-envelope fingerprints fail
closed before semantic validation; successful execution is independently
replayed from the emitted plan trace.
The `onnx-inspect` command decodes a standard ONNX protobuf and emits a
revisioned model manifest. It inventories graph operators and initializers,
validates external-data paths and extents inside the model package, streams
SHA-256 over every referenced sidecar, and normalizes optional onnx-genai
`manifest.json`, `genai_config.json`, or inference metadata. Missing
architecture evidence remains explicit; the importer does not infer model
semantics from tensor names.
`onnx-static` consumes that package directly with a hardware/pipeline config,
preserves exact initializer capacity, records all profile assumptions, and
then runs the shared static analyzer. MoE packages remain incomplete until
metadata supplies routed and shared expert bytes per layer.
The React workbench imports the same revision-2 manifest, validates it before
and inside the Worker, and exposes shadcn controls for hardware, runtime dtypes,
sequence shape, TP/PP/EP, and offload. Its model view reports per-device memory,
capacity feasibility, exact package/weight/parameter inventory, architecture
and operator summaries, heuristic forward FLOPs per token, modeled throughput,
and every profile assumption. It also separates ideal compute and
weight-bandwidth roofline ceilings from utilization-adjusted planning
estimates.
The serving workbench can also import a local model folder or a selected set
of package files without uploading them. A dedicated browser Worker decodes
each ONNX protobuf, incrementally hashes external-data sidecars, parses
`inference_metadata.yaml|json`, validates multi-model component and dataflow
references, and displays the resulting pipeline. Imported model fingerprints
and declared speculative families are bound into deterministic run input. The
selected target model's name, parameter count, weight bytes, active
attention/FFN streams, and approximate forward FLOPs are also bound into the
Worker input and result artifact. Serving and speculative plans apply a
per-invocation weight-bandwidth floor from the actual target memory domain;
they no longer time every model with one normalized token constant.
Target-only is the only decode mode enabled when metadata does not provide
specific speculative evidence; unknown proposal methods remain visible as
diagnostics and are never coerced to a similar simulator family.
The imported package view shows each model's size, parameter count,
architecture, leading ONNX operators, and estimated forward work. Its selected
topology speed boundary is explicitly a batch-1 weight-streaming upper bound:
only model-capable devices contribute hot-memory bandwidth, and the UI states
that compute, KV, communication, scheduling, and kernel overhead are excluded.
The dashboard does not present this bound as measured or simulated throughput.
`onnx-search` exhaustively ranks a bounded, explicitly declared static search
space across topology, runtime dtype, batch, sequence shape, TP/PP/EP/DP, and
offload. It reports declared, evaluated, eligible, returned, and per-reason
rejected counts with stable candidate IDs. The result is an evidence-labeled
ranking of that finite space, never an unexplained claim of a global optimum.
The ONNX workbench exposes the same contract through Analyze/Search modes.
Search scope controls expand only visible finite axes, show the declared
candidate count before execution, and render top-K candidates beside complete
structural and constraint rejection counts.
Completed dashboard artifacts are retained in a bounded IndexedDB history:
fingerprints deduplicate entries, least-recently-opened records are evicted
past 20 entries or 256 MiB, and replay re-runs the strict artifact parser
before Worker execution. UI timestamps are storage metadata only and never
enter deterministic simulation evidence.
The dashboard is React-based and extends the existing shadcn/Radix component
layer rather than duplicating UI primitives. Worker progress is tied to actual
validation, execution, replay, search, and artifact boundaries and remains
outside deterministic evidence. The topology selector accepts a bounded local
revision-5 scenario YAML/JSON file, validates it before selection, repeats the
same strict parse in the Worker, and embeds the complete scenario in exported
run evidence. The local filename is UI metadata only. Built-in and imported
topologies can be opened in a responsive editor that exposes every device's
execution provider, compute concurrency, dtypes, capabilities, accessible
memory capacity/bandwidth/latency/coherence, and every directed link's
endpoints, kind, bandwidth, latency, and lanes. Its Resources view preserves
installed capacity while setting independent per-domain allocation limits for
VRAM, RAM, unified memory, and SSD. SSD streaming is an explicit execution
feature: disabling it removes storage from the allocatable ledger and rejects
cold expert loads or storage prefetches rather than silently falling back.
Applying an edit marks the
scenario custom, advances `topologyEpoch`, replaces topology provenance with
explicit user-edited heuristic evidence, and validates the complete scenario
through the shared core parser before it can run. The selected topology is
also rendered as an interactive pan/zoom map in the configuration, editor, and
result views. The map has a physical-system parent for every `nodeId`, with
compute chips and memory domains contained beneath it; links are classified as
intra-node or inter-node. Device-to-memory access is shown separately from
directed transport links, and selecting a node or edge reveals the exact
capability or link evidence used by the scenario. The map is a deterministic, read-only
projection of the same scenario object; topology changes remain explicit
through the Devices and Links editor tabs rather than creating a second graph
state. Its expert-placement
selector and the CLI's
`placement_strategy` feed
the same contiguous/round-robin core contract; neither product surface
reimplements owner semantics. Cache slot controls are per EP owner for hot
residency and per node for warm residency; result snapshots expose both
aggregate and partition-level capacity evidence.
The Multi-GPU topology selector can instantiate validated 2-, 4-, or 8-rank
rings through the public core scenario builder. Each rank owns independent
VRAM, cache, workspace, PCIe, and ring-link resources; the Worker rejects
unrecognized rank counts instead of coercing them.
The speculative controls use the shared core family contract; design-only
self-speculative results are labeled explicitly. With no imported model these
controls are an unbound design exploration; once a local package is bound,
both the UI and Worker enforce its declared proposer families. The workbench
can import the
same revisioned calibration YAML/JSON used by the CLI, validates and fits it
before execution, reruns the fit inside the Worker, and displays its stable
fingerprint plus compute and transport NRMSE/P95 diagnostics.
Every completed browser run can export a revision-1 deterministic result
artifact. The artifact binds the dashboard input, summary, complete
mode-specific core evidence, and only the contract revisions used by that run
with canonical fingerprints. Browser wall-clock execution time is intentionally
excluded, so identical simulation input and evidence serialize identically.
The same JSON can be imported back into the workbench: current contracts and
embedded inputs are validated, the Worker re-executes the run, and input,
output, and envelope fingerprints are compared explicitly.
The Spec view primarily imports a pair of independently emitted target-only and
speculative runtime captures. It binds their provenance, reconstructs and
checks every speculative commit, then derives the revisioned token trace used
by the Worker. A preassembled token-trace YAML/JSON remains available as a
secondary debugging and compatibility path. Both paths preview structural and
value parity, rerun the oracle and composite state replay inside the Worker,
and surface parity or the first target-only token mismatch without converting a
well-formed mismatch into a generic execution failure.

## Direction

The simulation model is being built around the onnx-genai memory and
distributed-runtime contracts. Planned scenarios cover:

- CPU-only;
- discrete GPU plus CPU;
- multi-GPU;
- GPU plus NPU;
- unified memory; and
- multi-node and heterogeneous execution.

Speculative decoding is a first-class workload model. It covers
prompt-lookup, draft-model, MTP, EAGLE-3, shared-KV, and self-speculative
proposers with target-authoritative verification and composite checkpoint
restore. Self-speculative is modeled as a design projection because the current
onnx-genai runtime does not expose it as a released proposer mode.

## Development

```bash
pnpm install --frozen-lockfile
pnpm build
pnpm test

pnpm sim presets
pnpm sim scenario gpu-npu
pnpm sim scenario multi-gpu-ring-4
pnpm sim scenario /path/to/custom-scenario.yaml
pnpm sim static examples/mixtral-dgx-h100.yaml
pnpm sim speculative examples/speculative-mtp.yaml
pnpm sim speculative-trace examples/speculative-token-trace-mtp.yaml single-gpu-cpu
pnpm sim speculative-capture examples/runtime-capture-target-only.yaml examples/runtime-capture-speculative.yaml single-gpu-cpu
pnpm sim expert-cache examples/expert-cache.yaml
pnpm sim serving multi-gpu examples/serving.yaml
pnpm sim serving multi-gpu examples/serving-speculative.yaml
pnpm sim serving multi-gpu examples/serving-speculative-experts.yaml
pnpm sim serving multi-gpu-ring-4 examples/serving-speculative.yaml
pnpm sim serving-compare examples/serving-speculative.yaml
pnpm sim calibrate examples/calibration-synthetic.yaml
pnpm sim serving-compare examples/serving-speculative.yaml examples/calibration-synthetic.yaml
pnpm sim run multi-gpu examples/target-only.yaml
pnpm sim run multi-gpu-ring-8 examples/target-only.yaml
pnpm sim run /path/to/custom-scenario.yaml examples/target-only.yaml
pnpm sim compare examples/target-only.yaml
pnpm sim fault-campaign multi-gpu examples/target-only.yaml
pnpm sim concurrent-campaign multi-gpu examples/concurrent-campaign.yaml
pnpm sim concurrent-node-failure multi-node examples/concurrent-node-failure.yaml
pnpm sim node-failover multi-node single-gpu-cpu examples/node-failover.yaml
pnpm dev:web
```

`run`, `compare`, `serving`, `serving-compare`, and `fault-campaign` accept an
optional final calibration path; without one they use the bundled heuristic
cost model. The included calibration file is explicitly synthetic and remains
heuristic. A measured compute dataset does not by itself upgrade results from
the heuristic built-in topology presets: end-to-end timing uses the weakest
confidence among the performance inputs actually used. Exact-path
communication curves replace declared link performance only for the selected
path's duration.
Calibration revision 3 scopes communication curves to an exact scenario,
ordered directed-link path, participant count, algorithm, optional canonical
AllToAllV traffic signature, and observed byte range. Imported calibration
never silently falls back to topology bandwidth or extrapolates beyond that
range. Declared link latency and bandwidth still select the physical path for
each message size, so their provenance remains part of end-to-end confidence
even when an exact-path curve supplies the selected path's duration.
AllToAllV signatures use the GCD-reduced source-rank to expert-owner matrix, so
proportional message sizes share a curve while different skew shapes do not.

CLI commands that take one scenario accept a listed preset,
`multi-gpu-ring-N` for `N=2..64`, or a revision-5 scenario YAML/JSON file.
Custom files pass the same strict unknown-field, enum, safe-integer, topology,
placement, route, memory-ledger, and parallelism validation used by embedded
FrozenPlan scenarios before any workload executes. `compare` and
`serving-compare` intentionally retain the six fixed topology families so a
custom target does not silently change the comparison population.

See [docs/DESIGN.md](docs/DESIGN.md) for contracts, scope, confidence classes,
device semantics, speculative execution, and delivery gates.
See [docs/ONNX_GENAI_CAPTURE.md](docs/ONNX_GENAI_CAPTURE.md) for producing and
verifying paired runtime evidence from the onnx-genai integration branch.

Inspect an ONNX model package:

```bash
pnpm sim onnx-inspect \
  /path/to/model.onnx \
  /path/to/manifest.json
```

## License

MIT
