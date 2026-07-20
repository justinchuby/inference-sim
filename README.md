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
  collective fallback timing; and
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
The dashboard is React-based and extends the existing shadcn/Radix component
layer. Its expert-placement selector and the CLI's `placement_strategy` feed
the same contiguous/round-robin core contract; neither product surface
reimplements owner semantics. Cache slot controls are per EP owner for hot
residency and per node for warm residency; result snapshots expose both
aggregate and partition-level capacity evidence.
The Multi-GPU topology selector can instantiate validated 2-, 4-, or 8-rank
rings through the public core scenario builder. Each rank owns independent
VRAM, cache, workspace, PCIe, and ring-link resources; the Worker rejects
unrecognized rank counts instead of coercing them.
The speculative controls use the shared core family contract; design-only
self-speculative results are labeled explicitly. The workbench can import the
same revisioned calibration YAML/JSON used by the CLI, validates and fits it
before execution, reruns the fit inside the Worker, and displays its stable
fingerprint plus compute and transport NRMSE/P95 diagnostics.
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
communication curves replace declared link bandwidth as timing evidence.
Calibration revision 2 scopes communication curves to an exact scenario,
ordered directed-link path, participant count, algorithm, and observed byte
range. Imported calibration never silently falls back to topology bandwidth or
extrapolates beyond that range.

CLI commands that take one scenario accept either a listed preset or
`multi-gpu-ring-N` for `N=2..64`. `compare` and `serving-compare` intentionally
retain the six fixed topology families so a parameterized target does not
silently change the comparison population.

See [docs/DESIGN.md](docs/DESIGN.md) for contracts, scope, confidence classes,
device semantics, speculative execution, and delivery gates.
See [docs/ONNX_GENAI_CAPTURE.md](docs/ONNX_GENAI_CAPTURE.md) for producing and
verifying paired runtime evidence from the onnx-genai integration branch.

## License

MIT
