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

Phase 2 is in progress:

- capability-based scenarios for all six required topology families;
- exact physical memory-domain ledgers and shared-allocation aliases;
- FrozenPlan DAG validation and deterministic resource scheduling;
- collective ordering, transport-link contention, and allocation leases; and
- independent plan-trace replay with rank-local success/failure/abort
  quiescence; and
- workload-to-FrozenPlan compilation with device compute, directed transfers,
  tensor collectives, and topology-aware heuristic timing; and
- time-anchored device/link/topology-epoch fault injection, submission closure,
  quiescence, independent replay, and deterministic per-resource campaigns.

Phase 3 has an initial speculative workload slice:

- revisioned family contracts for prompt lookup, draft model, MTP, EAGLE-3,
  shared-KV, and design-only self-speculative execution;
- fail-closed eligibility for proposer availability, target KV, grammar,
  decoding mode, hidden outputs, shared-KV groups, and early-exit layers;
- committed-prefix, proposal-local, and borrowed state lifetimes instead of
  generic proposer KV;
- token-trace replay or seeded conditional first-mismatch acceptance;
- exact output-budget and target-only final-length parity;
- accepted-prefix, correction/bonus, rejection, and efficiency metrics; and
- composite target/proposer state rollback on every iteration;
- exact-capacity paged KV allocation with checkpoint-relative rollback,
  non-reused physical page identities, and independent trace replay; and
- speculative KV high-water, allocation, release, and final-reservation
  metrics.
- byte-capacity hot/warm expert caches with weighted routing without
  replacement, deterministic LRU eviction, asynchronous prefetch, and
  independent trace replay; and
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
  a six-proposer by six-topology execution matrix.

The initial CLI and browser dashboard are implemented. The dashboard runs core
simulation in a cancellable Web Worker and exposes topology selection,
continuous-serving, speculative, and expert-cache controls, including
target-only versus proposer-family serving, modeled latency/throughput, request
TTFT/ITL, memory, resource-utilization and caching charts, and a recent event
inspector.
The speculative controls use the shared core family contract; design-only
self-speculative results are labeled explicitly.

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
pnpm sim static examples/mixtral-dgx-h100.yaml
pnpm sim speculative examples/speculative-mtp.yaml
pnpm sim expert-cache examples/expert-cache.yaml
pnpm sim serving multi-gpu examples/serving.yaml
pnpm sim serving multi-gpu examples/serving-speculative.yaml
pnpm sim run multi-gpu examples/target-only.yaml
pnpm sim compare examples/target-only.yaml
pnpm sim fault-campaign multi-gpu examples/target-only.yaml
pnpm dev:web
```

`run` and `compare` use the bundled heuristic cost model. Their output carries
the confidence class and assumptions; calibrate the coefficients against
backend traces before treating results as hardware predictions.

See [docs/DESIGN.md](docs/DESIGN.md) for contracts, scope, confidence classes,
device semantics, speculative execution, and delivery gates.

## License

MIT
