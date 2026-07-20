# inference-sim

A discrete-event simulator for LLM inference resource allocation and scheduling.

Models hardware topologies, ONNX model profiles, and memory management policies to predict throughput, latency, memory flow, and bottlenecks — **without real hardware**.

## What it does

- **Static analysis**: Memory breakdown, feasibility check, bottleneck identification
- **Expert cache simulation**: Hit rates, prefetch effectiveness, eviction patterns
- **Governor simulation**: Pressure protocol behavior, resource contention
- **Visualization**: Memory timeline, expert heatmap, topology data flow

## Packages

| Package | Description |
|---------|-------------|
| `@inference-sim/core` | Pure computation engine, zero dependencies, runs in Node + Browser |
| `@inference-sim/cli` | Command-line interface with YAML config input |
| `@inference-sim/web` | React visualization dashboard |

## Quick Start

```bash
pnpm install
pnpm build

# Static memory analysis
pnpm sim --hardware h100-8x --model mixtral-8x22b --quantization fp8

# Full simulation
pnpm sim run --config examples/mixtral-dgx.yaml
```

## Design

See [docs/DESIGN.md](docs/DESIGN.md) for the full architecture and design decisions.

## License

MIT
