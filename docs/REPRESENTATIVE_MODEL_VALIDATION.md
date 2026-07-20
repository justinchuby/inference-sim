# Representative Model Validation

**Status:** architecture audit, 2026-07-20

This audit asks whether a selected model changes the executable simulation, not
merely whether its metadata can be displayed. A model family passes only when
its component invocations, weight capacity, placement constraints, and required
communication are bound to replayable workload evidence.

## Validation Matrix

| Representative family | Required behavior | Current result |
|---|---|---|
| Llama 3 dense decoder | Bind target weights and active attention/FFN work to decode execution and capacity | **Complete for the declared target model.** Kernel timing remains heuristic without calibration. |
| Gemma 4 / Qwen-VL composite | Run prompt-only vision/embedding or projector components, capacity-check their weights, place each component, and reserve declared dataflow transfers | **Partial.** Metadata, phases, components, and edges are preserved; only the target decoder is scheduled. |
| Mixtral / DeepSeek MoE | Derive expert count and top-k from the model, bind EP ownership and All-to-All traffic, and compose TP/EP according to the selected topology | **Partial.** Active expert weight traffic affects target timing, but the model does not configure the expert workload. |
| Target plus independent draft model | Capacity-check and place both models; derive proposer work from the imported draft profile; keep target verification separate | **Partial.** The family and target are bound, but draft execution still uses a family heuristic and ignores imported draft weights. |

These are deliberately adversarial checks. Parsing a valid
`inference_metadata.yaml`, rendering a pipeline graph, or producing generic
expert/speculative events does not satisfy the corresponding row.

## Executable Guards

The web test suite constructs small ONNX manifests for each architecture shape
and asserts the execution coverage embedded in `DashboardModelBinding`:

- dense single-target packages report `complete/full_model`;
- composite VLM packages report exactly which prompt components are unmodeled;
- independent draft packages disclose that proposer cost is not model-bound;
- representative MoE presets disclose that EP routing is not model-bound; and
- component phase gates such as `prompt_only` and `every_step` survive metadata
  normalization for a future pipeline scheduler.

Coverage is part of the deterministic dashboard input. Artifact replay cannot
silently upgrade a target-only simulation into a full-pipeline claim.

## Required Next Implementation

1. Compile metadata stages into per-request component invocations. Prompt-only
   stages run once per request; every-step stages run at their declared rate.
2. Add component execution profiles and capacity charges instead of one target
   profile. Compile dataflow edges into transfers when placements differ.
3. Make component placement an explicit user-editable mapping constrained by
   device capability, resource limits, and metadata preferences.
4. Bind model MoE dimensions to routed-expert workload generation. Treat manual
   expert-count/top-k changes as explicit overrides recorded in the artifact.
5. Bind an independent draft component to draft placement, memory, and
   architecture-derived work. Do not use the generic proposer coefficient when
   a complete draft profile is available.
6. Enforce metadata hardware requirements, including dtype support and minimum
   useful TP degree, during scenario validation.

Until these items land, throughput from a multi-model package means target
decoder throughput only. It is not end-to-end multimodal latency, model-bound
expert-parallel throughput, or model-bound draft speculative throughput.
