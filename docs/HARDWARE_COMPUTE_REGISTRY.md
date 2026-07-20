# Hardware compute registry

`packages/core/src/hardware-compute-profiles.ts` is the authoritative catalog of
published compute ceilings used by the simulator. Revision 1 covers major
inference-relevant NVIDIA, AMD, Intel, Qualcomm, and Apple products released
from 2022 onward.

## Evidence rules

- Every profile has an official vendor source and access date.
- Peaks preserve dtype, execution engine, accumulator precision, and the
  vendor's dense, structured-sparse, or unspecified semantics.
- Dense and sparse figures are separate records. Sparse figures are never the
  default roof.
- Generic NPU or platform "AI TOPS" with no dtype is stored as `vendor_ai` and
  is display-only.
- A vendor peak is theoretical throughput, not measured model throughput.
- Weight-only INT4/INT2/INT1 does not imply pure integer execution when the
  activation remains FP16 or BF16. Roofline selection therefore follows the
  activation dtype unless a future profile declares a compatible mixed mode.
- Missing disclosure remains missing. Apple, for example, does not publish
  absolute dtype-specific GPU peaks for current M-series products.

The catalog intentionally does not derive chip peaks from core counts, clock
rates, relative benchmark claims, aggregate CPU+GPU+NPU TOPS, or third-party
databases. Additions must include a primary source and a testable arithmetic
semantic.

## Coverage

The initial catalog includes Hopper, Ada, RTX Blackwell, CDNA 3/4, RDNA 3/4,
Gaudi 2/3, representative Intel Core Ultra NPUs, Qualcomm Cloud AI and Hexagon
NPUs, and Apple M-series disclosures. It is a curated family/SKU catalog, not a
claim that every regional or clock-bin variant is enumerated.
