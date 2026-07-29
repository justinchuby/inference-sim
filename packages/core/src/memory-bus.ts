/**
 * How a machine's main memory bandwidth is arrived at.
 *
 * Model geometry is checked against a published parameter count, so a preset
 * whose numbers drift is caught by arithmetic rather than by a reader. Machine
 * specifications had no equivalent: a bandwidth was a bare figure that could
 * be mistyped, or silently left behind when a preset was copied, and nothing
 * would notice.
 *
 * Peak bandwidth is not a free parameter. It is the transfer rate times the
 * bus width, so declaring those two alongside the vendor's figure lets the
 * product check it, exactly as a derived parameter count checks a published
 * one.
 */
export interface MemoryBusSpec {
  /** Vendor name for the memory, for example `LPDDR5X-9600`. */
  readonly label: string;
  /** Transfer rate in megatransfers per second. */
  readonly transferMtPerSec: number;
  /** Total bus width in bits, summed across channels. */
  readonly busWidthBits: number;
}

/** Peak bytes per second the declared bus can move. */
export function derivedMemoryBandwidth(spec: MemoryBusSpec): number {
  return spec.transferMtPerSec * 1e6 * (spec.busWidthBits / 8);
}
