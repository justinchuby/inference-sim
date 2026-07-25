/**
 * Deterministic ordering helpers.
 *
 * `String.prototype.localeCompare` orders by the host's locale and ICU build,
 * so the same identifiers can sort differently on different machines. Every
 * canonical ordering in the simulator — scheduling, trace emission, artifact
 * serialization — must use a code-unit comparison instead so a run is
 * reproducible across platforms.
 */

/** Total, locale-independent order over identifier strings. */
export function compareIds(left: string, right: string): number {
  if (left === right) {
    return 0;
  }
  return left < right ? -1 : 1;
}
