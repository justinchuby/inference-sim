export type LogDomain = readonly [number, number];

export function zoomLogDomain(
  domain: LogDomain,
  factor: number,
  anchor = 0.5,
): LogDomain {
  const [minimum, maximum] = validateDomain(domain);
  const boundedFactor = Math.min(8, Math.max(0.125, factor));
  const boundedAnchor = Math.min(1, Math.max(0, anchor));
  const low = Math.log10(minimum);
  const high = Math.log10(maximum);
  const span = high - low;
  const nextSpan = Math.max(0.08, Math.min(18, span * boundedFactor));
  const focus = low + span * boundedAnchor;
  const nextLow = focus - nextSpan * boundedAnchor;
  return [10 ** nextLow, 10 ** (nextLow + nextSpan)];
}

export function panLogDomain(
  domain: LogDomain,
  fraction: number,
): LogDomain {
  const [minimum, maximum] = validateDomain(domain);
  const low = Math.log10(minimum);
  const high = Math.log10(maximum);
  const shift = (high - low) * Math.max(-8, Math.min(8, fraction));
  return [10 ** (low + shift), 10 ** (high + shift)];
}

function validateDomain(domain: LogDomain): LogDomain {
  if (
    domain.length !== 2
    || !Number.isFinite(domain[0])
    || !Number.isFinite(domain[1])
    || domain[0] <= 0
    || domain[1] <= domain[0]
  ) {
    throw new Error("log domain must contain two increasing positive values");
  }
  return domain;
}
