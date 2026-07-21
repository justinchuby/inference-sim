export const PROMPT_TOKEN_STEPS = [
  16,
  32,
  64,
  128,
  256,
  512,
  1_024,
  2_048,
  4_096,
  8_192,
  16_384,
  32_768,
  65_536,
  131_072,
  262_144,
  524_288,
  1_048_576,
] as const;

export const OUTPUT_TOKEN_STEPS = [
  1,
  2,
  4,
  8,
  ...PROMPT_TOKEN_STEPS.filter((value) => value <= 32_768),
] as const;

export function nearestTokenStepIndex(
  value: number,
  steps: readonly number[],
): number {
  let nearestIndex = 0;
  let nearestDistance = Infinity;
  steps.forEach((step, index) => {
    const distance = Math.abs(Math.log2(Math.max(1, value) / step));
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestIndex = index;
    }
  });
  return nearestIndex;
}

export function formatTokenCount(value: number): string {
  if (value >= 1_048_576) {
    const millions = value / 1_048_576;
    return `${Number.isInteger(millions)
      ? millions
      : millions.toFixed(2)}M`;
  }
  if (value >= 1_024 && value % 1_024 === 0) {
    return `${value / 1_024}K`;
  }
  return value.toLocaleString("en-US");
}
