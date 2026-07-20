import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import { parse as parseYaml } from "yaml";

export async function readConfigFile(path: string): Promise<unknown> {
  const text = await readFile(path, "utf8");
  if (extname(path).toLowerCase() === ".json") {
    return JSON.parse(text) as unknown;
  }
  return parseYaml(text) as unknown;
}

export function requireRecord(
  value: unknown,
  path: string,
): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

export function requireString(
  record: Record<string, unknown>,
  key: string,
  path: string,
): string {
  const value = record[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${path}.${key} must be a non-empty string`);
  }
  return value;
}

export function optionalString(
  record: Record<string, unknown>,
  key: string,
  fallback: string,
  path: string,
): string {
  const value = record[key];
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${path}.${key} must be a non-empty string`);
  }
  return value;
}

export function requireNumber(
  record: Record<string, unknown>,
  key: string,
  path: string,
): number {
  const value = record[key];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${path}.${key} must be a finite number`);
  }
  return value;
}

export function optionalNumber(
  record: Record<string, unknown>,
  key: string,
  fallback: number,
  path: string,
): number {
  const value = record[key];
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${path}.${key} must be a finite number`);
  }
  return value;
}

export function requireNumberArray(
  record: Record<string, unknown>,
  key: string,
  path: string,
): number[] {
  const value = record[key];
  if (
    !Array.isArray(value)
    || value.some((entry) => typeof entry !== "number" || !Number.isFinite(entry))
  ) {
    throw new Error(`${path}.${key} must be an array of finite numbers`);
  }
  return value;
}
