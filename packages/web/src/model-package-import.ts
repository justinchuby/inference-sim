import {
  parseInferenceMetadata,
  type InferenceMetadataSummary,
  type OnnxModelManifest,
} from "@inference-sim/core";
import {
  inspectOnnxModelBytes,
  MAX_ONNX_PROTO_BYTES,
} from "@inference-sim/onnx-inspector";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";

export const MAX_MODEL_PACKAGE_FILES = 512;
export const MAX_MODEL_PACKAGE_BYTES = 8 * 1024 ** 3;
export const MAX_INFERENCE_METADATA_BYTES = 4 * 1024 * 1024;

export interface BrowserPackageFile {
  readonly name: string;
  readonly size: number;
  readonly webkitRelativePath?: string;
  readonly arrayBuffer: () => Promise<ArrayBuffer>;
  readonly stream: () => ReadableStream<Uint8Array>;
}

export interface ImportedOnnxModel {
  readonly fileName: string;
  readonly componentIds: readonly string[];
  readonly manifest: OnnxModelManifest;
}

export interface ImportedModelPackage {
  readonly metadataFileName?: string;
  readonly metadata: InferenceMetadataSummary;
  readonly models: readonly ImportedOnnxModel[];
  readonly fileCount: number;
  readonly packageByteLength: number;
  readonly unboundOnnxFiles: readonly string[];
}

export async function inspectBrowserModelPackage(
  selectedFiles: readonly BrowserPackageFile[],
): Promise<ImportedModelPackage> {
  if (selectedFiles.length === 0) {
    throw new Error("model package import requires at least one file");
  }
  if (selectedFiles.length > MAX_MODEL_PACKAGE_FILES) {
    throw new Error(
      `model package exceeds the ${MAX_MODEL_PACKAGE_FILES}-file limit`,
    );
  }
  const packageByteLength = checkedSum(
    selectedFiles.map((file) => file.size),
    "model package byte length",
  );
  if (packageByteLength > MAX_MODEL_PACKAGE_BYTES) {
    throw new Error("model package exceeds the 8 GiB inspection limit");
  }
  const files = normalizePackageFiles(selectedFiles);
  const metadataEntry = findMetadataFile(files);
  const metadataValue = metadataEntry === undefined
    ? {}
    : await parseMetadataFile(metadataEntry);
  const metadata = parseInferenceMetadata(metadataValue);
  const discoveredOnnxEntries = [...files.entries()]
    .filter(([path]) => path.toLowerCase().endsWith(".onnx"))
    .sort(([left], [right]) => left.localeCompare(right));

  const componentsByFile = new Map<string, string[]>();
  for (const component of metadata.components) {
    const componentPath = normalizeRelativePath(component.filename);
    if (!files.has(componentPath)) {
      throw new Error(
        `pipeline component ${component.id} is missing ${component.filename}`,
      );
    }
    const componentIds = componentsByFile.get(componentPath) ?? [];
    componentIds.push(component.id);
    componentsByFile.set(componentPath, componentIds);
  }
  const modelPaths = new Set([
    ...discoveredOnnxEntries.map(([path]) => path),
    ...componentsByFile.keys(),
  ]);
  if (modelPaths.size === 0) {
    throw new Error("model package does not contain an ONNX model");
  }
  const onnxEntries = [...modelPaths].sort().map((path) => (
    [path, files.get(path)!] as const
  ));

  const models: ImportedOnnxModel[] = [];
  for (const [modelPath, file] of onnxEntries) {
    if (file.size > MAX_ONNX_PROTO_BYTES) {
      throw new Error(`${modelPath} exceeds the 512 MiB ONNX protobuf limit`);
    }
    const modelBytes = new Uint8Array(await file.arrayBuffer());
    const modelDirectory = parentPath(modelPath);
    const manifest = await inspectOnnxModelBytes({
      modelFileName: modelPath,
      modelBytes,
      metadata: metadataValue,
      sha256: async (bytes) => bytesToHex(sha256(bytes)),
      resolveExternalData: async (location) => {
        const externalPath = resolvePackagePath(modelDirectory, location);
        const external = files.get(externalPath);
        if (external === undefined) {
          throw new Error(
            `${modelPath} references missing external data ${externalPath}`,
          );
        }
        return {
          byteLength: external.size,
          sha256: () => sha256Stream(external),
        };
      },
    });
    models.push({
      fileName: modelPath,
      componentIds: [...(componentsByFile.get(modelPath) ?? [])].sort(),
      manifest,
    });
  }

  return {
    ...(metadataEntry === undefined
      ? {}
      : { metadataFileName: metadataEntry.path }),
    metadata,
    models,
    fileCount: files.size,
    packageByteLength,
    unboundOnnxFiles: models
      .filter((model) => model.componentIds.length === 0)
      .map((model) => model.fileName),
  };
}

interface NormalizedMetadataFile {
  readonly path: string;
  readonly file: BrowserPackageFile;
}

function normalizePackageFiles(
  selectedFiles: readonly BrowserPackageFile[],
): ReadonlyMap<string, BrowserPackageFile> {
  const rawPaths = selectedFiles.map((file) => (
    normalizeSelectionPath(file.webkitRelativePath || file.name)
  ));
  const commonRoot = commonSelectionRoot(rawPaths);
  const files = new Map<string, BrowserPackageFile>();
  selectedFiles.forEach((file, index) => {
    const rawPath = rawPaths[index]!;
    const path = commonRoot === undefined
      ? rawPath
      : rawPath.slice(commonRoot.length + 1);
    if (files.has(path)) {
      throw new Error(`model package contains duplicate path ${path}`);
    }
    files.set(path, file);
  });
  return files;
}

function commonSelectionRoot(paths: readonly string[]): string | undefined {
  const first = paths[0]?.split("/")[0];
  return first !== undefined
    && paths.every((path) => path.startsWith(`${first}/`))
    ? first
    : undefined;
}

function findMetadataFile(
  files: ReadonlyMap<string, BrowserPackageFile>,
): NormalizedMetadataFile | undefined {
  const candidates = [...files.entries()].filter(([path]) => (
    /(^|\/)inference_metadata\.(ya?ml|json)$/i.test(path)
  ));
  if (candidates.length > 1) {
    throw new Error(
      "model package contains multiple inference_metadata files",
    );
  }
  const candidate = candidates[0];
  return candidate === undefined
    ? undefined
    : { path: candidate[0], file: candidate[1] };
}

async function parseMetadataFile(
  entry: NormalizedMetadataFile,
): Promise<unknown> {
  if (entry.file.size > MAX_INFERENCE_METADATA_BYTES) {
    throw new Error("inference metadata exceeds the 4 MiB limit");
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(
    await entry.file.arrayBuffer(),
  );
  try {
    if (entry.path.toLowerCase().endsWith(".json")) {
      return JSON.parse(text);
    }
    const { parse } = await import("yaml");
    return parse(text);
  } catch (error) {
    throw new Error(
      `invalid inference metadata: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

async function sha256Stream(file: BrowserPackageFile): Promise<string> {
  const hash = sha256.create();
  const reader = file.stream().getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        return bytesToHex(hash.digest());
      }
      hash.update(value);
    }
  } finally {
    reader.releaseLock();
  }
}

function resolvePackagePath(directory: string, location: string): string {
  const normalizedLocation = normalizeRelativePath(location);
  return directory.length === 0
    ? normalizedLocation
    : `${directory}/${normalizedLocation}`;
}

function parentPath(path: string): string {
  const separator = path.lastIndexOf("/");
  return separator < 0 ? "" : path.slice(0, separator);
}

function normalizeSelectionPath(path: string): string {
  const normalized = path.replaceAll("\\", "/");
  if (
    normalized.length === 0
    || normalized.startsWith("/")
    || /^[A-Za-z]:/.test(normalized)
    || normalized.split("/").some((segment) => (
      segment.length === 0 || segment === "." || segment === ".."
    ))
  ) {
    throw new Error(`unsafe model package path ${path}`);
  }
  return normalized;
}

function normalizeRelativePath(path: string): string {
  return normalizeSelectionPath(path);
}

function checkedSum(values: readonly number[], label: string): number {
  let sum = 0;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new Error(`${label} contains an invalid file size`);
    }
    sum += value;
    if (!Number.isSafeInteger(sum)) {
      throw new Error(`${label} exceeds safe integer range`);
    }
  }
  return sum;
}
