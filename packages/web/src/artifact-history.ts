import type {
  DashboardArtifactDownload,
  DashboardResult,
} from "./types.js";

const DATABASE_NAME = "inference-sim-artifacts";
const DATABASE_VERSION = 1;
const STORE_NAME = "artifacts";

export const DEFAULT_ARTIFACT_HISTORY_LIMITS = {
  maxEntries: 20,
  maxTotalBytes: 256 * 1024 * 1024,
} as const;

export interface ArtifactHistoryLimits {
  readonly maxEntries: number;
  readonly maxTotalBytes: number;
}

export interface ArtifactHistoryEntry {
  readonly fingerprint: string;
  readonly fileName: string;
  readonly byteLength: number;
  readonly savedAtMs: number;
  readonly lastOpenedAtMs: number;
  readonly runKind: string;
  readonly scenarioId: string;
  readonly mode: string;
}

interface StoredArtifactHistoryEntry extends ArtifactHistoryEntry {
  readonly blob: Blob;
}

export async function saveArtifactToHistory(
  artifact: DashboardArtifactDownload,
  result: DashboardResult,
  limits: ArtifactHistoryLimits = DEFAULT_ARTIFACT_HISTORY_LIMITS,
  nowMs = Date.now(),
): Promise<readonly ArtifactHistoryEntry[]> {
  validateLimits(limits);
  const database = await openDatabase();
  try {
    const existing = await getEntry(database, artifact.artifactFingerprint);
    await putEntry(database, {
      fingerprint: artifact.artifactFingerprint,
      fileName: artifact.fileName,
      byteLength: artifact.blob.size,
      savedAtMs: existing?.savedAtMs ?? nowMs,
      lastOpenedAtMs: nowMs,
      runKind: result.comparison
        ? `dashboard/${result.mode}/comparison`
        : `dashboard/${result.mode}`,
      scenarioId: result.scenario.id,
      mode: result.mode,
      blob: artifact.blob,
    });
    await enforceRetention(database, limits);
    return await listEntries(database);
  } finally {
    database.close();
  }
}

export async function listArtifactHistory(): Promise<
  readonly ArtifactHistoryEntry[]
> {
  const database = await openDatabase();
  try {
    return await listEntries(database);
  } finally {
    database.close();
  }
}

export async function readArtifactFromHistory(
  fingerprint: string,
  nowMs = Date.now(),
): Promise<{ readonly fileName: string; readonly text: string }> {
  const database = await openDatabase();
  try {
    const entry = await getEntry(database, fingerprint);
    if (entry === undefined) {
      throw new Error(`artifact history entry ${fingerprint} does not exist`);
    }
    await putEntry(database, { ...entry, lastOpenedAtMs: nowMs });
    return {
      fileName: entry.fileName,
      text: await entry.blob.text(),
    };
  } finally {
    database.close();
  }
}

export async function deleteArtifactFromHistory(
  fingerprint: string,
): Promise<readonly ArtifactHistoryEntry[]> {
  const database = await openDatabase();
  try {
    await requestToPromise(
      database.transaction(STORE_NAME, "readwrite")
        .objectStore(STORE_NAME)
        .delete(fingerprint),
    );
    return await listEntries(database);
  } finally {
    database.close();
  }
}

export async function clearArtifactHistory(): Promise<void> {
  const database = await openDatabase();
  try {
    await requestToPromise(
      database.transaction(STORE_NAME, "readwrite")
        .objectStore(STORE_NAME)
        .clear(),
    );
  } finally {
    database.close();
  }
}

function openDatabase(): Promise<IDBDatabase> {
  if (globalThis.indexedDB === undefined) {
    return Promise.reject(
      new Error("artifact history is unavailable in this browser"),
    );
  }
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: "fingerprint" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(
      request.error ?? new Error("failed to open artifact history"),
    );
    request.onblocked = () => reject(
      new Error("artifact history upgrade is blocked by another tab"),
    );
  });
}

async function enforceRetention(
  database: IDBDatabase,
  limits: ArtifactHistoryLimits,
): Promise<void> {
  const entries = await getAllEntries(database);
  entries.sort(compareOldestFirst);
  let totalBytes = entries.reduce((sum, entry) => sum + entry.byteLength, 0);
  let retainedEntries = entries.length;
  for (const entry of entries) {
    if (
      retainedEntries <= limits.maxEntries
      && totalBytes <= limits.maxTotalBytes
    ) {
      break;
    }
    await requestToPromise(
      database.transaction(STORE_NAME, "readwrite")
        .objectStore(STORE_NAME)
        .delete(entry.fingerprint),
    );
    retainedEntries--;
    totalBytes -= entry.byteLength;
  }
}

async function listEntries(
  database: IDBDatabase,
): Promise<readonly ArtifactHistoryEntry[]> {
  return (await getAllEntries(database))
    .sort((left, right) => (
      right.lastOpenedAtMs - left.lastOpenedAtMs
      || right.savedAtMs - left.savedAtMs
      || left.fingerprint.localeCompare(right.fingerprint)
    ))
    .map(({ blob: _blob, ...entry }) => entry);
}

function getAllEntries(
  database: IDBDatabase,
): Promise<StoredArtifactHistoryEntry[]> {
  return requestToPromise(
    database.transaction(STORE_NAME, "readonly")
      .objectStore(STORE_NAME)
      .getAll(),
  );
}

async function getEntry(
  database: IDBDatabase,
  fingerprint: string,
): Promise<StoredArtifactHistoryEntry | undefined> {
  return await requestToPromise(
    database.transaction(STORE_NAME, "readonly")
      .objectStore(STORE_NAME)
      .get(fingerprint),
  ) as StoredArtifactHistoryEntry | undefined;
}

function putEntry(
  database: IDBDatabase,
  entry: StoredArtifactHistoryEntry,
): Promise<IDBValidKey> {
  return requestToPromise(
    database.transaction(STORE_NAME, "readwrite")
      .objectStore(STORE_NAME)
      .put(entry),
  );
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(
      request.error ?? new Error("artifact history operation failed"),
    );
  });
}

function compareOldestFirst(
  left: ArtifactHistoryEntry,
  right: ArtifactHistoryEntry,
): number {
  return left.lastOpenedAtMs - right.lastOpenedAtMs
    || left.savedAtMs - right.savedAtMs
    || left.fingerprint.localeCompare(right.fingerprint);
}

function validateLimits(limits: ArtifactHistoryLimits): void {
  if (!Number.isSafeInteger(limits.maxEntries) || limits.maxEntries <= 0) {
    throw new Error("artifact history maxEntries must be a positive safe integer");
  }
  if (
    !Number.isSafeInteger(limits.maxTotalBytes)
    || limits.maxTotalBytes <= 0
  ) {
    throw new Error(
      "artifact history maxTotalBytes must be a positive safe integer",
    );
  }
}
