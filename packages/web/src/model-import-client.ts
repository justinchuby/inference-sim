import type { ImportedModelPackage } from "./model-package-import.js";
import type {
  ModelImportWorkerRequest,
  ModelImportWorkerResponse,
} from "./model-import-worker.js";

export function importModelPackage(
  files: readonly File[],
): Promise<ImportedModelPackage> {
  const worker = new Worker(
    new URL("./model-import-worker.ts", import.meta.url),
    { type: "module" },
  );
  return new Promise((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<ModelImportWorkerResponse>) => {
      worker.terminate();
      if (event.data.type === "result") {
        resolve(event.data.result);
      } else {
        reject(new Error(event.data.message));
      }
    };
    worker.onerror = (event) => {
      worker.terminate();
      reject(new Error(event.message || "model import worker failed"));
    };
    worker.postMessage({ files } satisfies ModelImportWorkerRequest);
  });
}
