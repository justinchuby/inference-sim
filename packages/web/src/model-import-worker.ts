/// <reference lib="webworker" />

import {
  inspectBrowserModelPackage,
  type BrowserPackageFile,
  type ImportedModelPackage,
} from "./model-package-import.js";

export interface ModelImportWorkerRequest {
  readonly files: readonly BrowserPackageFile[];
}

export type ModelImportWorkerResponse =
  | {
      readonly type: "result";
      readonly result: ImportedModelPackage;
    }
  | {
      readonly type: "error";
      readonly message: string;
    };

const worker = self as DedicatedWorkerGlobalScope;

worker.onmessage = (event: MessageEvent<ModelImportWorkerRequest>) => {
  void inspectBrowserModelPackage(event.data.files)
    .then((result) => {
      worker.postMessage({
        type: "result",
        result,
      } satisfies ModelImportWorkerResponse);
    })
    .catch((error) => {
      worker.postMessage({
        type: "error",
        message: error instanceof Error ? error.message : String(error),
      } satisfies ModelImportWorkerResponse);
    });
};

export {};
