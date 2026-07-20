import { fromBinary } from "@bufbuild/protobuf";
import {
  ModelProtoSchema,
  TensorProto_DataType,
  type GraphProto,
  type ModelProto,
  type TensorProto,
} from "onnx-buf";
import {
  ONNX_MODEL_MANIFEST_KIND,
  ONNX_MODEL_MANIFEST_REVISION,
  createOnnxModelManifest,
  type OnnxArchitectureEvidence,
  type OnnxExternalDataFileManifest,
  type OnnxInitializerManifest,
  type OnnxModelManifest,
  type OnnxOperatorCount,
} from "@inference-sim/core";

export const MAX_ONNX_PROTO_BYTES = 512 * 1024 * 1024;

export interface OnnxExternalDataSource {
  readonly byteLength: number;
  readonly sha256: () => Promise<string>;
}

export interface InspectOnnxModelInput {
  readonly modelFileName: string;
  readonly modelBytes: Uint8Array;
  readonly metadata?: unknown;
  readonly sha256: (bytes: Uint8Array) => Promise<string>;
  readonly resolveExternalData: (
    location: string,
  ) => Promise<OnnxExternalDataSource>;
}

export async function inspectOnnxModelBytes({
  modelFileName,
  modelBytes,
  metadata,
  sha256,
  resolveExternalData,
}: InspectOnnxModelInput): Promise<OnnxModelManifest> {
  if (modelBytes.byteLength > MAX_ONNX_PROTO_BYTES) {
    throw new Error("ONNX protobuf exceeds the 512 MiB inspection limit");
  }
  let decoded: ModelProto;
  try {
    decoded = fromBinary(ModelProtoSchema, modelBytes);
  } catch (error) {
    throw new Error(
      `invalid ONNX protobuf: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  if (!decoded.graph) {
    throw new Error("ONNX model does not contain an inference graph");
  }
  if (decoded.trainingInfo.length > 0) {
    throw new Error("training ONNX models are outside the inference manifest contract");
  }

  const operators = new Map<string, number>();
  const initializerRecords: Array<{
    readonly tensor: TensorProto;
    readonly scopedName: string;
  }> = [];
  walkGraph(decoded.graph, "main", operators, initializerRecords);
  const initializers = initializerRecords.map(({ tensor, scopedName }) => (
    inspectInitializer(tensor, scopedName)
  ));
  const externalDataFiles = await inspectExternalDataFiles(
    initializers,
    resolveExternalData,
  );
  const operatorInventory: OnnxOperatorCount[] = [...operators.entries()]
    .map(([identity, count]) => {
      const separator = identity.indexOf("\0");
      return {
        domain: identity.slice(0, separator),
        opType: identity.slice(separator + 1),
        count,
      };
    })
    .sort((left, right) => (
      left.domain.localeCompare(right.domain)
      || left.opType.localeCompare(right.opType)
    ));
  const architecture = normalizeArchitectureEvidence(metadata);
  const missingFields = [
    "headDimension",
    "hiddenSize",
    "intermediateSize",
    "numAttentionHeads",
    "numHiddenLayers",
    "numKeyValueHeads",
    "vocabSize",
    ...(architecture.numExperts === undefined
      ? []
      : [
          "activeExpertsPerToken",
          "expertBytesPerLayer",
          "sharedExpertBytesPerLayer",
        ]),
  ].filter((field) => (
    architecture[field as keyof OnnxArchitectureEvidence] === undefined
  )).sort();
  const totals = {
    initializerElements: checkedSum(
      initializers.map((tensor) => tensor.elementCount),
      "initializer element total",
    ),
    initializerLogicalBytes: checkedSum(
      initializers.map((tensor) => tensor.logicalByteLength),
      "initializer logical byte total",
    ),
    inlineInitializerBytes: checkedSum(
      initializers.filter((tensor) => tensor.storage.kind === "inline")
        .map((tensor) => tensor.storage.byteLength),
      "inline initializer byte total",
    ),
    externalInitializerBytes: checkedSum(
      initializers.filter((tensor) => tensor.storage.kind === "external")
        .map((tensor) => tensor.storage.byteLength),
      "external initializer byte total",
    ),
  };

  return createOnnxModelManifest({
    kind: ONNX_MODEL_MANIFEST_KIND,
    revision: ONNX_MODEL_MANIFEST_REVISION,
    source: {
      modelFileName,
      modelByteLength: modelBytes.byteLength,
      sha256: await sha256(modelBytes),
    },
    model: {
      irVersion: decoded.irVersion.toString(),
      producerName: decoded.producerName,
      producerVersion: decoded.producerVersion,
      domain: decoded.domain,
      modelVersion: decoded.modelVersion.toString(),
    },
    graph: {
      name: decoded.graph.name,
      nodeCount: checkedSum([...operators.values()], "ONNX node count"),
      initializerCount: initializers.length,
      inputNames: decoded.graph.input.map((input) => input.name),
      outputNames: decoded.graph.output.map((output) => output.name),
      operators: operatorInventory,
    },
    initializers,
    externalDataFiles,
    architecture,
    totals,
    profileReadiness: {
      ready: missingFields.length === 0,
      missingFields,
    },
  });
}

function walkGraph(
  graph: GraphProto,
  path: string,
  operators: Map<string, number>,
  initializers: Array<{
    readonly tensor: TensorProto;
    readonly scopedName: string;
  }>,
): void {
  for (const tensor of graph.initializer) {
    if (tensor.name.length === 0) {
      throw new Error(`ONNX initializer in ${path} has no name`);
    }
    initializers.push({
      tensor,
      scopedName: path === "main" ? tensor.name : `${path}::${tensor.name}`,
    });
  }
  if (graph.sparseInitializer.length > 0) {
    throw new Error("sparse ONNX initializers are not yet supported");
  }
  for (const [nodeIndex, node] of graph.node.entries()) {
    if (node.opType.length === 0) {
      throw new Error(`ONNX node ${path}/${nodeIndex} has no operator type`);
    }
    const domain = node.domain.length === 0 ? "ai.onnx" : node.domain;
    const identity = `${domain}\0${node.opType}`;
    operators.set(identity, (operators.get(identity) ?? 0) + 1);
    for (const [attributeIndex, attribute] of node.attribute.entries()) {
      if (
        attribute.t?.externalData.length
        || attribute.tensors.some((tensor) => tensor.externalData.length > 0)
      ) {
        throw new Error(
          `external tensor attribute ${path}/${nodeIndex}/${attributeIndex} is not supported`,
        );
      }
      if (attribute.g) {
        walkGraph(
          attribute.g,
          `${path}/${nodeIndex}/${attribute.name || attributeIndex}`,
          operators,
          initializers,
        );
      }
      for (const [graphIndex, nested] of attribute.graphs.entries()) {
        walkGraph(
          nested,
          `${path}/${nodeIndex}/${attribute.name || attributeIndex}/${graphIndex}`,
          operators,
          initializers,
        );
      }
    }
  }
}

function inspectInitializer(
  tensor: TensorProto,
  scopedName: string,
): OnnxInitializerManifest {
  if (tensor.segment !== undefined) {
    throw new Error(`segmented ONNX initializer ${scopedName} is unsupported`);
  }
  const dimensions = tensor.dims.map((dimension, index) => (
    safeBigIntNumber(dimension, `${scopedName} dimension ${index}`)
  ));
  const elementCount = dimensions.length === 0
    ? 1
    : checkedProduct(dimensions, `${scopedName} element count`);
  const dataType = tensorDataTypeName(tensor.dataType);
  const logicalByteLength = tensorLogicalByteLength(
    tensor,
    elementCount,
    dataType,
  );
  const external = Object.fromEntries(
    tensor.externalData.map((entry) => [entry.key, entry.value]),
  );
  if (tensor.dataLocation === 1 || tensor.externalData.length > 0) {
    const location = safeExternalLocation(external.location, scopedName);
    const offset = parseExternalInteger(
      external.offset ?? "0",
      `${scopedName} external offset`,
    );
    const byteLength = external.length === undefined
      ? logicalByteLength
      : parseExternalInteger(
          external.length,
          `${scopedName} external length`,
        );
    return {
      name: scopedName,
      dataType,
      dimensions,
      elementCount,
      logicalByteLength,
      storage: {
        kind: "external",
        location,
        offset,
        byteLength,
      },
    };
  }
  return {
    name: scopedName,
    dataType,
    dimensions,
    elementCount,
    logicalByteLength,
    storage: {
      kind: "inline",
      byteLength: tensor.rawData.byteLength > 0
        ? tensor.rawData.byteLength
        : logicalByteLength,
    },
  };
}

async function inspectExternalDataFiles(
  initializers: readonly OnnxInitializerManifest[],
  resolveExternalData: (
    location: string,
  ) => Promise<OnnxExternalDataSource>,
): Promise<OnnxExternalDataFileManifest[]> {
  const rangesByLocation = new Map<string, Array<readonly [number, number]>>();
  for (const tensor of initializers) {
    if (tensor.storage.kind !== "external") {
      continue;
    }
    const location = tensor.storage.location!;
    const start = tensor.storage.offset!;
    const end = checkedAdd(
      start,
      tensor.storage.byteLength,
      `${tensor.name} external extent`,
    );
    const ranges = rangesByLocation.get(location) ?? [];
    ranges.push([start, end]);
    rangesByLocation.set(location, ranges);
  }
  const files: OnnxExternalDataFileManifest[] = [];
  for (const location of [...rangesByLocation.keys()].sort()) {
    const source = await resolveExternalData(location);
    if (!Number.isSafeInteger(source.byteLength)) {
      throw new Error(`external-data file is too large: ${location}`);
    }
    const ranges = rangesByLocation.get(location)!;
    for (const [, end] of ranges) {
      if (end > source.byteLength) {
        throw new Error(
          `external-data range exceeds ${location}: ${end} > ${source.byteLength}`,
        );
      }
    }
    files.push({
      location,
      byteLength: source.byteLength,
      referencedByteLength: unionByteLength(ranges),
      sha256: await source.sha256(),
    });
  }
  return files;
}

function normalizeArchitectureEvidence(
  metadata: unknown,
): OnnxArchitectureEvidence {
  if (metadata === undefined) {
    return { source: "none" };
  }
  const root = record(metadata, "ONNX package metadata");
  if (root.model !== undefined) {
    const model = record(root.model, "ONNX package metadata model");
    if (model.decoder !== undefined) {
      const decoder = record(
        model.decoder,
        "ONNX package metadata model.decoder",
      );
      const headDimension = optionalPositiveInteger(
        decoder.head_size,
        "model.decoder.head_size",
      );
      const hiddenSize = optionalPositiveInteger(
        decoder.hidden_size,
        "model.decoder.hidden_size",
      );
      return compactArchitecture({
        source: "genai_config",
        modelType: optionalString(model.type, "model.type"),
        vocabSize: optionalPositiveInteger(model.vocab_size, "model.vocab_size"),
        hiddenSize,
        numHiddenLayers: optionalPositiveInteger(
          decoder.num_hidden_layers,
          "model.decoder.num_hidden_layers",
        ),
        numAttentionHeads: optionalPositiveInteger(
          decoder.num_attention_heads,
          "model.decoder.num_attention_heads",
        ),
        numKeyValueHeads: optionalPositiveInteger(
          decoder.num_key_value_heads,
          "model.decoder.num_key_value_heads",
        ),
        headDimension,
      });
    }
    const attention = model.attention === undefined
      ? undefined
      : record(model.attention, "inference metadata model.attention");
    return compactArchitecture({
      source: "inference_metadata",
      numAttentionHeads: optionalPositiveInteger(
        attention?.num_attention_heads,
        "model.attention.num_attention_heads",
      ),
      numKeyValueHeads: optionalPositiveInteger(
        attention?.num_kv_heads,
        "model.attention.num_kv_heads",
      ),
      headDimension: optionalPositiveInteger(
        attention?.head_dim,
        "model.attention.head_dim",
      ),
    });
  }
  return compactArchitecture({
    source: "onnx_genai_manifest",
    modelType: optionalString(root.architecture, "architecture"),
    hiddenSize: optionalPositiveInteger(root.hidden_size, "hidden_size"),
    intermediateSize: optionalPositiveInteger(
      root.intermediate_size,
      "intermediate_size",
    ),
    numHiddenLayers: optionalPositiveInteger(
      root.num_hidden_layers,
      "num_hidden_layers",
    ),
    numAttentionHeads: optionalPositiveInteger(
      root.num_attention_heads,
      "num_attention_heads",
    ),
    numKeyValueHeads: optionalPositiveInteger(
      root.num_key_value_heads,
      "num_key_value_heads",
    ),
    headDimension: optionalPositiveInteger(root.head_dim, "head_dim"),
    vocabSize: optionalPositiveInteger(root.vocab_size, "vocab_size"),
    numExperts: optionalPositiveInteger(root.num_experts, "num_experts"),
    activeExpertsPerToken: optionalPositiveInteger(
      root.active_experts_per_token ?? root.num_experts_per_tok,
      "active_experts_per_token",
    ),
    expertBytesPerLayer: optionalPositiveInteger(
      root.expert_bytes_per_layer,
      "expert_bytes_per_layer",
    ),
    sharedExpertBytesPerLayer: optionalNonNegativeInteger(
      root.shared_expert_bytes_per_layer,
      "shared_expert_bytes_per_layer",
    ),
  });
}

function compactArchitecture(
  input: OnnxArchitectureEvidence,
): OnnxArchitectureEvidence {
  return Object.fromEntries(
    Object.entries(input).filter(([, value]) => value !== undefined),
  ) as unknown as OnnxArchitectureEvidence;
}

function tensorDataTypeName(value: number): string {
  const name = TensorProto_DataType[value];
  if (typeof name !== "string" || value === TensorProto_DataType.UNDEFINED) {
    throw new Error(`unsupported ONNX tensor data type ${value}`);
  }
  return name.toLowerCase();
}

function tensorLogicalByteLength(
  tensor: TensorProto,
  elementCount: number,
  dataType: string,
): number {
  if (dataType === "string") {
    return checkedSum(
      tensor.stringData.map((entry) => entry.byteLength),
      `${tensor.name} string bytes`,
    );
  }
  const bits = dataTypeBits(dataType);
  const totalBits = checkedProduct(
    [elementCount, bits],
    `${tensor.name} logical bits`,
  );
  return Math.ceil(totalBits / 8);
}

function dataTypeBits(dataType: string): number {
  const widths: Readonly<Record<string, number>> = {
    float: 32,
    uint8: 8,
    int8: 8,
    uint16: 16,
    int16: 16,
    int32: 32,
    int64: 64,
    bool: 8,
    float16: 16,
    double: 64,
    uint32: 32,
    uint64: 64,
    complex64: 64,
    complex128: 128,
    bfloat16: 16,
    float8e4m3fn: 8,
    float8e4m3fnuz: 8,
    float8e5m2: 8,
    float8e5m2fnuz: 8,
    uint4: 4,
    int4: 4,
    float4e2m1: 4,
    float8e8m0: 8,
    uint2: 2,
    int2: 2,
  };
  const bits = widths[dataType];
  if (bits === undefined) {
    throw new Error(`unsupported ONNX tensor data type ${dataType}`);
  }
  return bits;
}

function safeExternalLocation(value: string | undefined, tensor: string): string {
  if (
    value === undefined
    || value.length === 0
    || value.startsWith("/")
    || value.startsWith("\\")
    || /^[A-Za-z]:/.test(value)
    || value.split(/[\\/]/).some((segment) => segment === "..")
  ) {
    throw new Error(
      `unsafe or missing external-data location for ${tensor}`,
    );
  }
  return value.replaceAll("\\", "/");
}

function parseExternalInteger(value: string, label: string): number {
  if (!/^(0|[1-9]\d*)$/.test(value)) {
    throw new Error(`${label} must be an unsigned decimal integer`);
  }
  const result = Number(value);
  if (!Number.isSafeInteger(result)) {
    throw new Error(`${label} exceeds safe integer range`);
  }
  return result;
}

function safeBigIntNumber(value: bigint, label: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${label} must be a non-negative safe integer`);
  }
  return Number(value);
}

function checkedProduct(values: readonly number[], label: string): number {
  let product = 1;
  for (const value of values) {
    product *= value;
    if (!Number.isSafeInteger(product)) {
      throw new Error(`${label} exceeds safe integer range`);
    }
  }
  return product;
}

function checkedSum(values: readonly number[], label: string): number {
  let sum = 0;
  for (const value of values) {
    sum = checkedAdd(sum, value, label);
  }
  return sum;
}

function checkedAdd(left: number, right: number, label: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw new Error(`${label} exceeds safe integer range`);
  }
  return sum;
}

function unionByteLength(
  ranges: readonly (readonly [number, number])[],
): number {
  const ordered = [...ranges].sort((left, right) => (
    left[0] - right[0] || left[1] - right[1]
  ));
  let total = 0;
  let start = -1;
  let end = -1;
  for (const [nextStart, nextEnd] of ordered) {
    if (nextStart > end) {
      if (start >= 0) {
        total = checkedAdd(total, end - start, "external referenced bytes");
      }
      start = nextStart;
      end = nextEnd;
    } else {
      end = Math.max(end, nextEnd);
    }
  }
  return start < 0
    ? 0
    : checkedAdd(total, end - start, "external referenced bytes");
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function optionalString(value: unknown, label: string): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function optionalPositiveInteger(
  value: unknown,
  label: string,
): number | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new Error(`${label} must be a positive safe integer`);
  }
  return value as number;
}

function optionalNonNegativeInteger(
  value: unknown,
  label: string,
): number | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${label} must be a non-negative safe integer`);
  }
  return value as number;
}
