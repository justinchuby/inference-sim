import type { SimDeviceKind } from "./scenario-types.js";

export const HARDWARE_COMPUTE_REGISTRY_REVISION = 1;

export type HardwareComputeDtype =
  | "fp64"
  | "fp32"
  | "tf32"
  | "bf16"
  | "fp16"
  | "fp8"
  | "fp6"
  | "fp4"
  | "int8"
  | "int4"
  | "int2"
  | "int1"
  | "vendor_ai";

export interface HardwareComputeSource {
  readonly label: string;
  readonly url: string;
  readonly accessedAt: string;
}

export interface HardwareComputePeak {
  readonly dtype: HardwareComputeDtype;
  readonly operationsPerSecond: number;
  readonly engine: "vector" | "matrix" | "tensor" | "npu" | "gpu_shader";
  readonly sparsity: "dense" | "structured" | "unspecified";
  readonly accumulationDtype?: HardwareComputeDtype;
  readonly sourceIndex: number;
  readonly notes?: string;
}

export interface HardwareComputeProfile {
  readonly id: string;
  readonly vendor: "NVIDIA" | "AMD" | "Intel" | "Qualcomm" | "Apple";
  readonly model: string;
  readonly releaseDate: string;
  readonly deviceKind: SimDeviceKind;
  readonly productClass: "datacenter" | "desktop" | "mobile" | "integrated";
  readonly aliases: readonly string[];
  readonly sources: readonly HardwareComputeSource[];
  readonly peaks: readonly HardwareComputePeak[];
  readonly notes?: string;
}

const T = 1e12;
const P = 1e15;
const ACCESSED_AT = "2026-07-20";

function source(label: string, url: string): HardwareComputeSource {
  return { label, url, accessedAt: ACCESSED_AT };
}

function peak(
  dtype: HardwareComputeDtype,
  value: number,
  engine: HardwareComputePeak["engine"],
  options: Partial<Omit<HardwareComputePeak,
    "dtype" | "operationsPerSecond" | "engine">> = {},
): HardwareComputePeak {
  return {
    dtype,
    operationsPerSecond: value,
    engine,
    sparsity: "dense",
    sourceIndex: 0,
    ...options,
  };
}

const NVIDIA_ADA_SOURCE = source(
  "NVIDIA Ada GPU Architecture whitepaper, table 7",
  "https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf",
);
const NVIDIA_BLACKWELL_SOURCE = source(
  "NVIDIA RTX Blackwell GPU Architecture whitepaper, table 5",
  "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf",
);
const AMD_MI300_SOURCE = source(
  "AMD Instinct MI300 Series accelerators specifications",
  "https://www.amd.com/en/products/accelerators/instinct/mi300.html",
);
const AMD_CDNA4_SOURCE = source(
  "AMD CDNA 4 Architecture whitepaper, table 2",
  "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf",
);
const NVIDIA_HOPPER_SOURCE = source(
  "NVIDIA H100 Tensor Core GPU specifications",
  "https://www.nvidia.com/en-us/data-center/h100/",
);
const INTEL_GAUDI_SOURCE = source(
  "Intel Gaudi 3 AI Accelerator whitepaper",
  "https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html",
);

export const HARDWARE_COMPUTE_PROFILES: readonly HardwareComputeProfile[] = [
  ...(["H100 SXM", "H200 SXM"] as const).map((model) => ({
    id: `nvidia-${model.toLowerCase().replaceAll(" ", "-")}`,
    vendor: "NVIDIA" as const,
    model,
    releaseDate: model.startsWith("H100") ? "2022-10" : "2024-04",
    deviceKind: "gpu" as const,
    productClass: "datacenter" as const,
    aliases: [model.replace(" SXM", "")],
    sources: [NVIDIA_HOPPER_SOURCE],
    peaks: [
      peak("fp16", 989.5 * T, "tensor"),
      peak("bf16", 989.5 * T, "tensor"),
      peak("fp8", 1_979 * T, "tensor"),
      peak("int8", 1_979 * T, "tensor"),
    ],
    notes: "Dense values are one-half of NVIDIA's starred structured-sparsity figures.",
  })),
  ...([
    ["nvidia-l4", "NVIDIA L4", "2023-03-21", 121, 242.5],
    ["nvidia-l40s", "NVIDIA L40S", "2023-08-08", 362.05, 733],
  ] as const).map(([id, model, releaseDate, fp16, fp8]) => ({
    id,
    vendor: "NVIDIA" as const,
    model,
    releaseDate,
    deviceKind: "gpu" as const,
    productClass: "datacenter" as const,
    aliases: [model.replace("NVIDIA ", "")],
    sources: [source(
      `${model} product specifications`,
      `https://www.nvidia.com/en-us/data-center/${id.replace("nvidia-", "")}/`,
    )],
    peaks: [
      peak("fp16", fp16 * T, "tensor"),
      peak("bf16", fp16 * T, "tensor"),
      peak("fp8", fp8 * T, "tensor"),
      peak("int8", fp8 * T, "tensor"),
    ],
    notes: "Dense values are one-half of NVIDIA's starred structured-sparsity figures.",
  })),
  {
    id: "nvidia-geforce-rtx-4090",
    vendor: "NVIDIA",
    model: "GeForce RTX 4090",
    releaseDate: "2022-10-12",
    deviceKind: "gpu",
    productClass: "desktop",
    aliases: ["RTX 4090", "AD102"],
    sources: [NVIDIA_ADA_SOURCE],
    peaks: [
      peak("fp32", 82.6 * T, "gpu_shader"),
      peak("tf32", 82.6 * T, "tensor", { accumulationDtype: "fp32" }),
      peak("bf16", 165.2 * T, "tensor", { accumulationDtype: "fp32" }),
      peak("fp16", 165.2 * T, "tensor", { accumulationDtype: "fp32" }),
      peak("fp16", 330.3 * T, "tensor", { accumulationDtype: "fp16" }),
      peak("fp8", 330.3 * T, "tensor", { accumulationDtype: "fp32" }),
      peak("fp8", 660.6 * T, "tensor", { accumulationDtype: "fp16" }),
      peak("int8", 660.6 * T, "tensor"),
      peak("int4", 1_321.2 * T, "tensor"),
    ],
  },
  ...([
    ["nvidia-geforce-rtx-5090", "GeForce RTX 5090", "2025-01-30", 209.5, 419, 838, 1_676],
    ["nvidia-geforce-rtx-5080", "GeForce RTX 5080", "2025-01-30", 112.6, 225.1, 450.2, 900.4],
    ["nvidia-geforce-rtx-5070-ti", "GeForce RTX 5070 Ti", "2025-02-20", 87.9, 175.8, 351.5, 703],
    ["nvidia-geforce-rtx-5070", "GeForce RTX 5070", "2025-03-05", 61.7, 123.5, 246.9, 493.9],
  ] as const).map(([id, model, releaseDate, fp16Fp32, fp16Fp16, fp8Fp16, fp4]) => ({
    id,
    vendor: "NVIDIA" as const,
    model,
    releaseDate,
    deviceKind: "gpu" as const,
    productClass: "desktop" as const,
    aliases: [model.replace("GeForce ", "")],
    sources: [NVIDIA_BLACKWELL_SOURCE],
    peaks: [
      peak("fp16", fp16Fp32 * T, "tensor", { accumulationDtype: "fp32" }),
      peak("bf16", fp16Fp32 * T, "tensor", { accumulationDtype: "fp32" }),
      peak("fp16", fp16Fp16 * T, "tensor", { accumulationDtype: "fp16" }),
      peak("fp8", fp16Fp16 * T, "tensor", { accumulationDtype: "fp32" }),
      peak("fp8", fp8Fp16 * T, "tensor", { accumulationDtype: "fp16" }),
      peak("fp4", fp4 * T, "tensor", { accumulationDtype: "fp32" }),
      peak("int8", fp8Fp16 * T, "tensor"),
    ],
  })),
  ...([
    ["amd-instinct-mi300x", "Instinct MI300X", "2023-12-06", 653.7, 1_307.4, 2_614.9],
    ["amd-instinct-mi325x", "Instinct MI325X", "2024-10-10", 653.7, 1_307.4, 2_614.9],
  ] as const).map(([id, model, releaseDate, tf32, fp16, fp8]) => ({
    id,
    vendor: "AMD" as const,
    model,
    releaseDate,
    deviceKind: "gpu" as const,
    productClass: "datacenter" as const,
    aliases: [model.replace("Instinct ", "")],
    sources: [AMD_MI300_SOURCE],
    peaks: [
      peak("tf32", tf32 * T, "matrix", { accumulationDtype: "fp32" }),
      peak("fp16", fp16 * T, "matrix"),
      peak("bf16", fp16 * T, "matrix"),
      peak("fp8", fp8 * T, "matrix"),
      peak("int8", fp8 * T, "matrix"),
    ],
  })),
  ...([
    ["amd-instinct-mi350x", "Instinct MI350X", "2025-06-12", 2.3, 4.6, 9.2],
    ["amd-instinct-mi355x", "Instinct MI355X", "2025-06-12", 2.5, 5, 10],
  ] as const).map(([id, model, releaseDate, fp16, fp8, fp4]) => ({
    id,
    vendor: "AMD" as const,
    model,
    releaseDate,
    deviceKind: "gpu" as const,
    productClass: "datacenter" as const,
    aliases: [model.replace("Instinct ", "")],
    sources: [AMD_CDNA4_SOURCE],
    peaks: [
      peak("fp16", fp16 * P, "matrix"),
      peak("bf16", fp16 * P, "matrix"),
      peak("fp8", fp8 * P, "matrix"),
      peak("fp6", fp4 * P, "matrix"),
      peak("fp4", fp4 * P, "matrix"),
      peak("int8", fp8 * P, "matrix"),
    ],
  })),
  {
    id: "amd-radeon-rx-9070-xt",
    vendor: "AMD",
    model: "Radeon RX 9070 XT",
    releaseDate: "2025-03-06",
    deviceKind: "gpu",
    productClass: "desktop",
    aliases: ["RX 9070 XT", "Navi 48"],
    sources: [source(
      "AMD Radeon RX 9070 XT product specifications",
      "https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html",
    )],
    peaks: [
      peak("fp32", 48.7 * T, "gpu_shader"),
      peak("fp16", 195 * T, "matrix"),
      peak("fp8", 389 * T, "matrix"),
      peak("int8", 389 * T, "matrix"),
      peak("int4", 779 * T, "matrix"),
    ],
  },
  {
    id: "amd-radeon-rx-7900-xtx",
    vendor: "AMD",
    model: "Radeon RX 7900 XTX",
    releaseDate: "2022-12-13",
    deviceKind: "gpu",
    productClass: "desktop",
    aliases: ["RX 7900 XTX", "Navi 31"],
    sources: [source(
      "AMD Radeon RX 7900 XTX product specifications",
      "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html",
    )],
    peaks: [
      peak("fp16", 123 * T, "matrix"),
      peak("int8", 123 * T, "matrix"),
      peak("int4", 246 * T, "matrix"),
    ],
  },
  ...intelProfiles(),
  ...qualcommProfiles(),
  ...appleProfiles(),
];

function intelProfiles(): readonly HardwareComputeProfile[] {
  const npuSource = (model: string, url: string) => source(
    `Intel ${model} processor specifications`,
    url,
  );
  return [
    {
      id: "intel-gaudi-2",
      vendor: "Intel",
      model: "Gaudi 2",
      releaseDate: "2022-05-10",
      deviceKind: "gpu",
      productClass: "datacenter",
      aliases: ["Habana Gaudi2", "HL-225B"],
      sources: [INTEL_GAUDI_SOURCE],
      peaks: [
        peak("bf16", 432 * T, "matrix", { sparsity: "unspecified" }),
        peak("fp8", 865 * T, "matrix", { sparsity: "unspecified" }),
      ],
    },
    {
      id: "intel-gaudi-3",
      vendor: "Intel",
      model: "Gaudi 3",
      releaseDate: "2024-04-09",
      deviceKind: "gpu",
      productClass: "datacenter",
      aliases: ["Habana Gaudi3", "HL-325L"],
      sources: [INTEL_GAUDI_SOURCE],
      peaks: [
        peak("fp32", 229 * T, "matrix", { accumulationDtype: "fp32", sparsity: "unspecified" }),
        peak("tf32", 459 * T, "matrix", { accumulationDtype: "fp32", sparsity: "unspecified" }),
        peak("fp16", 459 * T, "matrix", { accumulationDtype: "fp32", sparsity: "unspecified" }),
        peak("bf16", 1_678 * T, "matrix", { accumulationDtype: "fp32", sparsity: "unspecified" }),
        peak("fp8", 1_678 * T, "matrix", { accumulationDtype: "fp32", sparsity: "unspecified" }),
      ],
      notes: "MME peaks; TPC vector rates are intentionally not added to them.",
    },
    ...([
      ["intel-core-ultra-series-1-npu", "Core Ultra Series 1 NPU", "2023-12-14", 11, "https://www.intel.com/content/www/us/en/products/sku/236776/intel-core-ultra-7-processor-165h-24m-cache-up-to-5-00-ghz/specifications.html", "unspecified"],
      ["intel-core-ultra-9-288v-npu", "Core Ultra 9 288V NPU", "2024-09-03", 48, "https://www.intel.com/content/www/us/en/products/sku/240961/intel-core-ultra-9-processor-288v-12m-cache-up-to-5-10-ghz/specifications.html", "dense"],
      ["intel-core-ultra-9-285k-npu", "Core Ultra 9 285K NPU", "2024-10-24", 13, "https://www.intel.com/content/www/us/en/products/sku/241060/intel-core-ultra-9-processor-285k-36m-cache-up-to-5-70-ghz/specifications.html", "unspecified"],
      ["intel-core-ultra-9-285h-npu", "Core Ultra 9 285H NPU", "2025-01-06", 13, "https://www.intel.com/content/www/us/en/products/sku/241747/intel-core-ultra-9-processor-285h-24m-cache-up-to-5-40-ghz/specifications.html", "unspecified"],
      ["intel-core-ultra-x9-388h-npu", "Core Ultra X9 388H NPU", "2026", 50, "https://www.intel.com/content/www/us/en/products/sku/245526/intel-core-ultra-x9-processor-388h-18m-cache-up-to-5-10-ghz/specifications.html", "unspecified"],
    ] as const).map(([id, model, releaseDate, tops, url, sparsity]) => ({
      id,
      vendor: "Intel" as const,
      model,
      releaseDate,
      deviceKind: "npu" as const,
      productClass: "integrated" as const,
      aliases: [model.replace(" NPU", "")],
      sources: [npuSource(model, url)],
      peaks: [peak("int8", tops * T, "npu", { sparsity })],
      notes: "NPU-only peak; platform TOPS from CPU, GPU, and NPU are not included.",
    })),
  ];
}

function qualcommProfiles(): readonly HardwareComputeProfile[] {
  const profile = (
    id: string,
    model: string,
    releaseDate: string,
    dtype: HardwareComputeDtype,
    tops: number,
    url: string,
    sparsity: HardwareComputePeak["sparsity"] = "unspecified",
  ): HardwareComputeProfile => ({
    id,
    vendor: "Qualcomm",
    model,
    releaseDate,
    deviceKind: "npu",
    productClass: model.includes("Cloud") ? "datacenter" : "integrated",
    aliases: [model],
    sources: [source(`Qualcomm ${model} specifications`, url)],
    peaks: [peak(dtype, tops * T, "npu", { sparsity })],
  });
  return [
    {
      ...profile(
        "qualcomm-cloud-ai-100-ultra",
        "Cloud AI 100 Ultra",
        "2023-11-15",
        "int8",
        870,
        "https://www.qualcomm.com/data-center/products/cloud-ai-100-ultra",
      ),
      deviceKind: "npu" as const,
      peaks: [
        peak("int8", 870 * T, "npu", { sparsity: "unspecified" }),
        peak("fp16", 288 * T, "npu", { sparsity: "unspecified" }),
      ],
    },
    profile("qualcomm-snapdragon-x-elite-npu", "Snapdragon X Elite Hexagon NPU", "2024-06-18", "vendor_ai", 45, "https://www.qualcomm.com/laptops/products/snapdragon-x-elite"),
    profile("qualcomm-snapdragon-x-plus-npu", "Snapdragon X Plus Hexagon NPU", "2024-06-18", "vendor_ai", 45, "https://www.qualcomm.com/laptops/products/snapdragon-x-plus"),
    profile("qualcomm-snapdragon-x2-elite-npu", "Snapdragon X2 Elite Hexagon NPU", "2026", "int8", 85, "https://www.qualcomm.com/laptops/products/snapdragon-x2-elite"),
    profile("qualcomm-qcs8550-npu", "QCS8550 Hexagon NPU", "2023", "int8", 48, "https://www.qualcomm.com/internet-of-things/products/q8-series/qcs8550", "dense"),
    profile("qualcomm-dragonwing-iq-9075-100-npu", "Dragonwing IQ-9075 100-TOPS NPU", "2024", "int8", 100, "https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075", "dense"),
    profile("qualcomm-dragonwing-iq-8275-40-npu", "Dragonwing IQ-8275 40-TOPS NPU", "2024", "int8", 40, "https://www.qualcomm.com/internet-of-things/products/iq8-series/iq-8275", "dense"),
  ];
}

function appleProfiles(): readonly HardwareComputeProfile[] {
  const m2 = source(
    "Apple M2 announcement",
    "https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/",
  );
  const m2ProMax = source(
    "Apple M2 Pro and M2 Max announcement",
    "https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/",
  );
  const m2Ultra = source(
    "Apple M2 Ultra announcement",
    "https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/",
  );
  const m4 = source(
    "Apple M4 announcement",
    "https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/",
  );
  const knownNpu = [
    ["apple-m2-neural-engine", "M2 Neural Engine", "2022-06-06", 15.8, m2],
    ["apple-m2-pro-neural-engine", "M2 Pro Neural Engine", "2023-01-17", 15.8, m2ProMax],
    ["apple-m2-max-neural-engine", "M2 Max Neural Engine", "2023-01-17", 15.8, m2ProMax],
    ["apple-m2-ultra-neural-engine", "M2 Ultra Neural Engine", "2023-06-05", 31.6, m2Ultra],
    ["apple-m4-neural-engine", "M4 Neural Engine", "2024-05-07", 38, m4],
  ] as const;
  const undisclosed = [
    ["apple-m3-ultra-gpu", "M3 Ultra GPU", "2025-03-05", "gpu"],
    ["apple-m3-ultra-neural-engine", "M3 Ultra Neural Engine", "2025-03-05", "npu"],
    ["apple-m4-pro-gpu", "M4 Pro GPU", "2024-10-30", "gpu"],
    ["apple-m4-pro-neural-engine", "M4 Pro Neural Engine", "2024-10-30", "npu"],
    ["apple-m4-max-gpu", "M4 Max GPU", "2024-10-30", "gpu"],
    ["apple-m4-max-neural-engine", "M4 Max Neural Engine", "2024-10-30", "npu"],
  ] as const;
  const m3UltraSource = source(
    "Apple M3 Ultra announcement",
    "https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/",
  );
  const m4ProMaxSource = source(
    "Apple M4 Pro and M4 Max announcement",
    "https://www.apple.com/ca/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/",
  );
  return [
    ...knownNpu.map(([id, model, releaseDate, tops, itemSource]) => ({
      id,
      vendor: "Apple" as const,
      model,
      releaseDate,
      deviceKind: "npu" as const,
      productClass: "integrated" as const,
      aliases: [model],
      sources: [itemSource],
      peaks: [peak("vendor_ai", tops * T, "npu", {
        sparsity: "unspecified",
        notes: "Apple does not disclose the dtype or operation-count convention.",
      })],
      notes: "Generic Neural Engine TOPS are stored for display only and are not a dtype-specific roof.",
    })),
    ...undisclosed.map(([id, model, releaseDate, deviceKind]) => ({
      id,
      vendor: "Apple" as const,
      model,
      releaseDate,
      deviceKind,
      productClass: "integrated" as const,
      aliases: [model],
      sources: [model.startsWith("M3") ? m3UltraSource : m4ProMaxSource],
      peaks: [],
      notes: "Apple publishes core counts and relative performance, but no absolute dtype-specific compute peak.",
    })),
  ];
}

const PROFILE_BY_ID = new Map(HARDWARE_COMPUTE_PROFILES.map((profile) => (
  [profile.id, profile]
)));

export function hardwareComputeProfile(
  id: string | undefined,
): HardwareComputeProfile | undefined {
  return id === undefined ? undefined : PROFILE_BY_ID.get(id);
}

export function denseHardwareComputePeak(
  profileId: string | undefined,
  dtype: string,
): HardwareComputePeak | undefined {
  const profile = hardwareComputeProfile(profileId);
  if (profile === undefined) return undefined;
  const matches = profile.peaks.filter((candidate) => (
    candidate.dtype === dtype.toLowerCase()
    && candidate.sparsity === "dense"
  ));
  const conservative = matches.find((candidate) => (
    candidate.accumulationDtype === "fp32"
  ));
  return conservative ?? matches.reduce<HardwareComputePeak | undefined>(
    (best, candidate) => best === undefined
      || candidate.operationsPerSecond < best.operationsPerSecond
      ? candidate
      : best,
    undefined,
  );
}

export function hardwareComputeDtypes(
  profile: HardwareComputeProfile,
): readonly string[] {
  return [...new Set(profile.peaks.filter((item) => (
    item.dtype !== "vendor_ai"
  )).map((item) => item.dtype))];
}
