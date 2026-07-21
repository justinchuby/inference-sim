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
  ...cpuProfiles(),
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

function cpuProfiles(): readonly HardwareComputeProfile[] {
  const profile = (
    id: string,
    vendor: HardwareComputeProfile["vendor"],
    model: string,
    releaseDate: string,
    productClass: HardwareComputeProfile["productClass"],
    aliases: readonly string[],
    itemSource: HardwareComputeSource,
    options: Pick<HardwareComputeProfile, "peaks"> & { readonly notes?: string } = {
      peaks: [],
    },
  ): HardwareComputeProfile => ({
    id,
    vendor,
    model,
    releaseDate,
    deviceKind: "cpu",
    productClass,
    aliases,
    sources: [itemSource],
    peaks: options.peaks,
    notes: options.notes ?? `${vendor} does not publish an auditable dtype-specific CPU compute peak for this processor.`,
  });

  const appleM2 = source(
    "Apple M2 announcement",
    "https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/",
  );
  const appleM2ProMax = source(
    "Apple M2 Pro and M2 Max announcement",
    "https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/",
  );
  const appleM2Ultra = source(
    "Apple M2 Ultra announcement",
    "https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/",
  );
  const appleM3 = source(
    "Apple M3, M3 Pro, and M3 Max announcement",
    "https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/",
  );
  const appleM3Ultra = source(
    "Apple M3 Ultra announcement",
    "https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/",
  );
  const appleM4 = source(
    "Apple M4 announcement",
    "https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/",
  );
  const appleM4ProMax = source(
    "Apple M4 Pro and M4 Max announcement",
    "https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/",
  );
  const apple = [
    ["apple-m2-cpu", "M2 CPU", "2022-06-06", ["M2 8-core CPU"], appleM2],
    ["apple-m2-pro-cpu", "M2 Pro CPU", "2023-01-17", ["M2 Pro 12-core CPU"], appleM2ProMax],
    ["apple-m2-max-cpu", "M2 Max CPU", "2023-01-17", ["M2 Max 12-core CPU"], appleM2ProMax],
    ["apple-m2-ultra-cpu", "M2 Ultra CPU", "2023-06-05", ["M2 Ultra 24-core CPU"], appleM2Ultra],
    ["apple-m3-cpu", "M3 CPU", "2023-10-30", ["M3 8-core CPU"], appleM3],
    ["apple-m3-pro-cpu", "M3 Pro CPU", "2023-10-30", ["M3 Pro 12-core CPU"], appleM3],
    ["apple-m3-max-cpu", "M3 Max CPU", "2023-10-30", ["M3 Max 16-core CPU"], appleM3],
    ["apple-m3-ultra-cpu", "M3 Ultra CPU", "2025-03-05", ["M3 Ultra 32-core CPU"], appleM3Ultra],
    ["apple-m4-cpu", "M4 CPU", "2024-05-07", ["M4 10-core CPU"], appleM4],
    ["apple-m4-pro-cpu", "M4 Pro CPU", "2024-10-30", ["M4 Pro 14-core CPU"], appleM4ProMax],
    ["apple-m4-max-cpu", "M4 Max CPU", "2024-10-30", ["M4 Max 16-core CPU"], appleM4ProMax],
  ] as const;

  const intelDesktop = source(
    "Intel Core desktop processor generations and launch dates",
    "https://www.intel.com/content/www/us/en/support/articles/000099655/processors.html",
  );
  const intelMeteorLake = source(
    "Intel Core Ultra Series 1 product collection",
    "https://www.intel.com/content/www/us/en/products/details/processors/core-ultra/products.html",
  );
  const intelArrowLake = source(
    "Intel Core Ultra Series 2 product collection",
    "https://www.intel.com/content/www/us/en/products/details/processors/core-ultra/series-2/products.html",
  );
  const intelConsumer = [
    ["intel-core-i5-13600k-cpu", "Core i5-13600K CPU", "2022-10-20", ["i5-13600K", "Raptor Lake"], intelDesktop],
    ["intel-core-i7-13700k-cpu", "Core i7-13700K CPU", "2022-10-20", ["i7-13700K", "Raptor Lake"], intelDesktop],
    ["intel-core-i9-13900k-cpu", "Core i9-13900K CPU", "2022-10-20", ["i9-13900K", "Raptor Lake"], intelDesktop],
    ["intel-core-i5-14600k-cpu", "Core i5-14600K CPU", "2023-10-17", ["i5-14600K", "Raptor Lake Refresh"], intelDesktop],
    ["intel-core-i7-14700k-cpu", "Core i7-14700K CPU", "2023-10-17", ["i7-14700K", "Raptor Lake Refresh"], intelDesktop],
    ["intel-core-i9-14900k-cpu", "Core i9-14900K CPU", "2023-10-17", ["i9-14900K", "Raptor Lake Refresh"], intelDesktop],
    ["intel-core-ultra-5-125h-cpu", "Core Ultra 5 125H CPU", "2023-12-14", ["Ultra 5 125H", "Meteor Lake"], intelMeteorLake],
    ["intel-core-ultra-7-165h-cpu", "Core Ultra 7 165H CPU", "2023-12-14", ["Ultra 7 165H", "Meteor Lake"], intelMeteorLake],
    ["intel-core-ultra-9-185h-cpu", "Core Ultra 9 185H CPU", "2023-12-14", ["Ultra 9 185H", "Meteor Lake"], intelMeteorLake],
    ["intel-core-ultra-5-245k-cpu", "Core Ultra 5 245K CPU", "2024-10-24", ["Ultra 5 245K", "Arrow Lake"], intelArrowLake],
    ["intel-core-ultra-7-265k-cpu", "Core Ultra 7 265K CPU", "2024-10-24", ["Ultra 7 265K", "Arrow Lake"], intelArrowLake],
    ["intel-core-ultra-9-285k-cpu", "Core Ultra 9 285K CPU", "2024-10-24", ["Ultra 9 285K", "Arrow Lake"], intelArrowLake],
  ] as const;

  const intel8480 = source(
    "Intel Xeon Platinum 8480+ processor specifications",
    "https://www.intel.com/content/www/us/en/products/sku/231746/intel-xeon-platinum-8480-processor-105m-cache-2-00-ghz/specifications.html",
  );
  const intel8592 = source(
    "Intel Xeon Platinum 8592+ processor specifications",
    "https://www.intel.com/content/www/us/en/products/sku/237255/intel-xeon-platinum-8592-processor-320m-cache-1-90-ghz/specifications.html",
  );
  const intel6980 = source(
    "Intel Xeon 6980P processor specifications",
    "https://www.intel.com/content/www/us/en/products/sku/240777/intel-xeon-6980p-processor-504m-cache-2-00-ghz/specifications.html",
  );
  const intelServer = [
    profile("intel-xeon-platinum-8480-plus-cpu", "Intel", "Xeon Platinum 8480+ CPU", "2023-01-10", "datacenter", ["Xeon 8480+", "4th Gen Xeon", "Sapphire Rapids"], intel8480, {
      peaks: [peak("fp64", 3.584 * T, "vector"), peak("fp32", 7.168 * T, "vector")],
      notes: "Conservative base-clock vector roof: 56 cores at 2.0 GHz with two documented AVX-512 FMA units per core; AMX is excluded.",
    }),
    profile("intel-xeon-platinum-8592-plus-cpu", "Intel", "Xeon Platinum 8592+ CPU", "2023-12-14", "datacenter", ["Xeon 8592+", "5th Gen Xeon", "Emerald Rapids"], intel8592),
    profile("intel-xeon-6980p-cpu", "Intel", "Xeon 6980P CPU", "2024-09-24", "datacenter", ["Xeon 6980P", "6th Gen Xeon", "Granite Rapids"], intel6980, {
      peaks: [peak("fp64", 8.192 * T, "vector"), peak("fp32", 16.384 * T, "vector")],
      notes: "Conservative base-clock vector roof: 128 cores at 2.0 GHz with two AVX-512 FMA units per core; AMX is excluded.",
    }),
  ];

  const amdRyzen7000 = source(
    "AMD Ryzen 7000 Series desktop processor quick reference guide",
    "https://www.amd.com/content/dam/amd/en/documents/partner-hub/ryzen/amd-ryzen-7000-series-desktop-processors-quick-reference-guide.pdf",
  );
  const amdRyzen9000 = source(
    "AMD Ryzen 9000 Series desktop processors",
    "https://www.amd.com/en/products/processors/desktops/ryzen/9000-series.html",
  );
  const amdConsumer = [
    ["amd-ryzen-5-7600x-cpu", "Ryzen 5 7600X CPU", "2022-09-27", ["Ryzen 5 7600X", "Zen 4"], amdRyzen7000],
    ["amd-ryzen-7-7700x-cpu", "Ryzen 7 7700X CPU", "2022-09-27", ["Ryzen 7 7700X", "Zen 4"], amdRyzen7000],
    ["amd-ryzen-9-7950x-cpu", "Ryzen 9 7950X CPU", "2022-09-27", ["Ryzen 9 7950X", "Zen 4"], amdRyzen7000],
    ["amd-ryzen-5-9600x-cpu", "Ryzen 5 9600X CPU", "2024-08-08", ["Ryzen 5 9600X", "Zen 5"], amdRyzen9000],
    ["amd-ryzen-7-9700x-cpu", "Ryzen 7 9700X CPU", "2024-08-08", ["Ryzen 7 9700X", "Zen 5"], amdRyzen9000],
    ["amd-ryzen-9-9950x-cpu", "Ryzen 9 9950X CPU", "2024-08-15", ["Ryzen 9 9950X", "Zen 5"], amdRyzen9000],
  ] as const;
  const amdThreadripper7000 = source(
    "AMD Ryzen Threadripper PRO 7000 WX-Series announcement",
    "https://www.amd.com/en/newsroom/press-releases/2023-10-19-amd-introduces-new-amd-ryzen-threadripper-7000-ser.html",
  );
  const amdThreadripper9000 = source(
    "AMD Ryzen Threadripper PRO 9000 WX-Series launch",
    "https://www.amd.com/en/blogs/2025/amd-introduces-new-zen-5-based-ryzen-threadripper-pro.html",
  );
  const amd9654 = source(
    "AMD EPYC 9654 product specifications",
    "https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9654.html",
  );
  const amd9965 = source(
    "AMD EPYC 9005 Series processor specifications",
    "https://www.amd.com/en/products/processors/server/epyc/9005-series.html",
  );
  const amdWorkstationAndServer = [
    profile("amd-threadripper-pro-7995wx-cpu", "AMD", "Ryzen Threadripper PRO 7995WX CPU", "2023-11-21", "desktop", ["Threadripper PRO 7995WX", "Storm Peak"], amdThreadripper7000),
    profile("amd-threadripper-pro-9995wx-cpu", "AMD", "Ryzen Threadripper PRO 9995WX CPU", "2025-07-23", "desktop", ["Threadripper PRO 9995WX", "Shimada Peak"], amdThreadripper9000),
    profile("amd-epyc-9654-cpu", "AMD", "EPYC 9654 CPU", "2022-11-10", "datacenter", ["EPYC 9654", "EPYC 9004", "Genoa"], amd9654, {
      peaks: [peak("fp64", 3.6864 * T, "vector"), peak("fp32", 7.3728 * T, "vector")],
      notes: "Conservative base-clock vector roof: 96 cores at 2.4 GHz with two 256-bit FMA pipes per Zen 4 core.",
    }),
    profile("amd-epyc-9965-cpu", "AMD", "EPYC 9965 CPU", "2024-10-10", "datacenter", ["EPYC 9965", "EPYC 9005", "Turin Dense"], amd9965),
  ];

  const qualcommXPlus = source(
    "Qualcomm Snapdragon X Plus specifications",
    "https://www.qualcomm.com/laptops/products/snapdragon-x-plus",
  );
  const qualcommXElite = source(
    "Qualcomm Snapdragon X Elite specifications",
    "https://www.qualcomm.com/laptops/products/snapdragon-x-elite",
  );
  const qualcommX2 = source(
    "Qualcomm Snapdragon X2 Elite product brief",
    "https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Snapdragon-X2-Elite-Product-Brief.pdf",
  );
  const qualcomm = [
    ["qualcomm-snapdragon-x-plus-cpu", "Snapdragon X Plus CPU", "2024-06-18", ["Snapdragon X Plus", "X1P", "Oryon CPU"], qualcommXPlus],
    ["qualcomm-snapdragon-x-elite-cpu", "Snapdragon X Elite CPU", "2024-06-18", ["Snapdragon X Elite", "X1E", "12-core Oryon CPU"], qualcommXElite],
    ["qualcomm-snapdragon-x2-elite-cpu", "Snapdragon X2 Elite CPU", "2026", ["Snapdragon X2 Elite", "X2E-88-100", "X2E-80-100", "Oryon CPU"], qualcommX2],
    ["qualcomm-snapdragon-x2-elite-extreme-cpu", "Snapdragon X2 Elite Extreme CPU", "2026", ["Snapdragon X2 Elite Extreme", "X2E-96-100", "18-core Oryon CPU"], qualcommX2],
  ] as const;

  return [
    ...apple.map(([id, model, releaseDate, aliases, itemSource]) => (
      profile(id, "Apple", model, releaseDate, "integrated", aliases, itemSource)
    )),
    ...intelConsumer.map(([id, model, releaseDate, aliases, itemSource]) => (
      profile(id, "Intel", model, releaseDate, "desktop", aliases, itemSource)
    )),
    ...intelServer,
    ...amdConsumer.map(([id, model, releaseDate, aliases, itemSource]) => (
      profile(id, "AMD", model, releaseDate, "desktop", aliases, itemSource)
    )),
    ...amdWorkstationAndServer,
    ...qualcomm.map(([id, model, releaseDate, aliases, itemSource]) => (
      profile(id, "Qualcomm", model, releaseDate, "mobile", aliases, itemSource)
    )),
  ];
}

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
