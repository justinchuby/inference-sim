# Computer Presets

Built-in computer presets are convenient starting points for local inference
planning. They are separate from the six abstract topology templates used by
protocol campaigns and topology-wide comparisons.

| Preset | Compute topology | Physical memory | Default allocatable limit | Memory bandwidth |
|---|---|---:|---:|---:|
| RTX 4090 desktop | CPU + RTX 4090 over PCIe 4 x16 | 64 GiB RAM + 24 GiB VRAM | 56 GiB RAM + 22 GiB VRAM | 83 GB/s RAM; 1,008 GB/s VRAM |
| RTX 5090 desktop | CPU + RTX 5090 over PCIe 5 x16 | 128 GiB RAM + 32 GiB VRAM | 112 GiB RAM + 30 GiB VRAM | 90 GB/s RAM; 1,792 GB/s VRAM |
| Mac mini M4 Pro | CPU + GPU + Neural Engine | 64 GiB unified | 56 GiB | 273 GB/s |
| Mac Studio M3 Ultra | CPU + GPU + Neural Engine | 512 GiB unified | 480 GiB | 819 GB/s |
| Ryzen AI Max+ 395 | CPU + Radeon 8060S + XDNA 2 NPU | 128 GiB unified | 112 GiB | 256 GB/s |

The RTX frame-buffer sizes and bandwidths, Apple unified-memory sizes and
bandwidths, and AMD Halo memory configuration and bandwidth come from
vendor-published specifications:

- NVIDIA GeForce RTX 5090 specifications, including the RTX 4090 comparison:
  <https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/>
- Apple Mac mini specifications:
  <https://www.apple.com/mac-mini/specs/>
- Apple Mac Studio specifications:
  <https://www.apple.com/mac-studio/specs/>
- AMD Ryzen AI Halo developer platform:
  <https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html>

Host RAM bandwidth, PCIe service efficiency, latency, storage throughput, and
reserved-memory limits are conservative illustrative assumptions rather than
benchmarks of a specific OEM system. Device peaks do not determine realized
model throughput. The simulator labels timing heuristic until compatible
calibration evidence is imported.

Every preset includes local SSD backing and enables SSD streaming. The topology
editor can lower RAM, VRAM, unified-memory, or SSD allocation limits and can
disable SSD streaming without changing physical capacity.
