import { describe, expect, it } from "vitest";
import { buildScenarioPreset } from "@inference-sim/core";
import {
  buildTopologyGraph,
  formatDuration,
  formatRate,
} from "./topology-graph.js";

describe("topology graph projection", () => {
  it("projects every device, memory domain, access, and directed link", () => {
    const scenario = buildScenarioPreset("multi-gpu");
    const graph = buildTopologyGraph(scenario);
    const systemIds = new Set([
      ...scenario.devices.map((device) => device.nodeId),
      ...scenario.memoryDomains.map((domain) => domain.nodeId),
    ]);

    expect(graph.nodes).toHaveLength(
      systemIds.size + scenario.devices.length + scenario.memoryDomains.length,
    );
    expect(graph.nodes.find((node) => node.id === "system:node0"))
      .toMatchObject({
        type: "topologyGroup",
        data: {
          category: "system",
          title: "node0",
        },
      });
    expect(graph.nodes.find((node) => node.id === "node0:gpu0"))
      .toMatchObject({
        parentId: "system:node0",
        extent: "parent",
        data: {
          category: "device",
        },
      });
    expect(graph.edges).toHaveLength(
      scenario.devices.reduce(
        (sum, device) => sum + device.memoryDomainIds.length,
        0,
      ) + scenario.links.length,
    );
    expect(graph.edges.filter((edge) => edge.data.category === "link"))
      .toHaveLength(scenario.links.length);
    expect(graph.edges.find((edge) => edge.id === "node0:nvlink:forward"))
      .toMatchObject({
        source: "node0:gpu0:vram",
        target: "node0:gpu1:vram",
        label: "600 GB/s · 500 ns",
        markerEnd: { type: "arrowclosed" },
        data: { scope: "intra-node" },
      });
    expect(graph.edges.find((edge) => edge.id === "node0:pcie0:forward")?.label)
      .toBeUndefined();
    expect(graph.edges.find((edge) => edge.id === "node0:pcie1:forward")?.label)
      .toBe("32 GB/s · 1.5 us");
  });

  it("separates nodes deterministically across machines", () => {
    const first = buildTopologyGraph(buildScenarioPreset("multi-node"));
    const second = buildTopologyGraph(buildScenarioPreset("multi-node"));
    expect(first).toEqual(second);
    const node0 = first.nodes.find((node) => node.id === "system:node0")!;
    const node1 = first.nodes.find((node) => node.id === "system:node1")!;
    expect(node0.position.x + Number(node0.style?.width))
      .toBeLessThan(node1.position.x);
    expect(
      first.nodes
        .filter((node) => node.data.nodeId === "node0")
        .every((node) => (
          node.data.category === "system"
          || node.parentId === "system:node0"
        )),
    ).toBe(true);
    expect(
      first.edges
        .filter((edge) => edge.data.category === "link")
        .some((edge) => edge.data.scope === "inter-node"),
    ).toBe(true);
  });

  it("formats transport evidence without hiding units", () => {
    expect(formatRate(32_000_000_000)).toBe("32 GB/s");
    expect(formatDuration(500)).toBe("500 ns");
    expect(formatDuration(1_500)).toBe("1.5 us");
  });

  it("shows resource limits separately from physical capacity", () => {
    const preset = buildScenarioPreset("single-gpu-cpu");
    const scenario = {
      ...preset,
      memoryDomains: preset.memoryDomains.map((domain) => (
        domain.kind === "device"
          ? { ...domain, resourceLimitBytes: 24 * 1024 ** 3 }
          : domain
      )),
      execution: {
        ...preset.execution,
        features: { ssdStreaming: false },
      },
    };
    const graph = buildTopologyGraph(scenario);
    expect(graph.nodes.find((node) => node.id === "node0:gpu0:vram")?.data)
      .toMatchObject({
        details: [
          "24.0 GiB allocatable / 80.0 GiB physical",
          "2,000 GB/s local",
          "80 ns latency",
          "non-coherent",
        ],
      });
    expect(graph.nodes.find((node) => node.id === "node0:storage")?.data
      .details[0]).toBe("disabled / 2.0 TiB physical");
  });
});
