import { describe, expect, it } from "vitest";
import {
  buildMultiNodeLanScenario,
  buildScenarioPreset,
} from "@inference-sim/core";
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
      ) + 4,
    );
    expect(graph.edges.filter((edge) => edge.data.category === "link"))
      .toHaveLength(4);
    expect(graph.edges.find((edge) => edge.id === "node0:nvlink:forward"))
      .toMatchObject({
        source: "node0:gpu0:vram",
        target: "node0:gpu1:vram",
        label: "600 GB/s · 500 ns",
        markerStart: { type: "arrowclosed" },
        markerEnd: { type: "arrowclosed" },
        data: { scope: "intra-node", bidirectional: true },
      });
    expect(graph.edges.find((edge) => edge.id === "node0:pcie0:forward")?.label)
      .toBe("2× 32 GB/s · 1.5 us");
    expect(graph.edges.find((edge) => edge.id === "node0:pcie0:forward"))
      .toMatchObject({
        sourceHandle: "top-source",
        targetHandle: "bottom-target",
      });
    expect(graph.edges.find((edge) => edge.id === "node0:pcie1:forward")?.label)
      .toBeUndefined();
    const vram = graph.nodes.find((node) => node.id === "node0:gpu0:vram")!;
    const host = graph.nodes.find((node) => node.id === "node0:host")!;
    expect(vram.position.y).toBeLessThan(host.position.y);
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

  it("keeps asymmetric reverse links on separate visual paths", () => {
    const preset = buildScenarioPreset("multi-gpu");
    const scenario = {
      ...preset,
      links: preset.links.map((link) => link.id === "node0:nvlink:reverse"
        ? { ...link, bandwidthBytesPerSec: link.bandwidthBytesPerSec / 2 }
        : link),
    };
    const graph = buildTopologyGraph(scenario);
    const nvlinkEdges = graph.edges.filter((edge) => (
      edge.data.kind === "nvlink"
    ));

    expect(nvlinkEdges).toHaveLength(2);
    expect(nvlinkEdges.every((edge) => edge.markerStart === undefined))
      .toBe(true);
  });

  it("projects advanced network resources into the logical link path", () => {
    const scenario = buildMultiNodeLanScenario(2, {
      advanced: true,
      linkKind: "infiniband",
      transport: "gpudirect_rdma",
    });
    const graph = buildTopologyGraph(scenario);
    const logicalLink = scenario.links.find(
      (link) => link.id === "lan:node0:node1",
    )!;
    const pathEdges = graph.edges.filter((edge) => (
      edge.id.startsWith(`${logicalLink.id}:path:`)
    ));

    expect(graph.nodes.find((node) => node.id === "node0:nic0"))
      .toMatchObject({
        parentId: "system:node0",
        extent: "parent",
        data: {
          category: "network",
          kind: "NIC / HCA",
          accent: "nic",
        },
      });
    const fabric = graph.nodes.find((node) => node.id === "lan:fabric0")!;
    expect(fabric).not.toHaveProperty("parentId");
    expect(fabric).toMatchObject({
      data: {
        category: "network",
        kind: "switch fabric",
        accent: "fabric",
      },
    });
    expect(pathEdges.map((edge) => [edge.source, edge.target])).toEqual([
      ["node0:gpu0:vram", "node0:nic0"],
      ["node0:nic0", "lan:fabric0"],
      ["lan:fabric0", "node1:nic0"],
      ["node1:nic0", "node1:gpu0:vram"],
    ]);
    expect(pathEdges[1]).toMatchObject({
      label: "gpudirect_rdma · 50 GB/s · 3.0 us",
      data: {
        transport: "gpudirect_rdma",
        logicalSourceId: "node0:gpu0:vram",
        logicalTargetId: "node1:gpu0:vram",
        networkResourceIds: [
          "node0:nic0",
          "lan:fabric0",
          "node1:nic0",
        ],
        segmentCount: 4,
      },
    });
    expect(pathEdges[3].markerEnd).toMatchObject({ type: "arrowclosed" });
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
