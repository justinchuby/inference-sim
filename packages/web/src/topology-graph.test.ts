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

    expect(graph.nodes).toHaveLength(
      scenario.devices.length + scenario.memoryDomains.length,
    );
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
    const node0 = first.nodes.filter((node) => node.data.nodeId === "node0");
    const node1 = first.nodes.filter((node) => node.data.nodeId === "node1");
    expect(Math.max(...node0.map((node) => node.position.x)))
      .toBeLessThan(Math.min(...node1.map((node) => node.position.x)));
  });

  it("formats transport evidence without hiding units", () => {
    expect(formatRate(32_000_000_000)).toBe("32 GB/s");
    expect(formatDuration(500)).toBe("500 ns");
    expect(formatDuration(1_500)).toBe("1.5 us");
  });
});
