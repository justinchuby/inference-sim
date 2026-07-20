import { describe, expect, it } from "vitest";
import {
  buildMultiNodeLanScenario,
  buildScenarioPreset,
} from "@inference-sim/core";
import {
  bytesToGibibytes,
  bytesToGigabytesPerSecond,
  finalizeEditedTopology,
  gibibytesToBytes,
  gigabytesPerSecondToBytes,
  materializeNetworkResources,
} from "./topology-editor.js";

describe("topology editor boundary", () => {
  it("round-trips displayed memory and bandwidth units", () => {
    expect(bytesToGibibytes(gibibytesToBytes(80))).toBe(80);
    expect(bytesToGigabytesPerSecond(
      gigabytesPerSecondToBytes(600),
    )).toBe(600);
  });

  it("marks edited topology evidence and advances its epoch", () => {
    const preset = buildScenarioPreset("single-gpu-cpu");
    const edited = finalizeEditedTopology({
      ...preset,
      links: preset.links.map((link, index) => (
        index === 0
          ? { ...link, bandwidthBytesPerSec: 48_000_000_000 }
          : link
      )),
    });
    expect(edited.id).toBe("single-gpu-cpu-custom");
    expect(edited.family).toBe("custom");
    expect(edited.execution.topologyEpoch)
      .toBe(preset.execution.topologyEpoch + 1);
    expect(edited.links[0].bandwidthBytesPerSec).toBe(48_000_000_000);
    expect(edited.links[0].provenance.source).toContain("user-edited");
  });

  it("marks edited network resource evidence", () => {
    const scenario = buildMultiNodeLanScenario(2, {
      advanced: true,
      transport: "gpudirect_rdma",
    });
    const edited = finalizeEditedTopology({
      ...scenario,
      networkResources: scenario.networkResources?.map((resource) => (
        resource.id === "lan:fabric0"
          ? { ...resource, concurrencyLanes: 8 }
          : resource
      )),
    });

    expect(edited.networkResources).toHaveLength(3);
    expect(edited.networkResources?.find(
      (resource) => resource.id === "lan:fabric0",
    )).toMatchObject({
      concurrencyLanes: 8,
      provenance: {
        confidence: "heuristic",
        source: "user-edited in inference-sim web topology editor",
      },
    });
  });

  it("materializes an ordered NIC and shared fabric path for a simple LAN", () => {
    const scenario = buildMultiNodeLanScenario(2);
    const advanced = materializeNetworkResources(scenario);

    expect(advanced.networkResources?.map((resource) => resource.id)).toEqual([
      "node0:nic0",
      "node1:nic0",
      "lan:fabric0",
    ]);
    expect(advanced.links.find(
      (link) => link.id === "lan:node0:node1",
    )?.networkResourceIds).toEqual([
      "node0:nic0",
      "lan:fabric0",
      "node1:nic0",
    ]);
    expect(materializeNetworkResources(advanced)).toBe(advanced);
  });

  it("preserves physical capacity while applying resource limits and features", () => {
    const preset = buildScenarioPreset("single-gpu-cpu");
    const gpuDomain = preset.memoryDomains.find(
      (domain) => domain.kind === "device",
    )!;
    const edited = finalizeEditedTopology({
      ...preset,
      memoryDomains: preset.memoryDomains.map((domain) => (
        domain.id === gpuDomain.id
          ? { ...domain, resourceLimitBytes: 24 * 1024 ** 3 }
          : domain
      )),
      execution: {
        ...preset.execution,
        features: { ssdStreaming: false },
      },
    });
    const constrained = edited.memoryDomains.find(
      (domain) => domain.id === gpuDomain.id,
    )!;
    expect(constrained.capacityBytes).toBe(gpuDomain.capacityBytes);
    expect(constrained.resourceLimitBytes).toBe(24 * 1024 ** 3);
    expect(edited.execution.features.ssdStreaming).toBe(false);
  });

  it("rejects invalid edits through the shared scenario parser", () => {
    const preset = buildScenarioPreset("single-gpu-cpu");
    expect(() => finalizeEditedTopology({
      ...preset,
      devices: preset.devices.map((device, index) => (
        index === 0 ? { ...device, maxConcurrentCompute: 0 } : device
      )),
    })).toThrow("scenario validation failed");
    expect(() => gibibytesToBytes(0)).toThrow("positive safe integer");
  });
});
