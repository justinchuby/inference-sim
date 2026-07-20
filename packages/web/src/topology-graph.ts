import type { SimulationScenario } from "@inference-sim/core";

export interface TopologyGraphNodeData {
  readonly category: "system" | "device" | "memory";
  readonly title: string;
  readonly kind: string;
  readonly nodeId: string;
  readonly details: readonly string[];
  readonly accent:
    | "system"
    | "cpu"
    | "gpu"
    | "npu"
    | "host"
    | "device"
    | "unified"
    | "storage";
}

export interface TopologyGraphNode {
  readonly id: string;
  readonly type: "topology" | "topologyGroup";
  readonly position: { readonly x: number; readonly y: number };
  readonly parentId?: string;
  readonly extent?: "parent";
  readonly style?: {
    readonly width: number;
    readonly height: number;
  };
  readonly data: TopologyGraphNodeData;
}

export interface TopologyGraphEdgeData {
  readonly category: "access" | "link";
  readonly scope: "intra-node" | "inter-node";
  readonly kind: string;
  readonly bandwidthBytesPerSec?: number;
  readonly latencyNs?: number;
  readonly concurrencyLanes?: number;
}

export interface TopologyGraphEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly sourceHandle: string;
  readonly targetHandle: string;
  readonly type: "smoothstep";
  readonly label?: string;
  readonly animated: boolean;
  readonly style: {
    readonly stroke: string;
    readonly strokeWidth: number;
    readonly strokeDasharray?: string;
  };
  readonly markerEnd?: {
    readonly type: "arrowclosed";
    readonly color: string;
  };
  readonly data: TopologyGraphEdgeData;
}

const NODE_WIDTH = 210;
const NODE_GAP = 100;
const GROUP_GAP = 90;
const GROUP_X = 30;
const GROUP_Y = 30;
const GROUP_PADDING_X = 30;
const GROUP_HEADER_HEIGHT = 70;
const GROUP_HEIGHT = 390;
const DEVICE_Y = GROUP_HEADER_HEIGHT;
const MEMORY_Y = 245;

export function buildTopologyGraph(
  scenario: SimulationScenario,
): {
  readonly nodes: readonly TopologyGraphNode[];
  readonly edges: readonly TopologyGraphEdge[];
} {
  const nodeIds = [...new Set([
    ...scenario.devices.map((device) => device.nodeId),
    ...scenario.memoryDomains.map((domain) => domain.nodeId),
  ])].sort();
  const nodes: TopologyGraphNode[] = [];
  const positionById = new Map<string, { readonly x: number; readonly y: number }>();
  const systemByDomain = new Map(
    scenario.memoryDomains.map((domain) => [domain.id, domain.nodeId]),
  );
  const systemByDevice = new Map(
    scenario.devices.map((device) => [device.id, device.nodeId]),
  );
  let groupOffset = GROUP_X;

  for (const nodeId of nodeIds) {
    const devices = scenario.devices
      .filter((device) => device.nodeId === nodeId)
      .sort((left, right) => left.id.localeCompare(right.id));
    const domains = scenario.memoryDomains
      .filter((domain) => domain.nodeId === nodeId)
      .sort((left, right) => left.id.localeCompare(right.id));
    const columns = Math.max(devices.length, domains.length, 1);
    const groupWidth = GROUP_PADDING_X * 2
      + columns * NODE_WIDTH
      + Math.max(0, columns - 1) * NODE_GAP;
    nodes.push({
      id: `system:${nodeId}`,
      type: "topologyGroup",
      position: { x: groupOffset, y: GROUP_Y },
      style: { width: groupWidth, height: GROUP_HEIGHT },
      data: {
        category: "system",
        title: nodeId,
        kind: "system",
        nodeId,
        accent: "system",
        details: [
          `${devices.length} compute chip${devices.length === 1 ? "" : "s"}`,
          `${domains.length} memory domain${domains.length === 1 ? "" : "s"}`,
          "physical fault and transport boundary",
        ],
      },
    });
    devices.forEach((device, index) => {
      const localPosition = {
        x: GROUP_PADDING_X + centeredColumn(index, devices.length, columns),
        y: DEVICE_Y,
      };
      positionById.set(device.id, {
        x: groupOffset + localPosition.x,
        y: GROUP_Y + localPosition.y,
      });
      nodes.push({
        id: device.id,
        type: "topology",
        position: localPosition,
        parentId: `system:${nodeId}`,
        extent: "parent",
        data: {
          category: "device",
          title: device.id,
          kind: device.kind,
          nodeId,
          accent: device.kind,
          details: [
            device.executionProvider,
            `${device.maxConcurrentCompute} compute lane${device.maxConcurrentCompute === 1 ? "" : "s"}`,
            compactList(device.capabilities),
            compactList(device.supportedDtypes),
          ],
        },
      });
    });
    domains.forEach((domain, index) => {
      const localPosition = {
        x: GROUP_PADDING_X + centeredColumn(index, domains.length, columns),
        y: MEMORY_Y,
      };
      positionById.set(domain.id, {
        x: groupOffset + localPosition.x,
        y: GROUP_Y + localPosition.y,
      });
      nodes.push({
        id: domain.id,
        type: "topology",
        position: localPosition,
        parentId: `system:${nodeId}`,
        extent: "parent",
        data: {
          category: "memory",
          title: domain.id,
          kind: domain.kind,
          nodeId,
          accent: domain.kind,
          details: [
            `${formatBytes(domain.capacityBytes)} capacity`,
            `${formatRate(domain.bandwidthBytesPerSec)} local`,
            `${formatDuration(domain.latencyNs)} latency`,
            domain.coherent ? "coherent" : "non-coherent",
          ],
        },
      });
    });
    groupOffset += groupWidth + GROUP_GAP;
  }

  const accessEdges: TopologyGraphEdge[] = scenario.devices.flatMap(
    (device) => device.memoryDomainIds.map((domainId) => ({
      id: `access:${device.id}:${domainId}`,
      source: device.id,
      target: domainId,
      sourceHandle: "bottom-source",
      targetHandle: "top-target",
      type: "smoothstep" as const,
      animated: false,
      style: {
        stroke: "#a1a1aa",
        strokeWidth: 1,
        strokeDasharray: "4 4",
      },
      data: {
        category: "access" as const,
        scope: systemByDevice.get(device.id) === systemByDomain.get(domainId)
          ? "intra-node" as const
          : "inter-node" as const,
        kind: "memory access",
      },
    })),
  );
  const labeledPairs = new Set<string>();
  const linkEdges: TopologyGraphEdge[] = scenario.links.map((link) => {
    const color = linkColor(link.kind);
    const sourceX = positionById.get(link.sourceDomainId)?.x ?? 0;
    const targetX = positionById.get(link.targetDomainId)?.x ?? 0;
    const labelKey = [
      ...[link.sourceDomainId, link.targetDomainId].sort(),
      link.kind,
    ].join("::");
    const crossesNode = scenario.memoryDomains.some((domain) => {
      if (
        domain.id === link.sourceDomainId
        || domain.id === link.targetDomainId
      ) {
        return false;
      }
      const position = positionById.get(domain.id);
      return position !== undefined
        && position.y === positionById.get(link.sourceDomainId)?.y
        && position.x > Math.min(sourceX, targetX)
        && position.x < Math.max(sourceX, targetX);
    });
    const showLabel = !labeledPairs.has(labelKey) && !crossesNode;
    labeledPairs.add(labelKey);
    return {
      id: link.id,
      source: link.sourceDomainId,
      target: link.targetDomainId,
      sourceHandle: sourceX <= targetX ? "right-source" : "left-source",
      targetHandle: sourceX <= targetX ? "left-target" : "right-target",
      type: "smoothstep",
      ...(showLabel
        ? {
            label: `${formatRate(link.bandwidthBytesPerSec)} · ${formatDuration(link.latencyNs)}`,
          }
        : {}),
      animated: false,
      style: {
        stroke: color,
        strokeWidth: 2,
      },
      markerEnd: {
        type: "arrowclosed",
        color,
      },
      data: {
        category: "link",
        scope: systemByDomain.get(link.sourceDomainId)
            === systemByDomain.get(link.targetDomainId)
          ? "intra-node"
          : "inter-node",
        kind: link.kind,
        bandwidthBytesPerSec: link.bandwidthBytesPerSec,
        latencyNs: link.latencyNs,
        concurrencyLanes: link.concurrencyLanes,
      },
    };
  });
  return { nodes, edges: [...accessEdges, ...linkEdges] };
}

function centeredColumn(
  index: number,
  itemCount: number,
  columns: number,
): number {
  const start = ((columns - itemCount) * (NODE_WIDTH + NODE_GAP)) / 2;
  return start + index * (NODE_WIDTH + NODE_GAP);
}

function compactList(values: readonly string[]): string {
  if (values.length <= 3) {
    return values.join(", ") || "none";
  }
  return `${values.slice(0, 3).join(", ")} +${values.length - 3}`;
}

function linkColor(kind: SimulationScenario["links"][number]["kind"]): string {
  switch (kind) {
    case "nvlink":
    case "on-chip":
      return "#047857";
    case "pcie":
    case "thunderbolt":
      return "#0369a1";
    case "ethernet":
    case "infiniband":
      return "#7c3aed";
    case "storage":
      return "#b45309";
  }
}

export function formatBytes(bytes: number): string {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
}

export function formatRate(bytesPerSecond: number): string {
  return `${(bytesPerSecond / 1_000_000_000).toLocaleString(
    "en-US",
    { maximumFractionDigits: 1 },
  )} GB/s`;
}

export function formatDuration(nanoseconds: number): string {
  if (nanoseconds < 1_000) {
    return `${nanoseconds} ns`;
  }
  if (nanoseconds < 1_000_000) {
    return `${(nanoseconds / 1_000).toFixed(1)} us`;
  }
  return `${(nanoseconds / 1_000_000).toFixed(1)} ms`;
}
