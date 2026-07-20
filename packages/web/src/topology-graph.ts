import type {
  NetworkResourceSpec,
  SimulationScenario,
} from "@inference-sim/core";
import { hardwareComputeProfile } from "@inference-sim/core";

export interface TopologyGraphNodeData {
  readonly category: "system" | "device" | "memory" | "network";
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
    | "storage"
    | "nic"
    | "fabric";
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
  readonly transport?: SimulationScenario["links"][number]["transport"];
  readonly logicalSourceId?: string;
  readonly logicalTargetId?: string;
  readonly networkResourceIds?: readonly string[];
  readonly segmentIndex?: number;
  readonly segmentCount?: number;
  readonly bidirectional?: boolean;
  readonly logicalReverseId?: string;
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
  readonly markerStart?: {
    readonly type: "arrowclosed";
    readonly color: string;
  };
  readonly data: TopologyGraphEdgeData;
}

const NODE_WIDTH = 210;
const NODE_GAP = 120;
const GROUP_GAP = 90;
const GROUP_X = 30;
const GROUP_Y = 135;
const GROUP_PADDING_X = 30;
const GROUP_HEADER_HEIGHT = 70;
const GROUP_HEIGHT = 590;
const DEVICE_Y = GROUP_HEADER_HEIGHT;
const LOCAL_MEMORY_Y = 225;
const SHARED_MEMORY_Y = 345;
const NETWORK_Y = 465;
const FABRIC_Y = 30;

export function buildTopologyGraph(
  scenario: SimulationScenario,
): {
  readonly nodes: readonly TopologyGraphNode[];
  readonly edges: readonly TopologyGraphEdge[];
} {
  const nodeIds = [...new Set([
    ...scenario.devices.map((device) => device.nodeId),
    ...scenario.memoryDomains.map((domain) => domain.nodeId),
    ...(scenario.networkResources ?? []).flatMap((resource) => (
      resource.nodeId === undefined ? [] : [resource.nodeId]
    )),
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
    const networkResources = (scenario.networkResources ?? [])
      .filter((resource) => resource.nodeId === nodeId)
      .sort((left, right) => left.id.localeCompare(right.id));
    const columns = Math.max(
      devices.length,
      domains.length,
      networkResources.length,
      1,
    );
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
          `${networkResources.length} network adapter${networkResources.length === 1 ? "" : "s"}`,
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
            hardwareComputeProfile(device.computeProfileId)?.model
              ?? "compute peak not bound",
          ],
        },
      });
    });
    domains.forEach((domain, index) => {
      const enabled = domain.kind !== "storage"
        || scenario.execution.features.ssdStreaming;
      const localPosition = {
        x: GROUP_PADDING_X + centeredColumn(index, domains.length, columns),
        y: domain.kind === "device" || domain.kind === "unified"
          ? LOCAL_MEMORY_Y
          : SHARED_MEMORY_Y,
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
            enabled
              ? `${formatBytes(domain.resourceLimitBytes)} allocatable / ${formatBytes(domain.capacityBytes)} physical`
              : `disabled / ${formatBytes(domain.capacityBytes)} physical`,
            `${formatRate(domain.bandwidthBytesPerSec)} local`,
            `${formatDuration(domain.latencyNs)} latency`,
            domain.coherent ? "coherent" : "non-coherent",
          ],
        },
      });
    });
    networkResources.forEach((resource, index) => {
      const localPosition = {
        x: GROUP_PADDING_X
          + centeredColumn(index, networkResources.length, columns),
        y: NETWORK_Y,
      };
      positionById.set(resource.id, {
        x: groupOffset + localPosition.x,
        y: GROUP_Y + localPosition.y,
      });
      nodes.push(networkResourceNode(resource, localPosition, nodeId));
    });
    groupOffset += groupWidth + GROUP_GAP;
  }

  const fabrics = (scenario.networkResources ?? [])
    .filter((resource) => resource.nodeId === undefined)
    .sort((left, right) => left.id.localeCompare(right.id));
  const totalWidth = Math.max(groupOffset - GROUP_GAP - GROUP_X, NODE_WIDTH);
  fabrics.forEach((resource, index) => {
    const position = {
      x: GROUP_X + (totalWidth - NODE_WIDTH) / 2
        + (index - (fabrics.length - 1) / 2) * (NODE_WIDTH + 30),
      y: FABRIC_Y,
    };
    positionById.set(resource.id, position);
    nodes.push(networkResourceNode(resource, position));
  });

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
  const renderedLinks = new Set<string>();
  const linkEdges: TopologyGraphEdge[] = scenario.links.flatMap((link) => {
    if (renderedLinks.has(link.id)) return [];
    const reverse = scenario.links.find((candidate) => (
      candidate.id !== link.id
      && candidate.sourceDomainId === link.targetDomainId
      && candidate.targetDomainId === link.sourceDomainId
      && equivalentLinkContract(link, candidate)
    ));
    renderedLinks.add(link.id);
    if (reverse !== undefined) renderedLinks.add(reverse.id);
    const color = linkColor(link.kind);
    const path = [
      link.sourceDomainId,
      ...(link.networkResourceIds ?? []).filter((resourceId) => (
        positionById.has(resourceId)
      )),
      link.targetDomainId,
    ];
    const sourceX = positionById.get(path[0])?.x ?? 0;
    const targetX = positionById.get(path[path.length - 1])?.x ?? 0;
    return path.slice(0, -1).map((source, segmentIndex) => {
      const target = path[segmentIndex + 1];
      const segmentSourceX = positionById.get(source)?.x ?? sourceX;
      const segmentTargetX = positionById.get(target)?.x ?? targetX;
      const labelSegment = Math.floor((path.length - 2) / 2);
      return {
        id: path.length === 2 ? link.id : `${link.id}:path:${segmentIndex}`,
        source,
        target,
        sourceHandle: segmentSourceX <= segmentTargetX
          ? "right-source"
          : "left-source",
        targetHandle: segmentSourceX <= segmentTargetX
          ? "left-target"
          : "right-target",
        type: "smoothstep" as const,
        ...(segmentIndex === labelSegment
          ? {
              label: [
                link.transport,
                formatRate(link.bandwidthBytesPerSec),
                formatDuration(link.latencyNs),
              ].filter(Boolean).join(" · "),
            }
          : {}),
        animated: false,
        style: {
          stroke: color,
          strokeWidth: 2,
        },
        ...(segmentIndex === path.length - 2
          ? {
              markerEnd: {
                type: "arrowclosed" as const,
                color,
              },
            }
          : {}),
        ...(reverse !== undefined && segmentIndex === 0
          ? {
              markerStart: {
                type: "arrowclosed" as const,
                color,
              },
            }
          : {}),
        data: {
          category: "link" as const,
          scope: systemByDomain.get(link.sourceDomainId)
              === systemByDomain.get(link.targetDomainId)
            ? "intra-node" as const
            : "inter-node" as const,
          kind: link.kind,
          bandwidthBytesPerSec: link.bandwidthBytesPerSec,
          latencyNs: link.latencyNs,
          concurrencyLanes: link.concurrencyLanes,
          transport: link.transport,
          logicalSourceId: link.sourceDomainId,
          logicalTargetId: link.targetDomainId,
          networkResourceIds: link.networkResourceIds,
          segmentIndex,
          segmentCount: path.length - 1,
          bidirectional: reverse !== undefined,
          logicalReverseId: reverse?.id,
        },
      };
    });
  });
  return { nodes, edges: [...accessEdges, ...linkEdges] };
}

function equivalentLinkContract(
  left: SimulationScenario["links"][number],
  right: SimulationScenario["links"][number],
): boolean {
  return left.kind === right.kind
    && left.bandwidthBytesPerSec === right.bandwidthBytesPerSec
    && left.latencyNs === right.latencyNs
    && left.concurrencyLanes === right.concurrencyLanes
    && left.transport === right.transport
    && [...(left.networkResourceIds ?? [])].sort().join("\0")
      === [...(right.networkResourceIds ?? [])].sort().join("\0");
}

function networkResourceNode(
  resource: NetworkResourceSpec,
  position: { readonly x: number; readonly y: number },
  nodeId = "shared-network",
): TopologyGraphNode {
  return {
    id: resource.id,
    type: "topology",
    position,
    ...(resource.nodeId === undefined
      ? {}
      : {
          parentId: `system:${resource.nodeId}`,
          extent: "parent" as const,
        }),
    data: {
      category: "network",
      title: resource.id,
      kind: resource.kind === "nic" ? "NIC / HCA" : "switch fabric",
      nodeId,
      accent: resource.kind === "nic" ? "nic" : "fabric",
      details: [
        `${formatRate(resource.bandwidthBytesPerSec)} · ${formatDuration(resource.latencyNs)}`,
        `${resource.concurrencyLanes} lane${resource.concurrencyLanes === 1 ? "" : "s"}`,
        compactList(resource.supportedTransports),
        resource.directMemoryDomainIds.length === 0
          ? "host staged"
          : `direct: ${compactList(resource.directMemoryDomainIds)}`,
      ],
    },
  };
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
