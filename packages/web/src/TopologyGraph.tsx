import { memo, useMemo, useState } from "react";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { SimulationScenario } from "@inference-sim/core";
import {
  buildTopologyGraph,
  formatDuration,
  formatRate,
  type TopologyGraphEdgeData,
  type TopologyGraphNodeData,
} from "./topology-graph.js";

type FlowNodeData = TopologyGraphNodeData & Record<string, unknown>;
type FlowEdgeData = TopologyGraphEdgeData & Record<string, unknown>;
type FlowNode = Node<FlowNodeData, "topology" | "topologyGroup">;
type FlowEdge = Edge<FlowEdgeData>;

const nodeTypes = {
  topology: memo(TopologyNode),
  topologyGroup: memo(TopologyGroupNode),
};
const PARALLEL_HANDLE_OFFSETS = Array.from(
  { length: 8 },
  (_value, index) => ((index + 1) * 100) / 9,
);

export default function TopologyGraph({
  scenario,
  className = "h-[430px]",
}: {
  readonly scenario: SimulationScenario;
  readonly className?: string;
}): React.JSX.Element {
  const graph = useMemo(() => buildTopologyGraph(scenario), [scenario]);
  const [selection, setSelection] = useState<
    | {
        readonly category: "node";
        readonly id: string;
        readonly data: FlowNodeData;
      }
    | {
        readonly category: "edge";
        readonly id: string;
        readonly data: FlowEdgeData;
      }
    | undefined
  >();
  const nodes: FlowNode[] = useMemo(() => graph.nodes.map((node) => ({
    ...node,
    data: node.data as FlowNodeData,
    selected: selection?.category === "node" && selection.id === node.id,
  })), [graph.nodes, selection]);
  const edges: FlowEdge[] = useMemo(() => graph.edges.map((edge) => {
    const selected = selection?.category === "edge" && selection.id === edge.id;
    return {
      ...edge,
      selected,
      zIndex: selected ? 20 : edge.data.category === "link" ? 2 : 1,
      style: {
        ...edge.style,
        cursor: "pointer",
        strokeWidth: selected
          ? Math.max(4, edge.style.strokeWidth + 2)
          : edge.style.strokeWidth,
        ...(selected
          ? { filter: "drop-shadow(0 0 2px rgba(24, 24, 27, 0.45))" }
          : {}),
      },
      markerEnd: edge.markerEnd === undefined
        ? undefined
        : {
            type: MarkerType.ArrowClosed,
            color: edge.markerEnd.color,
          },
      markerStart: edge.markerStart === undefined
        ? undefined
        : {
            type: MarkerType.ArrowClosed,
            color: edge.markerStart.color,
          },
      data: edge.data as FlowEdgeData,
      labelStyle: {
        fill: selected ? "#18181b" : "#52525b",
        fontSize: 10,
        fontWeight: selected ? 700 : 600,
      },
      labelBgStyle: {
        fill: selected ? "#fef3c7" : "#ffffff",
        fillOpacity: selected ? 1 : 0.92,
      },
      labelBgPadding: [4, 2] as [number, number],
      labelBgBorderRadius: 2,
    };
  }), [graph.edges, selection]);

  return (
    <div
      className={`relative min-w-0 overflow-hidden rounded-md border border-zinc-200 bg-white ${className}`}
      role="region"
      aria-label="Device topology map"
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.16, maxZoom: 1.1 }}
        minZoom={0.2}
        maxZoom={1.8}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        panOnDrag
        zoomOnDoubleClick={false}
        deleteKeyCode={null}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_event, node) => setSelection({
          category: "node",
          id: node.id,
          data: node.data,
        })}
        onEdgeClick={(_event, edge) => {
          if (edge.data !== undefined) {
            setSelection({ category: "edge", id: edge.id, data: edge.data });
          }
        }}
        onPaneClick={() => setSelection(undefined)}
      >
        <Background color="#d4d4d8" gap={20} size={1} />
        <Controls
          position="bottom-right"
          showInteractive={false}
          fitViewOptions={{ padding: 0.16 }}
        />
      </ReactFlow>

      <div className="pointer-events-none absolute left-3 top-3 flex flex-wrap gap-2">
        <Legend color="border-zinc-400 bg-zinc-100" label="System" />
        <Legend color="border-sky-300 bg-sky-100" label="Compute" />
        <Legend color="border-emerald-300 bg-emerald-100" label="Memory" />
        <Legend color="border-rose-300 bg-rose-100" label="NIC / fabric" />
        <Legend color="bg-emerald-600" label="Local memory" />
        <Legend color="bg-zinc-400" label="Accessible memory" dashed />
        <Legend color="bg-zinc-700" label="Physical link" />
      </div>

      {selection
        ? (
            <div className="pointer-events-none absolute bottom-3 left-3 max-w-[min(360px,calc(100%-5rem))] border border-zinc-200 bg-white/95 px-3 py-2 shadow-sm backdrop-blur-sm">
              {selection.category === "node"
                ? (
                    <>
                      <div className="truncate text-xs font-bold">
                        {selection.data.title}
                      </div>
                      <div className="mt-0.5 text-[11px] text-zinc-500">
                        {selection.data.kind} · {selection.data.category}
                      </div>
                      <div className="mt-1 text-[11px] text-zinc-600">
                        {selection.data.details.join(" · ")}
                      </div>
                    </>
                  )
                : (
                    <>
                      <div className="text-xs font-bold">
                        {selection.data.kind}
                      </div>
                      <div className="mt-0.5 text-[11px] font-semibold text-zinc-500">
                        {selection.data.scope}
                      </div>
                      <div className="mt-1 text-[11px] text-zinc-600">
                        {selection.data.category === "access"
                          ? selection.data.memoryRelation === "local"
                            ? "Device-owned local memory relationship"
                            : "Device-visible shared or remote memory relationship"
                          : [
                              selection.data.transport,
                              `${selection.data.logicalSourceId} ${selection.data.bidirectional ? "↔" : "→"} ${selection.data.logicalTargetId}`,
                              selection.data.networkResourceIds?.length
                                ? selection.data.networkResourceIds.join(" → ")
                                : undefined,
                              `${formatRate(selection.data.bandwidthBytesPerSec!)} · ${formatDuration(selection.data.latencyNs!)} · ${selection.data.concurrencyLanes} lane${selection.data.concurrencyLanes === 1 ? "" : "s"}`,
                            ].filter(Boolean).join(" · ")}
                      </div>
                    </>
                  )}
            </div>
          )
        : null}
    </div>
  );
}

function TopologyGroupNode({
  data,
  selected,
}: NodeProps<FlowNode>): React.JSX.Element {
  return (
    <div
      className={[
        "size-full rounded-md border bg-zinc-50/80",
        selected
          ? "border-zinc-700 ring-2 ring-zinc-200"
          : "border-zinc-300",
      ].join(" ")}
    >
      <div className="flex h-12 items-center justify-between gap-4 border-b border-zinc-200 px-4">
        <div className="flex min-w-0 items-center gap-2">
          <span className="size-2 shrink-0 rounded-sm bg-zinc-800" />
          <div className="truncate text-xs font-bold">{data.title}</div>
        </div>
        <div className="shrink-0 text-[10px] font-semibold uppercase text-zinc-500">
          physical system
        </div>
      </div>
    </div>
  );
}

function TopologyNode({
  data,
  selected,
}: NodeProps<FlowNode>): React.JSX.Element {
  return (
    <div
      className={[
        "w-[210px] rounded-md border px-3 py-2 shadow-sm",
        nodeSurfaceColor(data.category),
        nodeSelectionColor(data.category, selected),
      ].join(" ")}
    >
      <Handle
        id="top-target"
        type="target"
        position={Position.Top}
        className="!size-2 !border-white !bg-zinc-400"
      />
      <Handle
        id="top-source"
        type="source"
        position={Position.Top}
        className="!size-2 !border-white !bg-zinc-400"
      />
      <Handle
        id="left-target"
        type="target"
        position={Position.Left}
        className="!size-2 !border-white !bg-zinc-400"
      />
      <Handle
        id="right-target"
        type="target"
        position={Position.Right}
        className="!size-2 !border-white !bg-zinc-400"
      />
      <Handle
        id="bottom-source"
        type="source"
        position={Position.Bottom}
        className="!size-2 !border-white !bg-zinc-400"
      />
      <Handle
        id="bottom-target"
        type="target"
        position={Position.Bottom}
        className="!size-2 !border-white !bg-zinc-400"
      />
      <Handle
        id="left-source"
        type="source"
        position={Position.Left}
        className="!size-2 !border-white !bg-zinc-400"
      />
      <Handle
        id="right-source"
        type="source"
        position={Position.Right}
        className="!size-2 !border-white !bg-zinc-400"
      />
      {PARALLEL_HANDLE_OFFSETS.flatMap((left, index) => ([
        <Handle
          key={`top-source-lane-${index}`}
          id={`top-source-lane-${index}`}
          type="source"
          position={Position.Top}
          style={{ left: `${left}%` }}
          className="!size-1 !border-transparent !bg-transparent !opacity-0"
        />,
        <Handle
          key={`top-target-lane-${index}`}
          id={`top-target-lane-${index}`}
          type="target"
          position={Position.Top}
          style={{ left: `${left}%` }}
          className="!size-1 !border-transparent !bg-transparent !opacity-0"
        />,
        <Handle
          key={`bottom-source-lane-${index}`}
          id={`bottom-source-lane-${index}`}
          type="source"
          position={Position.Bottom}
          style={{ left: `${left}%` }}
          className="!size-1 !border-transparent !bg-transparent !opacity-0"
        />,
        <Handle
          key={`bottom-target-lane-${index}`}
          id={`bottom-target-lane-${index}`}
          type="target"
          position={Position.Bottom}
          style={{ left: `${left}%` }}
          className="!size-1 !border-transparent !bg-transparent !opacity-0"
        />,
      ]))}
      <div className="flex items-center gap-2">
        <span className={`size-2 shrink-0 rounded-sm ${accentColor(data.accent)}`} />
        <div className="min-w-0 flex-1 truncate text-xs font-bold">
          {data.title}
        </div>
        <div className="shrink-0 text-[10px] font-semibold uppercase text-zinc-500">
          {data.kind}
        </div>
      </div>
      <div className="mt-2 space-y-0.5">
        {data.details.slice(0, 3).map((detail) => (
          <div key={detail} className="truncate text-[10px] text-zinc-500">
            {detail}
          </div>
        ))}
      </div>
    </div>
  );
}

function Legend({
  color,
  label,
  dashed = false,
}: {
  readonly color: string;
  readonly label: string;
  readonly dashed?: boolean;
}): React.JSX.Element {
  return (
    <span className="flex items-center gap-1.5 border border-zinc-200 bg-white/90 px-2 py-1 text-[10px] font-semibold text-zinc-600">
      <span className={dashed
        ? "h-0 w-3 border-t border-dashed border-zinc-500"
        : `size-3 rounded-sm border ${color}`}
      />
      {label}
    </span>
  );
}

function nodeSurfaceColor(
  category: TopologyGraphNodeData["category"],
): string {
  switch (category) {
    case "device":
      return "bg-sky-50/95";
    case "memory":
      return "bg-emerald-50/95";
    case "network":
      return "bg-rose-50/95";
    case "system":
      return "bg-zinc-50/95";
  }
}

function nodeSelectionColor(
  category: TopologyGraphNodeData["category"],
  selected: boolean,
): string {
  if (selected) {
    switch (category) {
      case "device":
        return "border-sky-700 ring-2 ring-sky-200";
      case "memory":
        return "border-emerald-700 ring-2 ring-emerald-200";
      case "network":
        return "border-rose-700 ring-2 ring-rose-200";
      case "system":
        return "border-zinc-700 ring-2 ring-zinc-200";
    }
  }
  switch (category) {
    case "device":
      return "border-sky-300";
    case "memory":
      return "border-emerald-300";
    case "network":
      return "border-rose-300";
    case "system":
      return "border-zinc-300";
  }
}

function accentColor(accent: TopologyGraphNodeData["accent"]): string {
  switch (accent) {
    case "system":
      return "bg-zinc-800";
    case "cpu":
      return "bg-zinc-700";
    case "gpu":
      return "bg-sky-700";
    case "npu":
      return "bg-violet-700";
    case "host":
      return "bg-emerald-700";
    case "device":
      return "bg-cyan-700";
    case "unified":
      return "bg-teal-700";
    case "storage":
      return "bg-amber-700";
    case "nic":
      return "bg-rose-700";
    case "fabric":
      return "bg-fuchsia-700";
  }
}
