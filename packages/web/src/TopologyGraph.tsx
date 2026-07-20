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
type FlowNode = Node<FlowNodeData, "topology">;
type FlowEdge = Edge<FlowEdgeData>;

const nodeTypes = {
  topology: memo(TopologyNode),
};

export default function TopologyGraph({
  scenario,
  className = "h-[430px]",
}: {
  readonly scenario: SimulationScenario;
  readonly className?: string;
}): React.JSX.Element {
  const graph = useMemo(() => buildTopologyGraph(scenario), [scenario]);
  const nodes: FlowNode[] = useMemo(() => graph.nodes.map((node) => ({
    ...node,
    data: node.data as FlowNodeData,
  })), [graph.nodes]);
  const edges: FlowEdge[] = useMemo(() => graph.edges.map((edge) => ({
    ...edge,
    markerEnd: edge.markerEnd === undefined
      ? undefined
      : {
          type: MarkerType.ArrowClosed,
          color: edge.markerEnd.color,
        },
    data: edge.data as FlowEdgeData,
    labelStyle: {
      fill: "#52525b",
      fontSize: 10,
      fontWeight: 600,
    },
    labelBgStyle: {
      fill: "#ffffff",
      fillOpacity: 0.92,
    },
    labelBgPadding: [4, 2] as [number, number],
    labelBgBorderRadius: 2,
  })), [graph.edges]);
  const [selection, setSelection] = useState<
    | { readonly category: "node"; readonly data: FlowNodeData }
    | { readonly category: "edge"; readonly data: FlowEdgeData }
    | undefined
  >();

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
          data: node.data,
        })}
        onEdgeClick={(_event, edge) => {
          if (edge.data !== undefined) {
            setSelection({ category: "edge", data: edge.data });
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
        <Legend color="bg-sky-700" label="Compute" />
        <Legend color="bg-emerald-700" label="Memory" />
        <Legend color="bg-zinc-400" label="Access" dashed />
        <Legend color="bg-zinc-700" label="Directed link" />
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
                        {selection.data.kind} · {selection.data.nodeId}
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
                      <div className="mt-1 text-[11px] text-zinc-600">
                        {selection.data.category === "access"
                          ? "Device-visible memory relationship"
                          : `${formatRate(selection.data.bandwidthBytesPerSec!)} · ${formatDuration(selection.data.latencyNs!)} · ${selection.data.concurrencyLanes} lane${selection.data.concurrencyLanes === 1 ? "" : "s"}`}
                      </div>
                    </>
                  )}
            </div>
          )
        : null}
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
        "w-[210px] rounded-md border bg-white px-3 py-2 shadow-sm",
        selected ? "border-sky-600 ring-2 ring-sky-100" : "border-zinc-300",
      ].join(" ")}
    >
      <Handle
        id="top-target"
        type="target"
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
        : `size-2 rounded-sm ${color}`}
      />
      {label}
    </span>
  );
}

function accentColor(accent: TopologyGraphNodeData["accent"]): string {
  switch (accent) {
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
  }
}
