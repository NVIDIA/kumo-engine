// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2019-2025 webkid GmbH
// SPDX-License-Identifier: Apache-2.0 AND MIT

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import {
    Edge,
    EdgeProps,
    Handle,
    MiniMap,
    Node,
    NodeProps,
    Position,
    ReactFlow,
    ReactFlowProvider,
    useEdgesState,
    useNodesState,
    useReactFlow
} from "@xyflow/react";

import { Button } from "@/components/ui/button";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger
} from "@/components/ui/tooltip";
import { OpenInFull } from "@/icons";
import { cn } from "@/lib/utils";

import "@xyflow/react/dist/style.css";

// ─── ReactFlow controls icons ─────────────────────────────────────────────────
// These are kept local rather than added to src/icons/MaterialIcons/ because
// they are not Material Icons — they use ReactFlow's own internal SVG paths and
// viewBoxes (copied from @xyflow/react internals) and have no meaning outside
// the controls panel of this component.
const ZoomInIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
        <path d="M32 18.133H18.133V32h-4.266V18.133H0v-4.266h13.867V0h4.266v13.867H32z" />
    </svg>
);

const ZoomOutIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 5">
        <path d="M0 0h32v4.2H0z" />
    </svg>
);

const FitScreenIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 30">
        <path d="M3.692 4.63c0-.53.4-.938.939-.938h5.215V0H4.708C2.13 0 0 2.054 0 4.63v5.216h3.692V4.631zM27.354 0h-5.2v3.692h5.17c.53 0 .984.4.984.939v5.215H32V4.631A4.624 4.624 0 0027.354 0zm.954 24.83c0 .532-.4.94-.939.94h-5.215v3.768h5.215c2.577 0 4.631-2.13 4.631-4.707v-5.139h-3.692v5.139zm-23.677.94c-.531 0-.939-.4-.939-.94v-5.138H0v5.139c0 2.577 2.13 4.707 4.708 4.707h5.138V25.77H4.631z" />
    </svg>
);

// ─── Design tokens ────────────────────────────────────────────────────────────
export const HOP_COLORS = [
    "var(--color-pink-500)", // hop 0 – entity
    "var(--color-cyan-500)", // hop 1
    "var(--color-orange-500)", // hop 2
    "var(--color-lime-500)", // hop 3
    "var(--color-indigo-500)", // hop 4
    "var(--color-teal-500)", // hop 5
    "var(--color-mauve-500)" // hop 6
] as const;

// ─── Layout constants ─────────────────────────────────────────────────────────
const COL_SPACING = 160;
const ROW_SPACING = 80;
// Height of the HTML header row above the ReactFlow canvas.
// 6px top padding + 16px text (text-sm leading-4) + 6px bottom padding.
const HEADER_ROW_HEIGHT = 28;
// Gap between canvas top and topmost node (at zoom=1).
const TOP_MARGIN = 36;
// Header row height in fullscreen — extra padding makes labels feel less cramped.
const HEADER_ROW_HEIGHT_FULLSCREEN = 44;

// ─── Node sizing ──────────────────────────────────────────────────────────────
// Fixed 36×36 px circles regardless of neighbourCount.
const NODE_RADIUS = 18;
// Space below the bottom node's circle for the label (absolute top: diameter+6 = 6px gap,
// text-sm line-height 1.25rem ≈ 20px text). Without this the canvas clips the label and
// canvasHeight never exceeds availableHeight-28, so no scrollbar appears.
const NODE_LABEL_SPACE = 26;

// ─── Public types ─────────────────────────────────────────────────────────────
export interface HopGraphNode {
    id: string;
    table: string;
    hop: number;
    neighborCount?: number;
}

export interface HopGraphEdge {
    id: string;
    sourceId: string;
    targetId: string;
    samplingBudget?: number;
}

export interface HopGraphProps {
    nodes: HopGraphNode[];
    edges: HopGraphEdge[];
    className?: string;
    onOpenFullscreen?: () => void;
    /** Pass when rendering inside a fullscreen sheet. Shows MiniMap + Close
     *  button stacked in the sticky bar so they don't overlap the host UI. */
    onCloseFullscreen?: () => void;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function hopColor(hop: number): string {
    if (hop === 0) return HOP_COLORS[0];
    // Cycle through hop colors (indices 1–N), keeping entity pink unique.
    return HOP_COLORS[((hop - 1) % (HOP_COLORS.length - 1)) + 1];
}

function hopLabel(hop: number): string {
    return hop === 0 ? "Entity" : `Hop ${hop}`;
}

function truncate(s: string, max = 14): string {
    return s.length > max ? s.slice(0, max - 1) + "…" : s;
}

// Node data shape — must extend Record<string, unknown> for ReactFlow generics.
type HopCircleData = Record<string, unknown> & {
    table: string;
    hop: number;
    neighborCount: number | undefined;
    color: string;
    radius: number;
};

type HopCircleNode = Node<HopCircleData, "hopCircle">;

// ─── React Flow builders ──────────────────────────────────────────────────────
// Positions each hop-H node at the minimum y of its hop-(H-1) sources so that
// 1-to-1 exclusive connections land on the same horizontal line.
// Minimum ROW_SPACING between adjacent nodes in the same column is enforced.
function buildRFNodes(
    nodes: HopGraphNode[],
    edges: HopGraphEdge[]
): HopCircleNode[] {
    const byHop = new Map<number, HopGraphNode[]>();
    for (const n of nodes) {
        if (!byHop.has(n.hop)) byHop.set(n.hop, []);
        byHop.get(n.hop)!.push(n);
    }

    // targetId → [sourceId]
    const sourcesOf = new Map<string, string[]>();
    for (const e of edges) {
        if (!sourcesOf.has(e.targetId)) sourcesOf.set(e.targetId, []);
        sourcesOf.get(e.targetId)!.push(e.sourceId);
    }

    const nodeY = new Map<string, number>();
    const rfNodes: HopCircleNode[] = [];

    for (const hop of Array.from(byHop.keys()).sort((a, b) => a - b)) {
        const hopNodes = byHop.get(hop)!;
        const cx = hop * COL_SPACING;

        // Ideal y = topmost (minimum) source y; hop-0 entity starts at y=0.
        // Using min rather than average prevents multi-source nodes from landing
        // visually "between" their sources.
        const withIdeal = hopNodes.map((n) => {
            const srcs = sourcesOf.get(n.id) ?? [];
            const ideal =
                srcs.length === 0
                    ? 0
                    : Math.min(...srcs.map((id) => nodeY.get(id) ?? 0));
            return { n, ideal };
        });

        // Sort by ideal y, then push down any node that would overlap the previous.
        withIdeal.sort((a, b) => a.ideal - b.ideal);
        const assigned: number[] = [];
        for (let i = 0; i < withIdeal.length; i++) {
            assigned.push(
                i === 0
                    ? withIdeal[i].ideal
                    : Math.max(
                          withIdeal[i].ideal,
                          assigned[i - 1] + ROW_SPACING
                      )
            );
        }

        withIdeal.forEach(({ n }, i) => {
            nodeY.set(n.id, assigned[i]);
            const radius = NODE_RADIUS;
            const diameter = radius * 2;
            // Position is the node's top-left corner (nodeOrigin=[0,0] default).
            // cx and assigned[i] are the circle centre; subtract radius to get top-left.
            rfNodes.push({
                id: n.id,
                type: "hopCircle",
                position: { x: cx - radius, y: assigned[i] - radius },
                sourcePosition: Position.Right,
                targetPosition: Position.Left,
                width: diameter,
                height: diameter,
                data: {
                    table: n.table,
                    hop: n.hop,
                    neighborCount: n.neighborCount,
                    color: hopColor(n.hop),
                    radius
                },
                draggable: false,
                selectable: false,
                focusable: false
            });
        });
    }
    return rfNodes;
}

function buildRFEdges(edges: HopGraphEdge[]): Edge[] {
    return edges.map((e) => ({
        id: e.id,
        source: e.sourceId,
        target: e.targetId,
        type: "hopEdge",
        selectable: false,
        focusable: false
    }));
}

// ─── Custom edge — straight line ──────────────────────────────────────────
function HopEdge({ sourceX, sourceY, targetX, targetY }: EdgeProps) {
    return (
        <line
            x1={sourceX}
            y1={sourceY}
            x2={targetX}
            y2={targetY}
            stroke="var(--color-slate-300)"
            strokeWidth={1.75}
        />
    );
}

// ─── Custom node ──────────────────────────────────────────────────────────────
function getHandleStyle(
    side: "left" | "right",
    radius: number
): React.CSSProperties {
    return {
        position: "absolute",
        [side]: 0,
        top: radius - 0.5,
        width: 1,
        height: 1,
        transform: "none",
        opacity: 0,
        pointerEvents: "none",
        border: "none",
        background: "transparent",
        borderRadius: 0
    };
}

function HopCircleNode({ data }: NodeProps<HopCircleNode>) {
    const { table, color, radius } = data;
    const diameter = radius * 2;

    return (
        <div className="relative" style={{ width: diameter, height: diameter }}>
            <Handle
                type="target"
                position={Position.Left}
                style={getHandleStyle("left", radius)}
            />
            <Handle
                type="source"
                position={Position.Right}
                style={getHandleStyle("right", radius)}
            />
            <div
                className="rounded-full shadow-sm"
                style={{
                    width: diameter,
                    height: diameter,
                    backgroundColor: color
                }}
            />
            <div
                className="pointer-events-none select-none"
                style={{
                    position: "absolute",
                    top: diameter + 6,
                    left: "50%",
                    transform: "translateX(-50%)",
                    whiteSpace: "nowrap"
                }}>
                <span className="text-title-primary text-sm font-medium">
                    {truncate(table)}
                </span>
            </div>
        </div>
    );
}

const nodeTypes = { hopCircle: HopCircleNode };
const edgeTypes = { hopEdge: HopEdge };

// ─── Inner graph ──────────────────────────────────────────────────────────────
interface NodeTooltip {
    table: string;
    neighborCount?: number;
    x: number;
    y: number;
}

interface HopGraphInnerProps extends HopGraphProps {
    /** Height and width of the wrapping card body, measured by the public HopGraph. */
    availableHeight: number;
    availableWidth: number;
}

function HopGraphInner({
    nodes: inputNodes,
    edges: inputEdges,
    availableHeight,
    availableWidth,
    onOpenFullscreen,
    onCloseFullscreen
}: HopGraphInnerProps) {
    const { fitView, setViewport, zoomIn, zoomOut } = useReactFlow();

    // Track viewport so the HTML header row stays aligned during pan/zoom.
    const [rfViewport, setRFViewport] = useState({ x: 0, y: 0, zoom: 1 });
    const [nodeTooltip, setNodeTooltip] = useState<NodeTooltip | null>(null);
    // Hide the canvas until fitView has run once — prevents the "nodes at top → snap to centre" flicker.
    const [canvasVisible, setCanvasVisible] = useState(false);

    // In fullscreen the canvas is sized via CSS flex (flex-1 min-h-0) so there is no
    // overflow and no pixel arithmetic needed. In normal (card) mode the canvas gets an
    // explicit pixel height so the card can scroll when there are many nodes.
    const isFullscreen = !!onCloseFullscreen;

    // False until the first viewport has been committed. Ensures the initial fit always
    // snaps instantly (duration=0) so the canvas never becomes visible mid-animation
    // from the ReactFlow default position {x:0,y:0,zoom:1} to the correct viewport.
    // Resets to false on every remount (key change), which happens when hopCount changes.
    const hasSetViewportOnceRef = useRef(false);

    // Measure the controls bar only in normal mode — used to compute canvasHeight.
    const controlsBarRef = useRef<HTMLDivElement>(null);
    const [controlsBarHeight, setControlsBarHeight] = useState(0);
    useEffect(() => {
        if (isFullscreen) return;
        const el = controlsBarRef.current;
        if (!el) return;
        const ro = new ResizeObserver(() =>
            setControlsBarHeight(el.offsetHeight)
        );
        ro.observe(el);
        return () => ro.disconnect();
    }, [isFullscreen]);

    const [rfNodes, setRFNodes, onNodesChange] = useNodesState<HopCircleNode>(
        buildRFNodes(inputNodes, inputEdges)
    );
    const [rfEdges, setRFEdges, onEdgesChange] = useEdgesState<Edge>(
        buildRFEdges(inputEdges)
    );

    // Hop list and per-hop counts — needed for canvas sizing, zoom limits, and doFit.
    const hops = useMemo(
        () =>
            Array.from(new Set(inputNodes.map((n) => n.hop))).sort(
                (a, b) => a - b
            ),
        [inputNodes]
    );

    const maxPerHop = useMemo(() => {
        const hopCounts = new Map<number, number>();
        inputNodes.forEach((n) =>
            hopCounts.set(n.hop, (hopCounts.get(n.hop) ?? 0) + 1)
        );
        return Math.max(...Array.from(hopCounts.values()), 1);
    }, [inputNodes]);

    // Minimum zoom: prevent zooming out so far that hop column headers overlap.
    // ~8.5px per character at text-sm; add a small margin and cap between 0.2–0.5.
    const minZoom = useMemo(() => {
        if (hops.length === 0) return 0.2;
        const maxLabelPx =
            Math.max(...hops.map((h) => hopLabel(h).length)) * 8.5;
        return Math.min(0.5, Math.max(0.2, maxLabelPx / COL_SPACING + 0.05));
    }, [hops]);

    // Minimum canvas height — exact node bounding box at zoom=1 with the top margin.
    // Controls overlay the canvas via negative marginTop (zero net flex height), so
    // they are NOT included here. Adding controlsBarHeight would grow the canvas past
    // availableHeight after the first ResizeObserver tick, creating a spurious scrollbar.
    const minCanvasHeight =
        TOP_MARGIN +
        (maxPerHop - 1) * ROW_SPACING +
        NODE_RADIUS * 2 +
        NODE_LABEL_SPACE;

    // Canvas height (normal/card mode only):
    //   • Controls overlay the canvas bottom via negative marginTop, so canvasHeight
    //     fills the full available area minus just the header row.
    //   • When minCanvasHeight exceeds that, the canvas grows past the card → scrollbar.
    const canvasHeight = isFullscreen
        ? 0
        : Math.max(
              minCanvasHeight,
              availableHeight > 0 ? availableHeight - HEADER_ROW_HEIGHT : 0
          );

    // Re-fit whenever the layout changes (resize, new data, or fullscreen mount).
    // Normal mode: manual setViewport — anchors topmost node TOP_MARGIN px from the
    // canvas top; zoom driven by width only so nodes render at full (zoom=1) size.
    // Fullscreen: delegate to fitView (canvas is CSS-sized, no pixel arithmetic needed).
    const doFit = useCallback(
        (animated = true) => {
            // First fit after mount/remount always snaps instantly — prevents the canvas
            // from becoming visible mid-animation from ReactFlow's default {x:0,y:0,zoom:1}.
            const duration =
                animated && hasSetViewportOnceRef.current ? 200 : 0;

            if (isFullscreen) {
                fitView({ padding: 0.2, maxZoom: 1, duration });
                hasSetViewportOnceRef.current = true;
                requestAnimationFrame(() => setCanvasVisible(true));
                return;
            }
            if (availableWidth <= 0 || canvasHeight <= 0) return;

            const hPad = 40; // px horizontal padding each side
            const maxHop = hops.length > 0 ? hops[hops.length - 1] : 0;
            const graphW = maxHop * COL_SPACING + NODE_RADIUS * 2;

            // Zoom is driven by width only — minCanvasHeight guarantees enough
            // vertical space so nodes are never shrunk to fit a short card.
            const fitW = availableWidth - 2 * hPad;
            const rawZoom = graphW > 0 && fitW > 0 ? fitW / graphW : 1;
            const zoom = Math.min(Math.max(rawZoom, minZoom), 1.0);

            // Center horizontally; anchor topmost node (world y = -NODE_RADIUS) at TOP_MARGIN.
            const vx =
                (availableWidth - graphW * zoom) / 2 + NODE_RADIUS * zoom;
            const vy = TOP_MARGIN + NODE_RADIUS * zoom;

            hasSetViewportOnceRef.current = true;
            setViewport({ x: vx, y: vy, zoom }, { duration });
            requestAnimationFrame(() => setCanvasVisible(true));
        },
        [
            isFullscreen,
            fitView,
            setViewport,
            availableWidth,
            canvasHeight,
            hops,
            minZoom
        ]
    );

    // Stable ref to the latest doFit — lets effects call it without listing doFit
    // as a dep, so a resize (which recreates doFit) never re-triggers the data-sync
    // effect and never flashes the canvas invisible.
    const doFitRef = useRef(doFit);
    doFitRef.current = doFit;

    // Sync nodes/edges only when graph data changes — hides canvas briefly to mask repaint.
    useEffect(() => {
        setCanvasVisible(false);
        setRFNodes(buildRFNodes(inputNodes, inputEdges));
        setRFEdges(buildRFEdges(inputEdges));
        const t = setTimeout(() => doFitRef.current(true), 50);
        return () => clearTimeout(t);
    }, [inputNodes, inputEdges, setRFNodes, setRFEdges]);

    // Re-fit on dimension changes (initial render after ResizeObserver + window/card resize).
    useEffect(() => {
        if (!availableWidth) return;
        const t = setTimeout(() => doFitRef.current(false), 50);
        return () => clearTimeout(t);
    }, [availableWidth, availableHeight, canvasHeight]);

    const legendItems = useMemo(() => {
        const seen = new Set<number>();
        return inputNodes
            .filter((n) => !seen.has(n.hop) && seen.add(n.hop))
            .sort((a, b) => a.hop - b.hop)
            .map((n) => ({
                hop: n.hop,
                color: hopColor(n.hop),
                label: hopLabel(n.hop)
            }));
    }, [inputNodes]);

    return (
        <div className={cn("flex w-full flex-col", isFullscreen && "h-full")}>
            {/* ── Row 1: hop column headers (HTML, aligned via live viewport) ── */}
            <div
                className="pointer-events-none relative shrink-0 overflow-hidden"
                style={{
                    height: isFullscreen
                        ? HEADER_ROW_HEIGHT_FULLSCREEN
                        : HEADER_ROW_HEIGHT
                }}>
                {hops.map((hop) => (
                    <span
                        key={hop}
                        className="text-title-secondary absolute select-none font-sans text-sm font-medium leading-4"
                        style={{
                            left:
                                rfViewport.x +
                                hop * COL_SPACING * rfViewport.zoom,
                            transform: "translateX(-50%)",
                            top: "50%",
                            marginTop: "-0.5em"
                        }}>
                        {hopLabel(hop)}
                    </span>
                ))}
            </div>

            {/* ── Row 2: ReactFlow canvas ────────────────────────────────────── */}
            {/* Fullscreen: flex-1 min-h-0 lets CSS fill remaining space exactly. */}
            {/* Normal:     explicit pixel height so the card can scroll for large graphs. */}
            {(isFullscreen || canvasHeight > 0) && (
                <div
                    className={
                        isFullscreen ? "relative min-h-0 flex-1" : "relative"
                    }
                    style={
                        isFullscreen
                            ? { opacity: canvasVisible ? 1 : 0 }
                            : {
                                  height: canvasHeight,
                                  opacity: canvasVisible ? 1 : 0
                              }
                    }>
                    <ReactFlow
                        nodes={rfNodes}
                        edges={rfEdges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        nodeTypes={nodeTypes}
                        edgeTypes={edgeTypes}
                        proOptions={{ hideAttribution: true }}
                        minZoom={minZoom}
                        maxZoom={1.4}
                        zoomOnScroll={false}
                        preventScrolling={false}
                        className="border-none"
                        onViewportChange={(vp) => setRFViewport(vp)}
                        onNodeMouseEnter={(e, node) => {
                            const n = node as HopCircleNode;
                            if (n.type !== "hopCircle") return;
                            setNodeTooltip({
                                table: n.data.table,
                                neighborCount: n.data.neighborCount,
                                x: e.clientX,
                                y: e.clientY
                            });
                        }}
                        onNodeMouseMove={(e, node) => {
                            if ((node as HopCircleNode).type !== "hopCircle")
                                return;
                            setNodeTooltip((prev) =>
                                prev
                                    ? { ...prev, x: e.clientX, y: e.clientY }
                                    : prev
                            );
                        }}
                        onNodeMouseLeave={(_, node) => {
                            if ((node as HopCircleNode).type !== "hopCircle")
                                return;
                            setNodeTooltip(null);
                        }}></ReactFlow>

                    {nodeTooltip &&
                        createPortal(
                            <TooltipProvider>
                                <Tooltip open>
                                    <TooltipTrigger asChild>
                                        <div
                                            style={{
                                                position: "fixed",
                                                left: nodeTooltip.x,
                                                top: nodeTooltip.y,
                                                width: 1,
                                                height: 1,
                                                pointerEvents: "none"
                                            }}
                                        />
                                    </TooltipTrigger>
                                    <TooltipContent
                                        style={{ pointerEvents: "none" }}
                                        side="top"
                                        sideOffset={20}>
                                        <div className="font-sans text-sm font-medium leading-5 text-white">
                                            Table: {nodeTooltip.table}
                                            {nodeTooltip.neighborCount !==
                                                undefined && (
                                                <>
                                                    <br />
                                                    Neighbor budget:{" "}
                                                    {nodeTooltip.neighborCount}
                                                </>
                                            )}
                                        </div>
                                    </TooltipContent>
                                </Tooltip>
                            </TooltipProvider>,
                            document.body
                        )}
                </div>
            )}

            {/* ── Bottom bar — controls+legend left, minimap/close right ────── */}
            {/* Fullscreen: shrink-0 in the flex column.                          */}
            {/* Normal:     negative marginTop pulls it up to overlay the canvas  */}
            {/*             bottom; sticky bottom-0 keeps it pinned when scrolling.*/}
            <div
                ref={controlsBarRef}
                style={{ marginTop: isFullscreen ? 0 : -controlsBarHeight }}
                className={cn(
                    "pointer-events-none z-10 flex w-full items-end justify-between px-2 pb-2",
                    isFullscreen ? "shrink-0" : "sticky bottom-0"
                )}>
                {/* Left: zoom controls + legend */}
                <div className="pointer-events-auto flex items-end gap-2">
                    <div
                        className="react-flow__controls border-border-secondary border"
                        style={{
                            position: "static",
                            transform: "none",
                            boxShadow: "none"
                        }}>
                        <button
                            className="react-flow__controls-button react-flow__controls-zoomin"
                            onClick={() => zoomIn({ duration: 200 })}
                            aria-label="Zoom in"
                            title="Zoom in">
                            <ZoomInIcon />
                        </button>
                        <button
                            className="react-flow__controls-button react-flow__controls-zoomout"
                            onClick={() => zoomOut({ duration: 200 })}
                            aria-label="Zoom out"
                            title="Zoom out">
                            <ZoomOutIcon />
                        </button>
                        <button
                            className="react-flow__controls-button react-flow__controls-fitview"
                            onClick={() => doFit()}
                            aria-label="Fit view"
                            title="Fit view">
                            <FitScreenIcon />
                        </button>
                        {onOpenFullscreen && (
                            <button
                                className="react-flow__controls-button [&_svg]:size-3"
                                onClick={onOpenFullscreen}
                                aria-label="Open full screen"
                                title="Open full screen">
                                <OpenInFull className="fill-slate-800! text-slate-800!" />
                            </button>
                        )}
                    </div>
                    <div className="border-border-secondary grid grid-flow-col grid-rows-3 gap-x-3 gap-y-1 rounded border bg-white/90 px-4 py-2 shadow-sm backdrop-blur-sm">
                        {legendItems.map(({ hop, color, label }) => (
                            <div key={hop} className="flex items-center gap-1">
                                <span
                                    className="inline-block size-2.5 shrink-0 rounded-full"
                                    style={{ backgroundColor: color }}
                                />
                                <span className="text-xs text-slate-600">
                                    {label}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
                {/* Right: minimap always shown; Close button stacked below in fullscreen */}
                <div className="pointer-events-auto flex flex-col items-end gap-1">
                    <MiniMap
                        style={{
                            position: "relative",
                            bottom: "auto",
                            right: "auto",
                            margin: 0,
                            background: "var(--background)",
                            width: 52,
                            height: 52
                        }}
                        maskColor="var(--kumo-grey-50)"
                        nodeColor="var(--kumo-grey-300)"
                        className="[&_svg]:size-13"
                    />
                    {onCloseFullscreen && (
                        <Button
                            variant="outline"
                            className="w-35"
                            onClick={onCloseFullscreen}>
                            Close
                        </Button>
                    )}
                </div>
            </div>
        </div>
    );
}

// ─── Public component ─────────────────────────────────────────────────────────
/**
 * HopGraph — visualises neighbourhood sampling of a GNN model.
 *
 * Layout:
 *   Row 1 — HTML hop-column headers, aligned with the ReactFlow viewport.
 *   Row 2 — ReactFlow canvas (explicit pixel height).
 *            • Few nodes  → canvas fills the card; fitView centres content.
 *            • Many nodes → canvas grows past card height; Card scrolls.
 *
 * Full-screen: consumer wraps HopGraph in a Sheet with side="top". Pass
 * onOpenFullscreen to show the expand button; pass onCloseFullscreen when
 * rendering inside the sheet — it shows the Close button in the bottom bar.
 */
export function HopGraph({
    nodes,
    edges,
    className,
    onOpenFullscreen,
    onCloseFullscreen
}: HopGraphProps) {
    // Measure the wrapper so the inner component can size and position the canvas.
    const wrapperRef = useRef<HTMLDivElement>(null);
    const [availableHeight, setAvailableHeight] = useState(0);
    const [availableWidth, setAvailableWidth] = useState(0);

    useEffect(() => {
        const el = wrapperRef.current;
        if (!el) return;
        const ro = new ResizeObserver(() => {
            setAvailableHeight(el.clientHeight);
            setAvailableWidth(el.clientWidth);
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    const hopCount = useMemo(
        () => new Set(nodes.map((n) => n.hop)).size,
        [nodes]
    );

    return (
        <ReactFlowProvider>
            <div
                ref={wrapperRef}
                data-slot="hop-graph"
                className={cn("h-full w-full", className)}>
                <HopGraphInner
                    key={hopCount}
                    nodes={nodes}
                    edges={edges}
                    availableHeight={availableHeight}
                    availableWidth={availableWidth}
                    onOpenFullscreen={onOpenFullscreen}
                    onCloseFullscreen={onCloseFullscreen}
                />
            </div>
        </ReactFlowProvider>
    );
}
