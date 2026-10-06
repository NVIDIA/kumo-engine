// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef } from "react";

import {
    Background,
    BackgroundVariant,
    ControlButton,
    Controls,
    MiniMap,
    type MiniMapNodeProps,
    type Node,
    Panel,
    ReactFlow,
    ReactFlowProvider,
    useReactFlow,
    useViewport
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { Button } from "@/components/ui/button";
import { OpenInFull } from "@/icons";
import {
    type PredictionItemNodeData,
    type SubgraphNodeData,
    type XaiSubgraphProps
} from "@/types/visual-graph";
import {
    buildPredictionsPanelEdge,
    buildPredictionsPanelNodes,
    PREDICTION_ITEM_PREFIX
} from "@/utils/xai-subgraph-panel-layout";

import {
    PredictionItemNode,
    PredictionsPanelBackgroundNode
} from "./predictions-panel";
import { LegendPopover } from "./xai-subgraph-legend";
import {
    CollapseChipNode,
    CollapseExpandChipNode,
    SubgraphNode
} from "./xai-subgraph-nodes";

const NODE_TYPES = {
    subgraphNode: SubgraphNode,
    collapseChip: CollapseChipNode,
    collapseExpandChip: CollapseExpandChipNode,
    predictionsPanelBackground: PredictionsPanelBackgroundNode,
    predictionItem: PredictionItemNode
};

const miniMapNodeColor = (n: Node) => {
    if (n.type === "predictionItem") {
        return (
            (n.data as PredictionItemNodeData).item.color ??
            "var(--color-slate-400)"
        );
    }
    return (n.data as SubgraphNodeData).color ?? "var(--color-slate-400)";
};

// Match LHS node visual size in the minimap (LHS nodes are ~14-20px square).
const MINIMAP_ITEM_SIZE = 10;

function MiniMapNodeRenderer({
    id,
    x,
    y,
    width,
    height,
    color,
    borderRadius
}: MiniMapNodeProps) {
    if (id.startsWith(PREDICTION_ITEM_PREFIX)) {
        const cx = x + width / 2;
        const cy = y + height / 2;
        return (
            <rect
                x={cx - MINIMAP_ITEM_SIZE / 2}
                y={cy - MINIMAP_ITEM_SIZE / 2}
                width={MINIMAP_ITEM_SIZE}
                height={MINIMAP_ITEM_SIZE}
                rx={MINIMAP_ITEM_SIZE / 2}
                style={{ fill: color }}
            />
        );
    }
    return (
        <rect
            x={x}
            y={y}
            width={width}
            height={height}
            rx={borderRadius}
            ry={borderRadius}
            style={{ fill: color }}
        />
    );
}

// Each canvas gets its own ReactFlowProvider so the fullscreen canvas's unmount
// doesn't reset the normal canvas's viewport (they briefly coexist during the
// Sheet exit animation).
export function XaiSubgraph(props: XaiSubgraphProps) {
    return (
        <ReactFlowProvider>
            <XaiSubgraphInner {...props} />
        </ReactFlowProvider>
    );
}

function XaiSubgraphInner({ tierLabels = [], ...props }: XaiSubgraphProps) {
    const { fitView } = useReactFlow();
    const { zoom: legendZoom } = useViewport();
    const isFullscreen = !!props.onCloseFullscreen;

    const { predictions } = props;

    const mergedNodes = useMemo(
        () =>
            predictions
                ? [
                      ...props.displayNodes,
                      ...buildPredictionsPanelNodes(
                          props.displayNodes,
                          predictions
                      )
                  ]
                : props.displayNodes,
        [props.displayNodes, predictions]
    );

    const mergedEdges = useMemo(
        () =>
            predictions
                ? [
                      ...props.displayEdges,
                      buildPredictionsPanelEdge(predictions.entityNodeId)
                  ]
                : props.displayEdges,
        [props.displayEdges, predictions]
    );

    const wrapperRef = useRef<HTMLDivElement | null>(null);
    const hasInitialFit = useRef(false);

    // ResizeObserver-driven initial fit + resize re-fits. 100ms debounce
    // coalesces frames while a parent animates to its final size.
    useEffect(() => {
        const el = wrapperRef.current;
        if (!el) return;
        let timer: ReturnType<typeof setTimeout> | undefined;
        const ro = new ResizeObserver(() => {
            if (timer) clearTimeout(timer);
            timer = setTimeout(() => {
                fitView({ padding: 0.15, duration: 0 });
                hasInitialFit.current = true;
            }, 100);
        });
        ro.observe(el);
        return () => {
            ro.disconnect();
            if (timer) clearTimeout(timer);
        };
    }, [fitView]);

    // Animated re-fit on filter/expand changes (after the initial fit).
    const nodeIds = mergedNodes.map((n) => n.id).join(",");
    useEffect(() => {
        if (!hasInitialFit.current) return;
        if (nodeIds.length === 0) return;
        fitView({ padding: 0.15, duration: 250 });
    }, [nodeIds, fitView]);

    return (
        <div ref={wrapperRef} className="relative size-full">
            <ReactFlow
                nodes={mergedNodes}
                edges={mergedEdges}
                nodeTypes={NODE_TYPES}
                onNodeMouseEnter={props.onNodeMouseEnter}
                onNodeMouseLeave={props.onNodeMouseLeave}
                onNodeClick={props.onNodeClick}
                onPaneClick={props.onPaneClick}
                nodesDraggable={false}
                nodesConnectable={false}
                elementsSelectable={false}
                panOnDrag
                zoomOnScroll
                preventScrolling
                proOptions={{ hideAttribution: true }}
                defaultEdgeOptions={{
                    type: "default",
                    style: { stroke: "var(--color-slate-300)", strokeWidth: 1 }
                }}>
                <Background
                    variant={BackgroundVariant.Dots}
                    gap={16}
                    size={1.2}
                    color="var(--color-slate-300)"
                />
                {isFullscreen ? (
                    <Panel position="bottom-left" className="mb-2! ml-2!">
                        <div className="flex items-end gap-2">
                            <Controls
                                showInteractive={false}
                                style={{ position: "relative", margin: 0 }}
                            />
                            <LegendPopover
                                isOpen={props.isLegendOpen}
                                setIsOpen={props.onLegendOpenChange}
                                items={props.legendItems}
                                zoom={legendZoom}
                                tierLabels={tierLabels}
                                predictedTypesSampleColor={
                                    predictions?.items[0]?.color
                                }
                            />
                        </div>
                    </Panel>
                ) : (
                    <>
                        <Controls
                            position="bottom-right"
                            showInteractive={false}
                            style={{ marginBottom: 8 }}>
                            <ControlButton
                                onClick={props.onOpenFullscreen}
                                title="Fullscreen">
                                <OpenInFull className="size-3!" />
                            </ControlButton>
                        </Controls>
                        <Panel position="bottom-left" className="mb-2! ml-2!">
                            <LegendPopover
                                isOpen={props.isLegendOpen}
                                setIsOpen={props.onLegendOpenChange}
                                items={props.legendItems}
                                zoom={legendZoom}
                                tierLabels={tierLabels}
                                predictedTypesSampleColor={
                                    predictions?.items[0]?.color
                                }
                            />
                        </Panel>
                    </>
                )}
                {isFullscreen ? (
                    <Panel position="bottom-right" className="mb-2! mr-2!">
                        <div className="flex flex-col items-end gap-2">
                            <MiniMap
                                nodeColor={miniMapNodeColor}
                                nodeComponent={MiniMapNodeRenderer}
                                style={{
                                    position: "relative",
                                    bottom: "auto",
                                    right: "auto",
                                    margin: 0,
                                    width: 80,
                                    height: 60
                                }}
                            />
                            <Button
                                variant="outline"
                                className="w-20"
                                onClick={props.onCloseFullscreen}>
                                Close
                            </Button>
                        </div>
                    </Panel>
                ) : (
                    <MiniMap
                        nodeColor={miniMapNodeColor}
                        nodeComponent={MiniMapNodeRenderer}
                        position="bottom-right"
                        style={{
                            width: 80,
                            height: 60,
                            margin: 0,
                            right: 49,
                            bottom: 8
                        }}
                    />
                )}
            </ReactFlow>
        </div>
    );
}
