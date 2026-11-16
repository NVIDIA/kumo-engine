// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MouseEventHandler, useEffect, useRef, useState } from "react";

import {
    Background,
    BaseEdge,
    ControlButton,
    Controls,
    Edge,
    EdgeProps,
    EdgeTypes,
    getBezierPath,
    Handle,
    MiniMap,
    Node,
    NodeTypes,
    OnConnect,
    OnEdgesChange,
    OnNodesChange,
    Position,
    ReactFlow,
    ReactFlowProvider,
    useReactFlow
} from "@xyflow/react";

import { Skeleton } from "@/components/ui/skeleton";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger
} from "@/components/ui/tooltip";
import { Edit, OpenInFull, Visibility } from "@/icons";
import { cn } from "@/lib/utils";
import { TableGraphNodeProps } from "@/types/visual-graph";
import "@xyflow/react/dist/style.css";
import { createPortal } from "react-dom";

interface TableGraphProps {
    proOptions: {
        hideAttribution: boolean;
    };
    nodes: Node[];
    edges: Edge[];
    onNodesChange: OnNodesChange;
    onEdgesChange: OnEdgesChange;
    onConnect: OnConnect;
    nodeTypes: NodeTypes;
    edgeTypes: EdgeTypes;
    height: number;
    width: number;
    selectedNode: Node | null;
    setSelectedNode: React.Dispatch<React.SetStateAction<Node | null>>;
    hasData: boolean;
    handleModalOpen: MouseEventHandler<HTMLButtonElement>;
}

export const TableGraph = (props: TableGraphProps) => {
    return (
        <ReactFlowProvider>
            <TableGraphInt {...props} />
        </ReactFlowProvider>
    );
};

const TableGraphInt = ({
    proOptions,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    nodeTypes,
    edgeTypes,
    selectedNode,
    setSelectedNode,
    hasData,
    handleModalOpen
}: TableGraphProps) => {
    const { fitView } = useReactFlow();
    const containerRef = useRef<HTMLDivElement>(null);
    const hasFittedView = useRef(false);

    useEffect(() => {
        hasFittedView.current = false;
    }, [nodes?.length, edges?.length]);

    useEffect(() => {
        if (
            nodes?.length > 0 &&
            !hasFittedView.current &&
            containerRef.current &&
            containerRef.current.offsetWidth > 0 &&
            containerRef.current.offsetHeight > 0
        ) {
            fitView({ padding: 0.1, maxZoom: nodes?.length === 1 ? 1.5 : 2 });
            hasFittedView.current = true;
        }
    }, [nodes, fitView]);

    return (
        <div ref={containerRef} className="h-full w-full">
            <ReactFlow
                nodeExtent={[
                    [-2500, -2200],
                    [2500, 2200]
                ]} //To limit the node drag area, so that it doesn't go out of view.
                proOptions={proOptions}
                nodes={nodes}
                nodeTypes={nodeTypes}
                onNodesChange={onNodesChange}
                onNodeMouseMove={(event, node) => {
                    if ((event?.target as HTMLElement)?.id === "pkey") {
                        setSelectedNode(node);
                    } else if ((event?.target as HTMLElement)?.id === "fkey") {
                        const findEdges = edges?.filter(
                            (edge) =>
                                edge?.targetHandle ===
                                    "target_" +
                                        (event?.target as HTMLElement)
                                            ?.innerText &&
                                edge?.target === node?.id
                        );
                        if (findEdges && findEdges.length > 0) {
                            const findNode = nodes?.filter(
                                (node) => node?.id === findEdges[0]?.source
                            );
                            setSelectedNode(findNode[0]);
                        }
                    } else {
                        if (selectedNode) {
                            setSelectedNode(null);
                        }
                    }
                }}
                onNodeMouseLeave={() => {
                    setSelectedNode(null);
                }}
                onEdgeMouseEnter={(_event, edge) => {
                    const findNode = nodes?.filter(
                        (node) => node?.id === edge?.source
                    );
                    setSelectedNode(findNode[0]);
                }}
                onEdgeMouseLeave={() => {
                    setSelectedNode(null);
                }}
                edges={edges}
                edgeTypes={edgeTypes}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                zoomOnScroll={false}
                preventScrolling={false}
                fitView={true}
                minZoom={0.5}
                maxZoom={2}
                onlyRenderVisibleElements={false}>
                <Background />
                <Controls>
                    <ControlButton
                        onClick={handleModalOpen}
                        aria-label="Open in full screen">
                        <OpenInFull className="!text-text-primary size-3 !fill-slate-800" />
                    </ControlButton>
                </Controls>
                <MiniMap className="[&_svg]:size-13" />
                {!hasData && (
                    <div className="flex items-center justify-center">
                        <Skeleton className="h-40 w-40 rounded-sm" />
                    </div>
                )}
            </ReactFlow>
        </div>
    );
};

export const TableGraphNode = ({
    data,
    enableEditTable,
    onEditTableClick
}: TableGraphNodeProps) => {
    const pkey: string = (data?.pkey && `PK: ${data?.pkey}`) || "";
    const foreignKeyCols: Array<string> = data?.cols
        ?.filter(({ name }: { name: string }) =>
            data?.foreignKeys?.includes(name as string)
        )
        ?.map(({ name }: { name: string }) => `FK: ${name}`);

    const timeCol: string = (data?.timeCol && `T: ${data.timeCol}`) || "";

    const handleClasses =
        "!absolute !top-1/2 !-translate-y-1/2 !w-2 !h-2 !border-0 !rounded-full";
    const sourceHandleClasses = "!-right-2";
    const targetHandleClasses = "!-left-2";

    return (
        <div className="min-w-63 w-fit overflow-hidden rounded-sm">
            <div className="flex flex-row justify-between !rounded-t-sm border !border-b-0 border-slate-500 bg-slate-100 px-3 py-2">
                <span className="text-text-secondary px-1 py-1 text-base font-bold">
                    {data?.label}
                </span>
                <span className="ml-2 flex items-center gap-2">
                    {enableEditTable && (
                        <button
                            type="button"
                            aria-label={`Edit ${data?.label} table`}
                            className="cursor-pointer"
                            id={`${data?.id}_edit_icon`}
                            data-testid={`${data?.id}_edit_icon`}
                            onClick={() => {
                                onEditTableClick(data?.id);
                            }}>
                            <Edit className="size-4" />
                        </button>
                    )}
                    <button
                        type="button"
                        aria-label={`View ${data?.label} table details`}
                        className="cursor-pointer"
                        id={`${data?.id}_show_icon`}
                        data-testid={`${data?.id}_show_icon`}
                        onClick={() => {
                            data?.setSelectedTable(data?.id);
                        }}>
                        <Visibility className="fill-title-secondary size-4" />
                    </button>
                </span>
            </div>

            <div className="!rounded-b-sm border !border-t-0 border-slate-500 bg-white px-3 py-2">
                {data?.pkey && (
                    <div
                        className={cn(
                            "relative flex h-10 min-h-[2.5rem] items-center",
                            "font-base text-text-primary rounded-sm px-1 text-base",
                            "hover:bg-kumo-grey-50",
                            data?.pkeySelected ? "!bg-kumo-grey-50" : ""
                        )}
                        id="pkey">
                        {pkey}
                        <Handle
                            type="source"
                            position={Position.Right}
                            className={cn(
                                handleClasses,
                                sourceHandleClasses,
                                data?.pkeySelected
                                    ? "!bg-slate-800"
                                    : "!bg-slate-500"
                            )}
                            id={"source_" + pkey}
                        />
                    </div>
                )}

                {data?.timeCol && (
                    <div className="font-base text-text-primary flex h-10 items-center px-1 text-base">
                        {timeCol}
                    </div>
                )}

                {foreignKeyCols.map?.((col: string) => {
                    const highLight = data?.highLightKeys?.includes(
                        "target_" + col
                    );
                    return (
                        <div
                            className={cn(
                                "relative flex h-10 min-h-[2.5rem] items-center",
                                "font-base text-text-primary rounded-sm px-1 text-base",
                                "hover:bg-kumo-grey-50",
                                highLight ? "!bg-kumo-grey-50" : ""
                            )}
                            key={data?.label + col}>
                            {col}
                            <Handle
                                type="target"
                                position={Position.Left}
                                className={cn(
                                    handleClasses,
                                    targetHandleClasses,
                                    highLight
                                        ? "!bg-slate-800"
                                        : "!bg-slate-500"
                                )}
                                id={"target_" + col}
                            />
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

// Edge color constant for weak edges (healthy edges use ReactFlow's default gray)
const WEAK_EDGE_COLOR = "#E1A42F"; // Orange/amber - indicates low data overlap

export const TableGraphEdge: React.FC<EdgeProps> = ({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    data,
    markerEnd,
    style
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

    const [edgePath] = getBezierPath({
        sourceX: sourceX,
        sourceY: sourceY,
        sourcePosition: sourcePosition,
        targetX: targetX,
        targetY: targetY,
        targetPosition: targetPosition
    });

    const handleMouseMove = (e: React.MouseEvent) => {
        setMousePos({ x: e.clientX, y: e.clientY });
    };

    // Apply weak edge styling if data.isWeakEdge is true
    // Healthy edges use ReactFlow's default gray color
    const edgeStyle = data?.isWeakEdge
        ? { ...style, stroke: WEAK_EDGE_COLOR, strokeWidth: 2 }
        : style;

    return (
        <>
            <BaseEdge path={edgePath} markerEnd={markerEnd} style={edgeStyle} />
            <path
                d={edgePath}
                fill="none"
                strokeWidth={20}
                stroke="transparent"
                onMouseEnter={() => setIsOpen(true)}
                onMouseLeave={() => setIsOpen(false)}
                onMouseMove={handleMouseMove}
                style={{ cursor: "pointer" }}
            />

            {/* Portal tooltip at mouse position */}
            {isOpen &&
                createPortal(
                    <TooltipProvider>
                        <Tooltip open={true}>
                            <TooltipTrigger asChild>
                                <div
                                    style={{
                                        position: "fixed",
                                        left: mousePos.x,
                                        top: mousePos.y,
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
                                <>
                                    {data?.isWeakEdge
                                        ? "Low Match Rate: Indicates low data overlap between connected tables"
                                        : "Healthy Match Rate: Indicates healthy data overlap between connected tables."}
                                </>
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>,
                    document.body
                )}
        </>
    );
};
