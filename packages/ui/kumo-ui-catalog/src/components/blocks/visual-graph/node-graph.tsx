// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState
} from "react";

import SmartBezierEdge from "@tisoap/react-flow-smart-edge";
import {
    addEdge,
    Connection,
    Controls,
    Handle,
    Node,
    NodeMouseHandler,
    NodeOrigin,
    NodeProps,
    OnConnect,
    Position,
    ReactFlow,
    ReactFlowProvider,
    useEdgesState,
    useNodesState,
    useReactFlow
} from "@xyflow/react";

import Spinner from "@/animations/spinner";
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@/components/ui/tooltip";
import useForceLayout from "@/hooks/useForceLayout";
import { Key } from "@/icons";
import { EdgeType, GraphEdge, GraphNode, NodeType } from "@/types/visual-graph";

import "@xyflow/react/dist/style.css";

type TableColorMap = Map<string, string>;
const nodeOrigin: NodeOrigin = [0.5, 0.5];

const subGraphColors = [
    "#7B2869", // Deep Purple
    "#004d00", // Dark green
    "#9A3412",
    "#075985",
    "#5B21B6",
    "#8D2A63",
    "#7B2869", // Deep Purple
    "#C70039", // Crimson Red
    "#4C3A51", // Dark Purple
    "#CD5888", // Rose Pink
    "#088395", // Teal
    "#FF8B13", // Orange
    "#116D6E" // Dark Teal
];

function scaleValue(
    minInput: number,
    maxInput: number,
    value: number,
    defaultOutput: number = 15
) {
    const minOutput: number = defaultOutput;
    const maxOutput: number = defaultOutput * 2;

    // Check if the value is within the defined range
    if (value <= minInput || value > maxInput) {
        return defaultOutput;
    }

    // Calculate the scaled value
    const scaledValue: number =
        minOutput +
        ((value - minInput) / (maxInput - minInput)) * (maxOutput - minOutput);

    return Math.ceil(scaledValue);
}

const getNodesFromData = (
    data: Array<GraphNode>,
    min: number,
    max: number
): Array<NodeType> => {
    const nodes: Array<NodeType> = [];
    const defaultNodeSize: number = data?.length > 100 ? 40 : 15;
    data?.forEach(({ id, data }, idx) => {
        const r: number = scaleValue(
            min,
            max,
            (data as { totalScore: number })?.totalScore,
            defaultNodeSize
        );
        const node: NodeType = {
            id: id,
            position: {
                x: Math.random() * idx,
                y: Math.random() * idx
            },
            data: {
                id: id,
                label: id,
                position: {
                    x: Math.random() * idx,
                    y: Math.random() * idx
                },
                size: r,
                tableName: (data as { tableName: string })?.tableName,
                keyColumn: (data as { idName: string })?.idName
            },
            type: "custom"
        };
        nodes?.push(node);
    });

    return nodes as never[];
};

const getEdgesFromGraphData = (data: Array<GraphEdge>): Array<EdgeType> => {
    return data?.map((e) => {
        return {
            id: e?.id,
            source: e?.source,
            target: e?.target,
            type: "straight"
        };
    });
};

interface NodeGraphDefProps<T extends object> {
    nodesArray: GraphNode[];
    linksArray: GraphEdge[];
    onSelect: (node: T | null) => void;
    show: boolean;
}

export const NodeGraph = <T extends object>({
    nodesArray,
    linksArray,
    onSelect,
    show
}: NodeGraphDefProps<T>) => {
    return (
        <ReactFlowProvider>
            <NodeGraphInt
                nodesArray={nodesArray}
                linksArray={linksArray}
                onSelect={onSelect}
                show={show}
            />
        </ReactFlowProvider>
    );
};

const NodeGraphInt = <T extends object>({
    nodesArray,
    linksArray,
    onSelect,
    show
}: NodeGraphDefProps<T>) => {
    const { strength, strokeWidth } = useMemo(() => {
        let strength = -1000;
        let strokeWidth = 1;
        if (nodesArray?.length > 100) {
            strength = -500;
            strokeWidth = 3;
        }

        return { strength, strokeWidth };
    }, [nodesArray]);
    const distance = 100;

    const { fitView } = useReactFlow();
    const [hiddenCategories, setHiddenCategories] = useState<Set<string>>(
        new Set()
    );
    const { min, max } = useMemo(() => {
        const totalScores = nodesArray?.map(
            (node) => (node?.data as { totalScore: number })?.totalScore ?? 0
        );
        const max: number = Math.max(...totalScores);
        const min: number = Math.min(...totalScores);
        return { min, max };
    }, [nodesArray]);

    const [selectedNode, setSelectedNode] = useState<string | null>(null);

    const initialNodes = useMemo(() => {
        return getNodesFromData(nodesArray, min, max);
    }, [max, min, nodesArray]);
    const initialEdges = useMemo(
        () => getEdgesFromGraphData(linksArray),
        [linksArray]
    );

    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    const [isLayoutStable, setIsLayoutStable] = useState(false);

    const [nodeGraphFadeClass, setNodeGraphFadeClass] = useState("");

    const hasFitViewRun = useRef(false);
    const [hoveredNode, setHoveredNode] = useState<string | null>(null);
    const tableColorMap = useMemo(() => {
        const categories = new Map<string, string>();
        let colorIndex = 0;
        nodesArray?.forEach((node) => {
            const label =
                (node?.data as { tableName: string })?.tableName ?? "";
            if (!categories?.has(label)) {
                categories?.set(
                    label,
                    subGraphColors[colorIndex % subGraphColors.length]
                );
                colorIndex++;
            }
        });
        return categories;
    }, [nodesArray]);

    useForceLayout({ strength, distance, setIsLayoutStable });

    useEffect(() => {
        setNodes(initialNodes as never[]);
        setEdges(initialEdges as never[]);
    }, [initialNodes, initialEdges, setNodes, setEdges]);

    useEffect(() => {
        if (!hasFitViewRun?.current && show && isLayoutStable) {
            setNodeGraphFadeClass("opacity-100");
            fitView({ padding: 0, duration: 0 });
            hasFitViewRun.current = true;
        }
    }, [fitView, nodes, edges, show, isLayoutStable]);

    const edgeTypes = useMemo(() => ({ smart: SmartBezierEdge }), []);
    useEffect(() => {
        if (
            nodesArray?.length > 0 &&
            nodesArray[0]?.id &&
            show &&
            isLayoutStable
        ) {
            setSelectedNode(null);
            onSelect(null);
        }
    }, [nodesArray, show, isLayoutStable, onSelect]);

    useEffect(() => {
        setEdges(((prevEdges: EdgeType[]) => {
            const updatedEdges = prevEdges?.map((edge: EdgeType) => ({
                ...edge,
                style: {
                    opacity:
                        selectedNode || hoveredNode
                            ? edge?.source === hoveredNode ||
                              edge?.target === hoveredNode
                                ? 1
                                : 0.3
                            : 1,
                    transition: "opacity 0.3s ease-in-out",
                    strokeWidth: strokeWidth
                }
            }));

            // Update nodes
            setNodes(((prevNodes: NodeType[]) => {
                return prevNodes?.map((node: NodeType) => {
                    const isHoveredOrSelected =
                        node?.id === hoveredNode || node?.id === selectedNode;

                    const isConnected = updatedEdges?.some(
                        (edge) =>
                            (edge?.source === hoveredNode &&
                                edge?.target === node?.id) ||
                            (edge?.target === hoveredNode &&
                                edge?.source === node?.id)
                    );

                    return {
                        ...node,
                        selected: node?.id === selectedNode,
                        hovered: isHoveredOrSelected || isConnected,
                        style: {
                            opacity:
                                selectedNode || hoveredNode
                                    ? isHoveredOrSelected || isConnected
                                        ? 1
                                        : 0.3
                                    : 1,
                            transition: "opacity 0.3s ease-in-out"
                        }
                    };
                });
            }) as unknown as never[]);

            return updatedEdges;
        }) as unknown as never[]);
    }, [selectedNode, strokeWidth, hoveredNode, setNodes, setEdges]);

    const nodeTypes = useMemo(
        () => ({
            custom: (props: NodeProps) => (
                <CustomNode
                    {...props}
                    setHoveredNode={setHoveredNode}
                    tableColorMap={tableColorMap}
                />
            )
        }),
        [tableColorMap, setHoveredNode]
    );

    const onConnect: OnConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge(params, eds)),
        [setEdges]
    );

    const onSelectNode: NodeMouseHandler = useCallback(
        (_event: React.MouseEvent, node: Node) => {
            const id = node?.id;
            if (id === selectedNode) {
                setSelectedNode(null);
                onSelect(null); // Reset the selected node info
                return;
            }
            setSelectedNode(id);
            const data: GraphNode = nodesArray?.find(
                (node) => node?.id === id
            ) as GraphNode;
            if (data) {
                onSelect(data.data as T);
            }
        },
        [nodesArray, selectedNode, onSelect]
    );

    const legendItems = useMemo(() => {
        const categories = new Map<string, string>();
        let colorIndex = 0;
        nodesArray?.forEach((node) => {
            const label =
                (node?.data as { tableName: string })?.tableName ?? "";
            if (!categories?.has(label)) {
                categories?.set(
                    label,
                    subGraphColors[colorIndex % subGraphColors.length]
                );
                colorIndex++;
            }
        });

        return Array?.from(categories, ([label, color]) => ({ label, color }));
    }, [nodesArray]);

    const toggleCategory = (label: string) => {
        setHiddenCategories((prev) => {
            const newSet = new Set(prev);
            if (newSet?.has(label)) {
                newSet?.delete(label);
            } else {
                newSet?.add(label);
            }
            return new Set(newSet);
        });
    };

    const filteredNodes = useMemo(() => {
        return nodesArray?.filter((node) => {
            const category =
                (node?.data as { tableName: string })?.tableName ?? ""; // Adjust as needed
            return !hiddenCategories?.has(category);
        });
    }, [nodesArray, hiddenCategories]);

    const filteredEdges = useMemo(() => {
        return linksArray?.filter(
            (edge) =>
                filteredNodes?.some((node) => node?.id === edge?.source) &&
                filteredNodes?.some((node) => node?.id === edge?.target)
        );
    }, [linksArray, filteredNodes]);

    useEffect(() => {
        setNodes(getNodesFromData(filteredNodes, min, max) as never[]);
        setEdges(getEdgesFromGraphData(filteredEdges) as never[]);
    }, [filteredNodes, filteredEdges, min, max, setNodes, setEdges]);

    const GraphLegend: React.FC<{
        items: { label: string; color: string }[];
        onToggle: (label: string) => void;
        hiddenCategories: Set<string>;
    }> = ({ items, onToggle, hiddenCategories }) => (
        <div className="absolute z-10 mb-1 flex flex-col flex-wrap px-1 py-1">
            {items?.map(({ label, color }) => {
                const isHidden = hiddenCategories?.has(label);
                return (
                    <div
                        key={label}
                        className="flex cursor-pointer items-center gap-2 rounded-md px-1 py-0.5 transition-colors hover:bg-slate-100"
                        onClick={() => onToggle(label)}>
                        <span
                            className="h-4 w-4 rounded-sm transition-all"
                            style={{
                                backgroundColor: isHidden ? "#F0F0F0" : color,
                                border: isHidden
                                    ? "1px solid #6E84A3"
                                    : `1px solid ${color}`
                            }}></span>
                        <span
                            className={`text-sm font-medium transition-all ${
                                isHidden
                                    ? "text-slate-500 line-through"
                                    : "text-slate-800"
                            }`}>
                            {label}
                        </span>
                    </div>
                );
            })}
        </div>
    );

    return (
        <div
            className={`relative h-[50vh] w-full ${show ? "block" : "hidden"}`}>
            <GraphLegend
                items={legendItems}
                onToggle={toggleCategory}
                hiddenCategories={hiddenCategories}
            />
            {!isLayoutStable && (
                <div className="flex h-full w-full items-center justify-center">
                    <Spinner size="md" />
                </div>
            )}
            <Controls showInteractive={false} className="!left-0 !mb-1 !ml-1" />
            <ReactFlow
                nodes={nodes}
                edges={edges}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                proOptions={{ hideAttribution: true }}
                onConnect={onConnect}
                nodeOrigin={nodeOrigin}
                onNodeClick={onSelectNode}
                fitView
                minZoom={0.1}
                maxZoom={1.2}
                className={`origin-center border-none opacity-0 transition-all duration-500 ease-in-out ${nodeGraphFadeClass}`}
            />
        </div>
    );
};

function CustomNode({
    data,
    setHoveredNode,
    tableColorMap
}: NodeProps & {
    setHoveredNode: (id: string | null) => void;
    tableColorMap: TableColorMap;
}) {
    const size = Number.isNaN(data?.size) ? 15 : (data?.size ?? 15);

    const getColorForTable = useMemo(() => {
        if (!data?.tableName) return subGraphColors[0];
        return (
            tableColorMap.get((data as { tableName: string })?.tableName) ??
            subGraphColors[0]
        );
    }, [data, tableColorMap]);

    const borderColor = getColorForTable;
    const backgroundColor = getColorForTable;

    const handleMouseEnter = () => {
        setHoveredNode((data as { id: string })?.id);
    };

    const handleMouseLeave = () => {
        setHoveredNode(null);
    };

    return (
        <div onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
            <Tooltip>
                <TooltipTrigger>
                    <div
                        key={(data as { id: string })?.id}
                        id={`nodetip${data?.id}`}
                        style={{
                            borderRadius: `${size}px`,
                            width: `${size}px`,
                            height: `${size}px`,
                            border: `1px solid ${borderColor}`,
                            backgroundColor: backgroundColor
                        }}
                        className="relative flex cursor-pointer items-center justify-center shadow-md transition-colors duration-200 ease-in-out hover:shadow-lg">
                        <Handle
                            type="source"
                            position={Position.Left}
                            id={`source_${data?.id}`}
                            className="!pointer-events-none !absolute !left-1/2 !top-1/2 !-translate-x-1/2 !-translate-y-1/2 !border-none !bg-transparent !opacity-0"
                        />

                        <Handle
                            type="target"
                            position={Position.Right}
                            id={`target_${data?.id}`}
                            className="!pointer-events-none !absolute !left-1/2 !top-1/2 !-translate-x-1/2 !-translate-y-1/2 !border-none !bg-transparent !opacity-0"
                        />
                    </div>
                </TooltipTrigger>
                <TooltipContent arrow={false} side={"right"}>
                    <div className="flex flex-col">
                        <p className="!mb-0 !pb-0 text-sm font-semibold leading-5">
                            {(data as { tableName: string })?.tableName &&
                                (data as { tableName: string })?.tableName}
                        </p>
                        {(data as { keyColumn: string })?.keyColumn && (
                            <p className="!m-0 flex flex-col items-start !p-0 text-xs font-normal leading-4">
                                <label className="flex items-center gap-1">
                                    <Key className="size-4 fill-white" />
                                    <span className="ml-1">
                                        {
                                            (data as { keyColumn: string })
                                                ?.keyColumn
                                        }
                                    </span>
                                </label>
                            </p>
                        )}
                    </div>
                </TooltipContent>
            </Tooltip>
        </div>
    );
}
