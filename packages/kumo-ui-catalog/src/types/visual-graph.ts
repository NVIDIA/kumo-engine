// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Edge, Node } from "@xyflow/react";

import {
    type PredictedNodeType,
    type PredictionItem
} from "@/utils/xai-subgraph-shapes";

export type TableGraphNodeData = {
    id: string;
    label: string;
    cols: Array<{ name: string }>;
    position: { x: number; y: number };
    pkey: string;
    timeCol: string;
    earliestFact: string;
    latestFact: string;
    foreignKeys: Array<string>;
    setSelectedTable: (t: string) => void;
    pkeySelected?: boolean;
    highLightKeys?: Array<string>;
};

export type TableGraphNodeProps = {
    data: TableGraphNodeData;
    enableEditTable: boolean;
    onEditTableClick: (id: string) => void;
};

export type GraphEdge = {
    id: string;
    source: string;
    target: string;
    type?: string;
};

export type CustomNodeType = {
    id: string;
    label: string;
    position: { x: number; y: number };
    color?: string;
    size?: number;
    tableName?: string;
    keyColumn?: string;
};

export type CustomEdgeType = {
    color?: string;
    target: string;
    source: string;
    id: string;
};

export type NodeType = Node<CustomNodeType>;
export type EdgeType = Edge<CustomEdgeType>;

export type GraphNode = {
    id: string;
    nodeId: string;
    position?: { x: number; y: number };
    data: object;
};

// ----- XaiSubgraph -----

export interface SubgraphNodeInfo {
    tableName: string;
    hopLevel: number;
    totalScore: number;
}

export type NodeStatus = "active" | "table-excluded" | "path-only" | "hidden";

export type SubgraphNodeData = {
    graphData: SubgraphNodeInfo;
    color: string;
    halfSize: number;
    status: NodeStatus;
    isHovered: boolean;
    isSelected: boolean;
    isDimmedByHover: boolean;
    /** Drives shape variant. Undefined → circle. */
    nodeType?: PredictedNodeType;
};

export type ChipNodeData = {
    count: number;
};

export type PredictionsPanelBackgroundData = {
    title: string;
    width: number;
    height: number;
};

export type PredictionItemNodeData = {
    item: PredictionItem;
    /** Cross-highlight key (= matching LHS node id when present); used for hoveredNodeId. */
    hoverKey: string;
    isHovered: boolean;
    isSelected: boolean;
    isDimmed: boolean;
};

/** TLP Predictions panel inputs. Omit for non-TLP tasks. */
export type XaiSubgraphPredictions = {
    items: PredictionItem[];
    /** Source node for the dashed entity→panel edge. */
    entityNodeId: string;
    /** Cross-highlight hover key (= prediction-item node's `data.hoverKey`). */
    hoveredId: string | null;
    /** Selection id (same key as hoveredId). */
    selectedId: string | null;
};

export type XaiSubgraphProps = {
    displayNodes: Node[];
    displayEdges: Edge[];
    isLegendOpen: boolean;
    legendItems: { color: string; label: string; isEntity: boolean }[];
    onNodeMouseEnter: (e: React.MouseEvent, node: Node) => void;
    onNodeMouseLeave: () => void;
    onNodeClick: (e: React.MouseEvent, node: Node) => void;
    onPaneClick: () => void;
    onLegendOpenChange: (v: boolean) => void;
    onOpenFullscreen: () => void;
    onCloseFullscreen?: () => void;
    /** Node-size tier labels, bottom→top. Radii derived (3.5px + 0.5px per tier). */
    tierLabels?: string[];
    /** TLP tasks only. */
    predictions?: XaiSubgraphPredictions;
};
