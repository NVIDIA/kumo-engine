// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
    type Edge,
    type Node,
    type NodeStatus,
    type SubgraphNodeData
} from "@kumo-ai/kumo-ui-catalog";

import type { GraphNode, GraphNodeData, GraphXaiScores } from "types/explorer";

import { updateColumnsWithXaiData } from "utils/Converters/explorerConverter";
import { getKeys } from "utils/Helpers/objectHelpers";
import { getPathToRoot } from "utils/subgraphTreeViewUtils";

import { resolvePredictionItemClick } from "./subgraphPredictionItem";

interface UseSubgraphInteractionParams {
    visibleNodes: GraphNode[];
    nodeStatuses: Record<string, NodeStatus>;
    layoutedNodes: Node[];
    collapseExpandChipNodes: Node[];
    structuralEdges: Edge[];
    parentMap: Record<string, string>;
    expandedChipParents: Set<string>;
    setExpandedChipParents: React.Dispatch<React.SetStateAction<Set<string>>>;
    isFullExpand: boolean;
    setIsFullExpand: React.Dispatch<React.SetStateAction<boolean>>;
    isUserCollapsingRef: React.MutableRefObject<boolean>;
    isAnyFilterActive: boolean;
    totalNodes: number;
    setActiveSubGraph?: (v: GraphNodeData | null) => void;
    xaiData?: GraphXaiScores;
    anchorTime: string;
}

export function useSubgraphInteraction({
    visibleNodes,
    nodeStatuses,
    layoutedNodes,
    collapseExpandChipNodes,
    structuralEdges,
    parentMap,
    expandedChipParents,
    setExpandedChipParents,
    isFullExpand,
    setIsFullExpand,
    isUserCollapsingRef,
    isAnyFilterActive,
    totalNodes,
    setActiveSubGraph,
    xaiData,
    anchorTime
}: UseSubgraphInteractionParams) {
    const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
    const [selectedNodeData, setSelectedNodeData] =
        useState<GraphNodeData | null>(null);
    const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
    const [isLegendOpen, setIsLegendOpen] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);

    const selectedNodeDataRef = useRef(selectedNodeData);
    selectedNodeDataRef.current = selectedNodeData;
    const leaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        if (!xaiData || !getKeys(xaiData).length) return;
        const data = selectedNodeDataRef.current;
        if (!data?.cols?.length || data.cols[0]?.isXaiUpdated) return;
        const newCols = updateColumnsWithXaiData(
            data.cols,
            data.tableName,
            xaiData,
            anchorTime
        );
        setSelectedNodeData((prev) =>
            prev ? { ...prev, cols: newCols } : prev
        );
    }, [xaiData, anchorTime]);

    // RHS prediction-item ids aren't in nodeStatuses; the `in` check skips them.
    useEffect(() => {
        if (
            selectedNodeId &&
            selectedNodeId in nodeStatuses &&
            nodeStatuses[selectedNodeId] !== "active"
        ) {
            setSelectedNodeId(null);
            setSelectedNodeData(null);
            setActiveSubGraph?.(null);
        }
    }, [nodeStatuses, selectedNodeId, setActiveSubGraph]);

    useEffect(() => {
        if (
            !isFullExpand &&
            !isAnyFilterActive &&
            !isUserCollapsingRef.current &&
            expandedChipParents.size > 0 &&
            visibleNodes.length === totalNodes
        ) {
            setIsFullExpand(true);
        }
    }, [
        isFullExpand,
        isAnyFilterActive,
        isUserCollapsingRef,
        expandedChipParents,
        visibleNodes.length,
        totalNodes,
        setIsFullExpand
    ]);

    useEffect(() => {
        return () => {
            if (leaveTimer.current) clearTimeout(leaveTimer.current);
        };
    }, []);

    const hoveredPath = useMemo(
        () =>
            hoveredNodeId
                ? getPathToRoot(hoveredNodeId, parentMap)
                : new Set<string>(),
        [hoveredNodeId, parentMap]
    );

    const displayNodes = useMemo<Node[]>(() => {
        const nodes: Node[] = layoutedNodes.map((n) => ({
            ...n,
            data: {
                ...(n.data as SubgraphNodeData),
                isHovered: n.id === hoveredNodeId,
                isSelected: n.id === selectedNodeId,
                isDimmedByHover:
                    hoveredNodeId !== null &&
                    !hoveredPath.has(n.id) &&
                    n.type !== "collapseChip"
            }
        }));
        return [...nodes, ...collapseExpandChipNodes];
    }, [
        layoutedNodes,
        hoveredNodeId,
        selectedNodeId,
        hoveredPath,
        collapseExpandChipNodes
    ]);

    const displayEdges = useMemo<Edge[]>(
        () =>
            structuralEdges.map((e) => ({
                ...e,
                style:
                    hoveredNodeId !== null
                        ? hoveredPath.has(e.source) && hoveredPath.has(e.target)
                            ? {
                                  stroke: "var(--color-pink-500)",
                                  strokeWidth: 1,
                                  opacity: 1
                              }
                            : {
                                  stroke: "var(--color-slate-200)",
                                  strokeWidth: 1,
                                  opacity: 0.25
                              }
                        : {
                              stroke: "var(--color-slate-300)",
                              strokeWidth: 1,
                              opacity: 1
                          }
            })),
        [structuralEdges, hoveredNodeId, hoveredPath]
    );

    const onNodeMouseEnter = useCallback((_: React.MouseEvent, node: Node) => {
        // Non-interactive nodes leave the pending leave timer running.
        if (
            node.type === "collapseChip" ||
            node.type === "collapseExpandChip" ||
            node.type === "predictionsPanelBackground"
        ) {
            return;
        }
        if (leaveTimer.current) {
            clearTimeout(leaveTimer.current);
            leaveTimer.current = null;
        }
        if (node.type === "predictionItem") {
            const hk = (node.data as { hoverKey?: string }).hoverKey;
            if (hk) setHoveredNodeId(hk);
            return;
        }
        const d = node.data as SubgraphNodeData;
        if (d.status !== "active") return;
        setHoveredNodeId(node.id);
    }, []);

    const onNodeMouseLeave = useCallback(() => {
        leaveTimer.current = setTimeout(() => {
            setHoveredNodeId(null);
        }, 80);
    }, []);

    const toggleSelectedNode = useCallback(
        (id: string, data: GraphNodeData | null) => {
            setSelectedNodeId((prev) => {
                if (prev === id) {
                    setSelectedNodeData(null);
                    setActiveSubGraph?.(null);
                    return null;
                }
                setSelectedNodeData(data);
                setActiveSubGraph?.(data);
                return id;
            });
        },
        [setActiveSubGraph]
    );

    const onNodeClick = useCallback(
        (_: React.MouseEvent, node: Node) => {
            if (node.type === "collapseChip") {
                isUserCollapsingRef.current = false;
                const parentId = node.id.slice("chip_".length);
                setExpandedChipParents((prev) => new Set([...prev, parentId]));
                return;
            }
            if (node.type === "collapseExpandChip") {
                isUserCollapsingRef.current = true;
                const parentId = node.id.slice("collapsechip_".length);
                setIsFullExpand(false);
                setExpandedChipParents((prev) => {
                    const next = new Set(prev);
                    next.delete(parentId);
                    return next;
                });
                return;
            }
            if (node.type === "predictionsPanelBackground") return;
            if (node.type === "predictionItem") {
                const result = resolvePredictionItemClick(node, visibleNodes);
                if (!result) return;
                toggleSelectedNode(result.hk, result.data);
                return;
            }
            const d = node.data as SubgraphNodeData;
            if (d.status !== "active") return;
            const fullNodeData =
                visibleNodes.find((n) => n.id === node.id)?.data ?? null;
            toggleSelectedNode(node.id, fullNodeData);
        },
        [
            visibleNodes,
            setExpandedChipParents,
            setIsFullExpand,
            isUserCollapsingRef,
            toggleSelectedNode
        ]
    );

    const onPaneClick = useCallback(() => {
        setSelectedNodeId(null);
        setSelectedNodeData(null);
        setActiveSubGraph?.(null);
    }, [setActiveSubGraph]);

    return {
        selectedNodeId,
        selectedNodeData,
        hoveredNodeId,
        isLegendOpen,
        setIsLegendOpen,
        isFullscreen,
        setIsFullscreen,
        displayNodes,
        displayEdges,
        onNodeMouseEnter,
        onNodeMouseLeave,
        onNodeClick,
        onPaneClick
    };
}
