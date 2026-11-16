// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

import {
    type ChipNodeData,
    type Edge,
    type Node,
    type NodeStatus,
    type PredictedNodeType,
    type SubgraphNodeData,
    xaiSubgraphApplyLayout
} from "@kumo-ai/kumo-ui-catalog";

import type { GraphNode } from "types/explorer";

import {
    DEFAULT_MAX_HOP,
    TIER_HALF_SIZES,
    TIER_LABELS
} from "utils/subgraphTreeViewUtils";

interface UseSubgraphGraphParams {
    nodeArr: GraphNode[];
    parentMap: Record<string, string>;
    childrenMap: Record<string, string[]>;
    tableColorMap: Record<string, string>;
    scoreTierMap: Record<string, number>;
    expandedChipParents: Set<string>;
    isAnyFilterActive: boolean;
    isFullExpand: boolean;
    isTableFilterActive: boolean;
    isSizeFilterActive: boolean;
    selectedTables: string[];
    selectedSizes: string[];
    nodeTypeMap?: Map<string, PredictedNodeType>;
}

type VisibleResult = {
    visibleNodes: GraphNode[];
    nodeStatuses: Record<string, NodeStatus>;
    tableExcludedByTable: Record<string, number>;
    pathOnlyCount: number;
};

/** Visible set when no filter is active: hop-limited nodes + expanded chip subtrees. */
function computeUnfilteredVisible(
    nodeArr: GraphNode[],
    isFullExpand: boolean,
    expandedChipParents: Set<string>,
    childrenMap: Record<string, string[]>
): VisibleResult {
    const visibleIds = new Set<string>();
    for (const n of nodeArr) {
        const hop = n.data.hopLevel || 0;
        if (isFullExpand || hop <= DEFAULT_MAX_HOP) visibleIds.add(n.id);
    }
    const toExpand = [...expandedChipParents].filter((id) =>
        visibleIds.has(id)
    );
    while (toExpand.length > 0) {
        const pid = toExpand.shift()!;
        for (const cid of childrenMap[pid] ?? []) {
            if (!visibleIds.has(cid)) {
                visibleIds.add(cid);
                if (expandedChipParents.has(cid)) toExpand.push(cid);
            }
        }
    }
    const visible = nodeArr.filter((n) => visibleIds.has(n.id));
    const statuses: Record<string, NodeStatus> = {};
    visible.forEach((n) => {
        statuses[n.id] = "active";
    });
    return {
        visibleNodes: visible,
        nodeStatuses: statuses,
        tableExcludedByTable: {},
        pathOnlyCount: 0
    };
}

/** Visible set when a filter is active: matched nodes, table-excluded, and path-only ancestors. */
function computeFilteredVisible(
    nodeArr: GraphNode[],
    parentMap: Record<string, string>,
    isTableFilterActive: boolean,
    selectedTables: string[],
    scoreTierMap: Record<string, number>,
    isSizeFilterActive: boolean,
    selectedSizes: string[]
): VisibleResult {
    const statuses: Record<string, NodeStatus> = {};
    const tableExcluded: Record<string, number> = {};
    for (const node of nodeArr) {
        const matchesTable =
            !isTableFilterActive ||
            selectedTables.includes(node.data.tableName);
        const tierLabel = TIER_LABELS[scoreTierMap[node.id] ?? 0];
        const matchesSize =
            !isSizeFilterActive || selectedSizes.includes(tierLabel);
        if (matchesTable && matchesSize) {
            statuses[node.id] = "active";
        } else if (!matchesTable && matchesSize && isSizeFilterActive) {
            statuses[node.id] = "table-excluded";
            const tbl = node.data.tableName;
            tableExcluded[tbl] = (tableExcluded[tbl] ?? 0) + 1;
        }
    }

    const alreadyVisible = new Set(Object.entries(statuses).map(([id]) => id));
    const pathOnlyIds = new Set<string>();
    for (const [id, status] of Object.entries(statuses)) {
        if (status !== "active" && status !== "table-excluded") continue;
        let cur = parentMap[id];
        while (cur) {
            if (!alreadyVisible.has(cur) && !pathOnlyIds.has(cur)) {
                pathOnlyIds.add(cur);
            }
            cur = parentMap[cur];
        }
    }
    pathOnlyIds.forEach((id) => {
        statuses[id] = "path-only";
    });
    return {
        visibleNodes: nodeArr.filter((n) => statuses[n.id] !== undefined),
        nodeStatuses: statuses,
        tableExcludedByTable: tableExcluded,
        pathOnlyCount: pathOnlyIds.size
    };
}

export function useSubgraphGraph({
    nodeArr,
    parentMap,
    childrenMap,
    tableColorMap,
    scoreTierMap,
    expandedChipParents,
    isAnyFilterActive,
    isFullExpand,
    isTableFilterActive,
    isSizeFilterActive,
    selectedTables,
    selectedSizes,
    nodeTypeMap
}: UseSubgraphGraphParams) {
    const { visibleNodes, nodeStatuses, tableExcludedByTable, pathOnlyCount } =
        useMemo(
            () =>
                isAnyFilterActive
                    ? computeFilteredVisible(
                          nodeArr,
                          parentMap,
                          isTableFilterActive,
                          selectedTables,
                          scoreTierMap,
                          isSizeFilterActive,
                          selectedSizes
                      )
                    : computeUnfilteredVisible(
                          nodeArr,
                          isFullExpand,
                          expandedChipParents,
                          childrenMap
                      ),
            [
                nodeArr,
                parentMap,
                childrenMap,
                expandedChipParents,
                isAnyFilterActive,
                isFullExpand,
                isTableFilterActive,
                isSizeFilterActive,
                selectedTables,
                selectedSizes,
                scoreTierMap
            ]
        );

    const collapseChips = useMemo(() => {
        if (isAnyFilterActive || isFullExpand) return [];
        const chips: { id: string; parentId: string; count: number }[] = [];
        for (const node of visibleNodes) {
            const directHidden = (childrenMap[node.id] ?? []).filter(
                (cId) => !nodeStatuses[cId]
            );
            if (directHidden.length > 0) {
                chips.push({
                    id: `chip_${node.id}`,
                    parentId: node.id,
                    count: directHidden.length
                });
            }
        }
        return chips;
    }, [
        visibleNodes,
        childrenMap,
        nodeStatuses,
        isAnyFilterActive,
        isFullExpand
    ]);

    const structuralNodes = useMemo<Node[]>(() => {
        const nodes: Node[] = visibleNodes.map((n) => {
            const nodeType = nodeTypeMap?.get(n.id);
            const halfSize = nodeType
                ? TIER_HALF_SIZES[TIER_HALF_SIZES.length - 1]
                : TIER_HALF_SIZES[scoreTierMap[n.id] || 0];
            return {
                id: n.id,
                type: "subgraphNode",
                position: { x: 0, y: 0 },
                zIndex: 1,
                data: {
                    graphData: {
                        tableName: n.data.tableName,
                        hopLevel: n.data.hopLevel ?? 0,
                        totalScore: n.data.totalScore
                    },
                    color:
                        tableColorMap[n.data.tableName] ??
                        "var(--color-kumo-grey-400)",
                    halfSize,
                    status: nodeStatuses[n.id] ?? "active",
                    isHovered: false,
                    isSelected: false,
                    isDimmedByHover: false,
                    nodeType
                } satisfies SubgraphNodeData,
                draggable: false,
                selectable: nodeStatuses[n.id] === "active"
            };
        });
        collapseChips.forEach((chip) => {
            nodes.push({
                id: chip.id,
                type: "collapseChip",
                position: { x: 0, y: 0 },
                data: { count: chip.count } satisfies ChipNodeData,
                draggable: false,
                selectable: false
            });
        });
        return nodes;
    }, [
        visibleNodes,
        collapseChips,
        tableColorMap,
        scoreTierMap,
        nodeStatuses,
        nodeTypeMap
    ]);

    const structuralEdges = useMemo<Edge[]>(() => {
        const visibleIds = new Set(visibleNodes.map((n) => n.id));
        const edges: Edge[] = [];
        let idx = 0;
        for (const node of visibleNodes) {
            const pid = parentMap[node.id];
            if (pid && visibleIds.has(pid)) {
                edges.push({
                    id: `e_${idx++}`,
                    source: pid,
                    target: node.id,
                    style: { stroke: "var(--color-slate-300)", strokeWidth: 1 }
                });
            }
        }
        collapseChips.forEach((chip) => {
            edges.push({
                id: `e_chip_${chip.parentId}`,
                source: chip.parentId,
                target: chip.id,
                style: { stroke: "var(--color-slate-300)", strokeWidth: 1 }
            });
        });
        return edges;
    }, [visibleNodes, parentMap, collapseChips]);

    const layoutedNodes = useMemo(
        () => xaiSubgraphApplyLayout(structuralNodes, structuralEdges),
        [structuralNodes, structuralEdges]
    );

    const collapseExpandChipNodes = useMemo<Node[]>(() => {
        if (isAnyFilterActive) return [];
        return layoutedNodes.flatMap((n) => {
            if (n.type !== "subgraphNode" || !expandedChipParents.has(n.id))
                return [];
            const halfSize = (n.data as SubgraphNodeData).halfSize;
            return [
                {
                    id: `collapsechip_${n.id}`,
                    type: "collapseExpandChip",
                    position: {
                        x: n.position.x + halfSize + 2,
                        y: n.position.y - 16
                    },
                    zIndex: 2,
                    data: {},
                    draggable: false,
                    selectable: false
                } satisfies Node
            ];
        });
    }, [layoutedNodes, expandedChipParents, isAnyFilterActive]);

    const legendItems = useMemo(
        () =>
            Object.entries(tableColorMap).map(([label, color], index) => ({
                label,
                color,
                isEntity: index === 0
            })),
        [tableColorMap]
    );

    return {
        visibleNodes,
        nodeStatuses,
        tableExcludedByTable,
        pathOnlyCount,
        visibleCount: visibleNodes.length,
        layoutedNodes,
        collapseExpandChipNodes,
        structuralEdges,
        legendItems
    };
}
