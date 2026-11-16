// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

import type {
    HistoricalItems,
    KumoapiExplainSubgraphSubgraph
} from "gen/v0/types";

import type {
    GraphNodeData,
    GraphXaiScores,
    LinkPredictionEntityOutput
} from "types/explorer";
import { TaskType } from "types/predictiveQueries";

import { isLinkPredictionType } from "utils/Helpers/typeGuards";
import { TIER_HALF_SIZES } from "utils/subgraphTreeViewUtils";

import {
    buildNodeTypeMap,
    buildPkeyToLhsNodeInfo
} from "./subgraphNodeTypeBuild";
import { usePredictionsConfig } from "./usePredictionsConfig";
import { useSubgraphData } from "./useSubgraphData";
import { useSubgraphFilters } from "./useSubgraphFilters";
import { useSubgraphGraph } from "./useSubgraphGraph";
import { useSubgraphInteraction } from "./useSubgraphInteraction";

export type SubgraphTreeViewProps = {
    subgraph: KumoapiExplainSubgraphSubgraph | undefined;
    entityId: string;
    xaiData?: GraphXaiScores;
    anchorTime: string;
    setActiveSubGraph?: (v: GraphNodeData | null) => void;
    /** TLP-only: prediction/target data drives the floating Predictions panel. */
    taskType?: TaskType;
    entityOutput?: LinkPredictionEntityOutput | null;
    historicalItems?: HistoricalItems | null;
    /** Predicted-item table name (used for the panel item tooltip + color). */
    targetTableName?: string;
    isLoading?: boolean;
};

export function useSubgraphTreeView({
    subgraph,
    xaiData,
    anchorTime,
    setActiveSubGraph,
    taskType,
    entityOutput,
    historicalItems,
    targetTableName
}: SubgraphTreeViewProps) {
    // Matches TLP, SLP, and the deprecated bare LINK_PREDICTION.
    const isLinkPredTask = isLinkPredictionType(taskType || "");

    const {
        nodeArr,
        parentMap,
        childrenMap,
        tableColorMap,
        scoreTierMap,
        allTableNames,
        totalNodes,
        allChipBoundaryParents,
        rootId,
        predictedItemTableName
    } = useSubgraphData(subgraph, isLinkPredTask);

    const {
        selectedTables,
        selectedSizes,
        expandedChipParents,
        setExpandedChipParents,
        isWarningDismissed,
        setIsWarningDismissed,
        isTableFilterActive,
        isSizeFilterActive,
        isAnyFilterActive,
        isFullExpand,
        setIsFullExpand,
        isUserCollapsingRef,
        onTableValuesChange,
        onSizeValuesChange,
        tableOptions,
        selectedTableCount,
        selectedSizeCount
    } = useSubgraphFilters({ allTableNames, subgraph });

    const resolvedTargetTableName = useMemo(() => {
        if (!isLinkPredTask) return undefined;
        if (targetTableName) return targetTableName;
        if (predictedItemTableName) return predictedItemTableName;
        const entityTableName = nodeArr.find((n) => n.id === rootId)?.data
            .tableName;
        return allTableNames.find((t) => t !== entityTableName);
    }, [
        isLinkPredTask,
        targetTableName,
        predictedItemTableName,
        nodeArr,
        rootId,
        allTableNames
    ]);

    const nodeTypeMap = useMemo(() => {
        if (!isLinkPredTask) return undefined;
        return buildNodeTypeMap({
            nodeArr,
            entityOutput,
            historicalItems,
            targetTableName: resolvedTargetTableName
        });
    }, [
        isLinkPredTask,
        nodeArr,
        entityOutput,
        historicalItems,
        resolvedTargetTableName
    ]);

    // halfSize map — RHS prediction items adopt their LHS counterpart's size.
    const halfSizeByNodeId = useMemo(() => {
        const m = new Map<string, number>();
        for (const n of nodeArr) {
            m.set(n.id, TIER_HALF_SIZES[scoreTierMap[n.id] || 0]);
        }
        return m;
    }, [nodeArr, scoreTierMap]);

    const pkeyToLhsNodeInfo = useMemo(
        () =>
            buildPkeyToLhsNodeInfo({
                nodeArr,
                targetTableName: resolvedTargetTableName,
                halfSizeByNodeId
            }),
        [nodeArr, resolvedTargetTableName, halfSizeByNodeId]
    );

    const {
        visibleNodes,
        nodeStatuses,
        tableExcludedByTable,
        pathOnlyCount,
        visibleCount,
        layoutedNodes,
        collapseExpandChipNodes,
        structuralEdges,
        legendItems
    } = useSubgraphGraph({
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
    });

    const {
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
    } = useSubgraphInteraction({
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
    });

    const predictions = usePredictionsConfig({
        taskType,
        entityOutput,
        historicalItems,
        targetTableName: resolvedTargetTableName,
        targetColor: resolvedTargetTableName
            ? tableColorMap[resolvedTargetTableName]
            : undefined,
        hoveredId: hoveredNodeId,
        selectedId: selectedNodeId,
        entityNodeId: rootId,
        pkeyToLhsNodeInfo
    });

    const isConflict =
        isTableFilterActive &&
        isSizeFilterActive &&
        Object.keys(tableExcludedByTable).length > 0;
    const showInfoBanner =
        isAnyFilterActive && !isConflict && pathOnlyCount > 0;
    const showWarningBanner = isConflict && !isWarningDismissed;
    const showWarningChip = isConflict && isWarningDismissed;

    const warningLine1 = useMemo(() => {
        const entries = Object.entries(tableExcludedByTable);
        if (entries.length === 0) return "";
        if (entries.length === 1) {
            const [tbl, cnt] = entries[0];
            return `${cnt} important node${cnt !== 1 ? "s" : ""} from table '${tbl}' ${cnt !== 1 ? "are" : "is"} dimmed because that table is deselected. Re-select to interact.`;
        }
        const total = entries.reduce((s, [, c]) => s + c, 0);
        const names = entries.map(([t]) => `'${t}'`).join(", ");
        return `${total} important nodes from tables ${names} are dimmed because those tables are deselected. Re-select to interact.`;
    }, [tableExcludedByTable]);

    return {
        displayNodes,
        displayEdges,
        legendItems,
        isLegendOpen,
        setIsLegendOpen,
        isFullscreen,
        setIsFullscreen,
        isFullExpand,
        setIsFullExpand,
        isUserCollapsingRef,
        expandedChipParents,
        setExpandedChipParents,
        allChipBoundaryParents,
        selectedNodeData,
        tableColorMap,
        selectedTables,
        selectedSizes,
        tableOptions,
        selectedTableCount,
        selectedSizeCount,
        onTableValuesChange,
        onSizeValuesChange,
        onNodeMouseEnter,
        onNodeMouseLeave,
        onNodeClick,
        onPaneClick,
        predictions,
        visibleCount,
        totalNodes,
        isAnyFilterActive,
        showInfoBanner,
        showWarningBanner,
        showWarningChip,
        isWarningDismissed,
        setIsWarningDismissed,
        warningLine1,
        pathOnlyCount
    };
}
