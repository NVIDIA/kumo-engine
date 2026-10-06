// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Builds XaiSubgraphPredictions for the RHS panel from entityOutput +
// historicalItems. node_type derivation lives in ./subgraphNodeTypeBuild.
import { useMemo } from "react";

import type {
    PredictionItem,
    XaiSubgraphPredictions
} from "@kumo-ai/kumo-ui-catalog";
import type { HistoricalItems } from "gen/v0/types";

import type { LinkPredictionEntityOutput } from "types/explorer";
import { TaskType } from "types/predictiveQueries";

import { isLinkPredictionType } from "utils/Helpers/typeGuards";

import {
    deriveNodeType,
    type LhsNodeInfo,
    toIdSet
} from "./subgraphNodeTypeBuild";

interface UsePredictionsConfigInput {
    taskType: TaskType | undefined;
    entityOutput: LinkPredictionEntityOutput | null | undefined;
    historicalItems: HistoricalItems | null | undefined;
    targetTableName: string | undefined;
    targetColor: string | undefined;
    hoveredId: string | null;
    selectedId: string | null;
    entityNodeId: string | undefined;
    pkeyToLhsNodeInfo?: Map<string, LhsNodeInfo>;
}

export function usePredictionsConfig({
    taskType,
    entityOutput,
    historicalItems,
    targetTableName,
    targetColor,
    hoveredId,
    selectedId,
    entityNodeId,
    pkeyToLhsNodeInfo
}: UsePredictionsConfigInput): XaiSubgraphPredictions | undefined {
    const items = useMemo<PredictionItem[] | undefined>(() => {
        if (!isLinkPredictionType(taskType ?? "")) return undefined;
        if (!entityOutput?.prediction || entityOutput.prediction.length === 0)
            return undefined;
        if (!targetTableName || !targetColor) return undefined;
        const historyIds = toIdSet(historicalItems?.nodes);
        const gtIds = toIdSet(entityOutput.target);
        return entityOutput.prediction.map((p, idx) => {
            const pkey = String(p.id);
            const lhsInfo = pkeyToLhsNodeInfo?.get(pkey);
            return {
                id: lhsInfo?.id ?? pkey,
                rank: idx + 1,
                tableName: targetTableName,
                nodeType: deriveNodeType(
                    historyIds.has(pkey),
                    true,
                    gtIds.has(pkey)
                ),
                color: targetColor,
                hopLevel: lhsInfo?.hopLevel,
                halfSize: lhsInfo?.halfSize,
                idName: p.id_name ?? undefined,
                cols: p.cols.map((col) => ({
                    ...col,
                    grad_score: col.grad_score ?? undefined
                }))
            };
        });
    }, [
        taskType,
        entityOutput,
        historicalItems,
        targetTableName,
        targetColor,
        pkeyToLhsNodeInfo
    ]);

    return useMemo(() => {
        if (!items || !entityNodeId) return undefined;
        return { items, entityNodeId, hoveredId, selectedId };
    }, [items, entityNodeId, hoveredId, selectedId]);
}
