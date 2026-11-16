// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Client-side node_type derivation. Remove when backend #28750 ships.

import type { PredictedNodeType } from "@kumo-ai/kumo-ui-catalog";
import type { HistoricalItems, NodeOutputData } from "gen/v0/types";

import type { GraphNode, LinkPredictionEntityOutput } from "types/explorer";

export interface LhsNodeInfo {
    id: string;
    hopLevel: number;
    halfSize: number | undefined;
}

export function deriveNodeType(
    inHistory: boolean,
    inPredictions: boolean,
    inGT: boolean
): PredictedNodeType {
    if (inPredictions) {
        if (inHistory) return inGT ? "recurring_hit" : "recurring";
        if (inGT) return "hit";
        return "exploratory";
    }
    return "organic";
}

// Runtime guard — non-link-pred tasks (e.g. regression) ship a scalar, not an array.
export function toIdSet(nodes: NodeOutputData[] | undefined): Set<string> {
    if (!Array.isArray(nodes)) return new Set();
    return new Set(nodes.map((n) => String(n.id)));
}

export function buildPkeyToLhsNodeInfo({
    nodeArr,
    targetTableName,
    halfSizeByNodeId
}: {
    nodeArr: GraphNode[];
    targetTableName: string | undefined;
    halfSizeByNodeId?: Map<string, number>;
}): Map<string, LhsNodeInfo> {
    const map = new Map<string, LhsNodeInfo>();
    if (!targetTableName) return map;
    for (const node of nodeArr) {
        if (node.data.tableName !== targetTableName) continue;
        map.set(String(node.data.value), {
            id: node.id,
            hopLevel: node.data.hopLevel ?? 0,
            halfSize: halfSizeByNodeId?.get(node.id)
        });
    }
    return map;
}

export function buildNodeTypeMap({
    nodeArr,
    entityOutput,
    historicalItems,
    targetTableName
}: {
    nodeArr: GraphNode[];
    entityOutput: LinkPredictionEntityOutput | null | undefined;
    historicalItems: HistoricalItems | null | undefined;
    targetTableName: string | undefined;
}): Map<string, PredictedNodeType> {
    const map = new Map<string, PredictedNodeType>();
    if (!targetTableName) return map;
    const historyIds = toIdSet(historicalItems?.nodes);
    const predictionIds = toIdSet(entityOutput?.prediction);
    const gtIds = toIdSet(entityOutput?.target);
    for (const node of nodeArr) {
        if (node.data.tableName !== targetTableName) continue;
        const value = String(node.data.value);
        const inHistory = historyIds.has(value);
        const inPrediction = predictionIds.has(value);
        const inGT = gtIds.has(value);
        if (!inHistory && !inPrediction && !inGT) continue;
        map.set(node.id, deriveNodeType(inHistory, inPrediction, inGT));
    }
    return map;
}
