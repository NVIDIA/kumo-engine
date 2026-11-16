// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Helpers for handling RHS prediction-item clicks. Extracted from
// useSubgraphInteraction so the hook stays under the migration-baseline
// complexity ceiling.

import type { Node, PredictionItem } from "@kumo-ai/kumo-ui-catalog";

import type { GraphNode, GraphNodeData, XaiColumnArgs } from "types/explorer";
import type { ColumnArgs } from "types/tables";

const stripSchema = (s: string | undefined) =>
    s ? (s.split(".").pop() ?? s) : "";

// Build a GraphNodeData for a prediction item that has no LHS counterpart.
// Uses the prediction API's own cols + idName when available; otherwise
// falls back to borrowing idName from any same-table LHS node (matching by
// table suffix to ignore schema prefixes) and leaving cols empty.
export function synthesizePredictionItemData(
    item: PredictionItem,
    visibleNodes: GraphNode[]
): GraphNodeData {
    const itemTable = stripSchema(item.tableName);
    const sameTableNode = visibleNodes.find(
        (n) => stripSchema(n.data.tableName) === itemTable
    );
    return {
        cols: (item.cols ?? []).map((c) => c as ColumnArgs & XaiColumnArgs),
        tableName: item.tableName,
        idName: item.idName || sameTableNode?.data.idName || "id",
        value: item.id,
        totalScore: 0,
        hopLevel: item.hopLevel,
        nodeType: item.nodeType
    };
}

// Resolves a RHS prediction-item click into { hoverKey, GraphNodeData }.
// Prefers the LHS counterpart's data (full cols / scores) when present;
// falls back to synthesized data when the item is RHS-only.
export function resolvePredictionItemClick(
    node: Node,
    visibleNodes: GraphNode[]
): { hk: string; data: GraphNodeData | null } | null {
    const { hoverKey: hk, item } = node.data as {
        hoverKey?: string;
        item?: PredictionItem;
    };
    if (!hk) return null;
    const lhsData = visibleNodes.find((n) => n.id === hk)?.data ?? null;
    const data =
        lhsData ??
        (item ? synthesizePredictionItemData(item, visibleNodes) : null);
    return { hk, data };
}
