// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Predictions panel layout: grid math + ReactFlow node/edge builders.

import { type Edge, type Node } from "@xyflow/react";

import {
    type PredictionItemNodeData,
    type PredictionsPanelBackgroundData,
    type XaiSubgraphPredictions
} from "@/types/visual-graph";

/** Handle id on the entity node used as the source for the dashed entity→panel edge. */
export const ENTITY_PREDICTIONS_HANDLE_ID = "predictions";

const PREDICTIONS_PANEL_NODE_ID = "__predictions-panel";
export const PREDICTION_ITEM_PREFIX = "__prediction-item-";
/** Gap (px) between the rightmost subgraph node and the panel's left edge. */
const PREDICTIONS_PANEL_GAP = 72;
const PANEL_PADDING_X = 24;
const PANEL_PADDING_Y = 16;
const PANEL_TITLE_BLOCK_HEIGHT = 28;
// ITEM_WIDTH matches the actual visible content (#N text + NodeShape, ≤ ~18px);
// larger values accumulate unused space on the panel's right edge.
const ITEM_WIDTH = 20;
const ITEM_HEIGHT = 38;
const ITEM_GAP_X = 12;
const ITEM_GAP_Y = 8;
const PREDICTION_GRID_COLS = 4;

/** Panel position: right of the rightmost node, top-aligned with the topmost. */
function computePredictionsPanelPosition(displayNodes: Node[]): {
    x: number;
    y: number;
} {
    if (displayNodes.length === 0) return { x: 0, y: 0 };
    let maxRight = -Infinity;
    let minTop = Infinity;
    for (const n of displayNodes) {
        const right = n.position.x + (n.width ?? 0);
        if (right > maxRight) maxRight = right;
        if (n.position.y < minTop) minTop = n.position.y;
    }
    return { x: maxRight + PREDICTIONS_PANEL_GAP, y: minTop };
}

/** Builds the panel's background node + one node per prediction item, grid-positioned. */
export function buildPredictionsPanelNodes(
    displayNodes: Node[],
    predictions: XaiSubgraphPredictions
): Node[] {
    const basePos = computePredictionsPanelPosition(displayNodes);
    const rowCount = Math.ceil(predictions.items.length / PREDICTION_GRID_COLS);
    const gridWidth =
        PREDICTION_GRID_COLS * ITEM_WIDTH +
        (PREDICTION_GRID_COLS - 1) * ITEM_GAP_X;
    const gridHeight =
        rowCount * ITEM_HEIGHT + Math.max(0, rowCount - 1) * ITEM_GAP_Y;
    const backgroundNode: Node = {
        id: PREDICTIONS_PANEL_NODE_ID,
        type: "predictionsPanelBackground",
        position: basePos,
        data: {
            title: "Predictions",
            width: gridWidth + PANEL_PADDING_X * 2,
            height: PANEL_TITLE_BLOCK_HEIGHT + gridHeight + PANEL_PADDING_Y * 2
        } satisfies PredictionsPanelBackgroundData,
        draggable: false,
        selectable: false,
        zIndex: 0
    };
    const itemNodes: Node[] = predictions.items.map((item, idx) => {
        const col = idx % PREDICTION_GRID_COLS;
        const row = Math.floor(idx / PREDICTION_GRID_COLS);
        const data: PredictionItemNodeData = {
            item,
            hoverKey: item.id,
            isHovered: predictions.hoveredId === item.id,
            isSelected: predictions.selectedId === item.id,
            isDimmed:
                predictions.hoveredId !== null &&
                predictions.hoveredId !== item.id
        };
        return {
            id: `${PREDICTION_ITEM_PREFIX}${item.id}`,
            type: "predictionItem",
            position: {
                x:
                    basePos.x +
                    PANEL_PADDING_X +
                    col * (ITEM_WIDTH + ITEM_GAP_X),
                y:
                    basePos.y +
                    PANEL_PADDING_Y +
                    PANEL_TITLE_BLOCK_HEIGHT +
                    row * (ITEM_HEIGHT + ITEM_GAP_Y)
            },
            width: ITEM_WIDTH,
            height: ITEM_HEIGHT,
            data,
            draggable: false,
            selectable: false,
            zIndex: 1
        };
    });
    return [backgroundNode, ...itemNodes];
}

/** Bezier dashed edge from the entity to the panel background (Figma S-curve). */
export function buildPredictionsPanelEdge(entityNodeId: string): Edge {
    return {
        id: "__predictions-panel-edge",
        source: entityNodeId,
        sourceHandle: ENTITY_PREDICTIONS_HANDLE_ID,
        target: PREDICTIONS_PANEL_NODE_ID,
        type: "default",
        style: {
            stroke: "var(--color-slate-300)",
            strokeWidth: 1,
            strokeDasharray: "4 4"
        }
    };
}
