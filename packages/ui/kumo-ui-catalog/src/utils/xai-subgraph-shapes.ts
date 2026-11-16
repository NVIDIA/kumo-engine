// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// TLP node-type system: classification, shapes, and prediction item metadata.

/** TLP node classification — drives shape + emerald GT ring. */
export const NODE_TYPE = {
    ORGANIC: "organic",
    RECURRING: "recurring",
    EXPLORATORY: "exploratory",
    HIT: "hit",
    RECURRING_HIT: "recurring_hit"
} as const;

export type PredictedNodeType = (typeof NODE_TYPE)[keyof typeof NODE_TYPE];

/** Visual shape variants. Derived from PredictedNodeType, or "circle" by default. */
export type NodeShapeVariant = "circle" | "diamond" | "triangle";

/** px halo each side when a node carries the emerald GT ring (Figma spec). */
export const GT_RING_HALO = 2;

/** Maps a TLP node type to its shape + GT-ring visual treatment. */
export function getShapeForNodeType(t?: PredictedNodeType): {
    variant: NodeShapeVariant;
    hasRing: boolean;
} {
    switch (t) {
        case NODE_TYPE.RECURRING:
            return { variant: "diamond", hasRing: false };
        case NODE_TYPE.EXPLORATORY:
            return { variant: "triangle", hasRing: false };
        case NODE_TYPE.HIT:
            return { variant: "triangle", hasRing: true };
        case NODE_TYPE.RECURRING_HIT:
            return { variant: "diamond", hasRing: true };
        case NODE_TYPE.ORGANIC:
        default:
            return { variant: "circle", hasRing: false };
    }
}

/** Total layout footprint of a subgraph node (base shape + GT ring halo). */
export function getNodeFootprint(
    halfSize: number,
    nodeType?: PredictedNodeType
): number {
    const { hasRing } = getShapeForNodeType(nodeType);
    return halfSize * 2 + (hasRing ? GT_RING_HALO * 2 : 0);
}

/** One column on a predicted item, as shipped by the prediction API. */
export interface PredictionItemCol {
    name: string;
    value: unknown;
    stype?: string;
    grad_score?: number;
}

/**
 * One entry in the TLP Predictions panel. `id` matches the LHS subgraph
 * node's id when the item exists on both sides (drives cross-highlight).
 */
export interface PredictionItem {
    id: string;
    rank: number;
    tableName: string;
    nodeType: PredictedNodeType;
    color: string;
    /** Set for items that also appear in the LHS subgraph; drives 2-line tooltip. */
    hopLevel?: number;
    /** Inscribed half-size (px). When set, must match LHS counterpart for visual identity. */
    halfSize?: number;
    /** Pkey column name (`NodeOutputData.id_name`). */
    idName?: string;
    /** Column values (`NodeOutputData.cols`). */
    cols?: PredictionItemCol[];
}

/** Per-type label + info-tooltip description (Figma spec). */
export const PREDICTED_NODE_TYPE_INFO: Record<
    PredictedNodeType,
    { label: string; description: string }
> = {
    [NODE_TYPE.ORGANIC]: {
        label: "Organic",
        description:
            "This item is in the entity's history but wasn't in the model's predictions."
    },
    [NODE_TYPE.RECURRING]: {
        label: "Recurring",
        description:
            "This item is in the entity's history and the model is predicting it again."
    },
    [NODE_TYPE.EXPLORATORY]: {
        label: "Exploratory",
        description:
            "This item is a first-time prediction, never in the entity's history."
    },
    [NODE_TYPE.HIT]: {
        label: "Hit",
        description: "Predicted item that matches Ground Truth."
    },
    [NODE_TYPE.RECURRING_HIT]: {
        label: "Recurring hit",
        description:
            "This item is in the entity's history, predicted again, and confirmed by Ground Truth."
    }
};

/** Ordering used for the legend's "Predicted node types" section (Figma spec). */
export const PREDICTED_TYPE_LEGEND_ORDER: PredictedNodeType[] = [
    NODE_TYPE.ORGANIC,
    NODE_TYPE.RECURRING,
    NODE_TYPE.EXPLORATORY,
    NODE_TYPE.HIT,
    NODE_TYPE.RECURRING_HIT
];
