// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Edge, type Node } from "@xyflow/react";

import {
    getNodeFootprint,
    type PredictedNodeType
} from "./xai-subgraph-shapes";

export const TABLE_COLORS = [
    "var(--color-pink-500)", // entity / root
    "var(--color-cyan-600)",
    "var(--color-orange-600)",
    "var(--color-lime-600)",
    "var(--color-indigo-600)",
    "var(--color-teal-600)",
    "var(--color-mauve-600)",
    "var(--color-amber-600)",
    "var(--color-violet-600)",
    "var(--color-green-600)",
    "var(--color-sky-600)",
    "var(--color-mist-600)",
    "var(--color-fuchsia-600)",
    "var(--color-yellow-600)"
];

/** Assigns colors from TABLE_COLORS; rootTable gets TABLE_COLORS[0] (pink-500). */
export function buildTableColorMap(
    tableNames: string[],
    rootTable?: string
): Record<string, string> {
    const seen = new Set<string>();
    const ordered: string[] = [];
    if (rootTable && tableNames.includes(rootTable)) {
        seen.add(rootTable);
        ordered.push(rootTable);
    }
    for (const t of tableNames) {
        if (t && !seen.has(t)) {
            seen.add(t);
            ordered.push(t);
        }
    }
    const map: Record<string, string> = {};
    ordered.forEach((t, i) => {
        map[t] = TABLE_COLORS[i % TABLE_COLORS.length];
    });
    return map;
}

// Matches the ~20px expand-chip pill width so it fits in the sibling gap.
export const XAI_SUBGRAPH_LAYOUT_NODESEP = 20;

const LAYOUT_RANKSEP = 55;

function getNodeSize(n: Node): { w: number; h: number } {
    if (n.type === "subgraphNode") {
        const data = n.data as {
            halfSize: number;
            nodeType?: PredictedNodeType;
        };
        const total = getNodeFootprint(data.halfSize, data.nodeType);
        return { w: total, h: total };
    }
    if (n.type === "collapseChip") return { w: 52, h: 16 };
    return { w: 20, h: 20 };
}

export function xaiSubgraphApplyLayout(nodes: Node[], edges: Edge[]): Node[] {
    if (nodes.length === 0) return nodes;

    const MARGIN = 20;
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));

    // Build parent→children map from tree edges
    const children: Record<string, string[]> = {};
    const hasParent = new Set<string>();
    for (const n of nodes) children[n.id] = [];
    for (const e of edges) {
        if (children[e.source] !== undefined && nodeMap.has(e.target)) {
            children[e.source].push(e.target);
            hasParent.add(e.target);
        }
    }

    const roots = nodes.filter((n) => !hasParent.has(n.id));

    // Bottom-up: minimum width needed by each subtree
    const subtreeW: Record<string, number> = {};
    const calcWidth = (id: string): number => {
        const ch = children[id] ?? [];
        const { w } = getNodeSize(nodeMap.get(id)!);
        if (ch.length === 0) return (subtreeW[id] = w);
        const chTotal =
            ch.reduce((s, c) => s + calcWidth(c), 0) +
            (ch.length - 1) * XAI_SUBGRAPH_LAYOUT_NODESEP;
        return (subtreeW[id] = Math.max(w, chTotal));
    };
    roots.forEach((r) => calcWidth(r.id));

    // Top-down: assign positions — children centered under parent, no crossings
    const pos: Record<string, { x: number; y: number }> = {};
    const assignPos = (id: string, leftBound: number, depth: number): void => {
        const ch = children[id] ?? [];
        const sw = subtreeW[id];
        pos[id] = { x: leftBound + sw / 2, y: MARGIN + depth * LAYOUT_RANKSEP };
        if (ch.length === 0) return;
        const chTotal =
            ch.reduce((s, c) => s + subtreeW[c], 0) +
            (ch.length - 1) * XAI_SUBGRAPH_LAYOUT_NODESEP;
        let childLeft = leftBound + (sw - chTotal) / 2;
        for (const cid of ch) {
            assignPos(cid, childLeft, depth + 1);
            childLeft += subtreeW[cid] + XAI_SUBGRAPH_LAYOUT_NODESEP;
        }
    };
    let rootLeft = MARGIN;
    for (const r of roots) {
        assignPos(r.id, rootLeft, 0);
        rootLeft += subtreeW[r.id] + XAI_SUBGRAPH_LAYOUT_NODESEP;
    }

    return nodes.map((n) => {
        const { x, y } = pos[n.id] ?? { x: 0, y: 0 };
        const { w, h } = getNodeSize(n);
        return {
            ...n,
            position: { x: x - w / 2, y: y - h / 2 },
            width: w,
            height: h
        };
    });
}
