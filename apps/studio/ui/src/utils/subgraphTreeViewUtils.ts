// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { extractNodesFromSubgraphWithHops } from "utils/Converters/explorerConverter";

export const TIER_LABELS = ["Bottom 25%", "25–50%", "50–75%", "Top 25%"];
export const TIER_HALF_SIZES = [3.5, 4, 4.5, 5]; // px half-size per tier (bottom→top), diameters: 7, 8, 9, 10px
export const DEFAULT_MAX_HOP = 2;

export const SIZE_OPTIONS = [
    ...[...TIER_LABELS].reverse().map((label) => ({ value: label, label })),
    { value: "__clear__", label: "Clear", showCheckbox: false }
];

type NodeArr = ReturnType<typeof extractNodesFromSubgraphWithHops>["nodeArr"];

export function computeScoreTierMap(nodeArr: NodeArr): Record<string, number> {
    const scores = nodeArr.map((n) => n.data.totalScore).sort((a, b) => a - b);
    const len = scores.length;
    if (len === 0) return {};
    const q25 = scores[Math.floor(len * 0.25)] ?? 0;
    const q50 = scores[Math.floor(len * 0.5)] ?? 0;
    const q75 = scores[Math.floor(len * 0.75)] ?? 0;
    const map: Record<string, number> = {};
    for (const node of nodeArr) {
        const s = node.data.totalScore;
        map[node.id] = s >= q75 ? 3 : s >= q50 ? 2 : s >= q25 ? 1 : 0;
    }
    return map;
}

export function getPathToRoot(
    nodeId: string,
    parentMap: Record<string, string>
): Set<string> {
    const path = new Set<string>();
    let cur: string | undefined = nodeId;
    while (cur) {
        path.add(cur);
        cur = parentMap[cur];
    }
    return path;
}
