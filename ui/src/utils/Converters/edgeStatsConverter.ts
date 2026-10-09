// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EdgeHealthStats } from "gen/v1/types";

import { EdgeStats } from "types/graphs";

import { graphEdgeThreshold } from "utils/Constants";
import { parseNumbers } from "utils/Helpers/numberHelpers";

type LinkStat = NonNullable<EdgeHealthStats["link_stats"]>[number];

/** Column identity + "table.column" display fields for one link. */
function edgeColumnFields(element: LinkStat): Partial<EdgeStats> {
    const colA = element?.col_a;
    const colB = element?.col_b;
    return {
        col_a_table_name: colA?.table_name,
        col_a_column_name: colA?.column_name,
        col_a_table_id: colA?.table_id,
        col_b_table_name: colB?.table_name,
        col_b_column_name: colB?.column_name,
        col_b_table_id: colB?.table_id,
        col_a: colA?.table_name + "." + colA?.column_name,
        col_b: colB?.table_name + "." + colB?.column_name
    };
}

function edgeStatFromLink(element: LinkStat): EdgeStats {
    const matches = element?.matches;
    const percents = element?.percents;
    const percentAInB = percents?.a_in_b ?? 0; // TODO: Handle downstream
    const percentBInA = percents?.b_in_a ?? 0; // TODO: Handle downstream
    const weakAInB = percentAInB < graphEdgeThreshold;
    const weakBInA = percentBInA < graphEdgeThreshold;
    return {
        ...edgeColumnFields(element),
        link_type: element?.link_type,
        matches_a_in_b: parseNumbers(matches?.a_in_b) as string,
        matches_a_in_b_original: matches?.a_in_b ?? 0, // TODO: Handle downstream
        matches_b_in_a: parseNumbers(matches?.b_in_a) as string,
        matches_b_in_a_original: matches?.b_in_a ?? 0, // TODO: Handle downstream
        weak_a_in_b: weakAInB,
        weak_b_in_a: weakBInA,
        weak_edge: weakAInB && weakBInA,
        /* Add % symbol to percent_a_in_b and percent_b_in_a */
        percent_a_in_b: percentAInB.toFixed(2) + "%",
        percent_b_in_a: percentBInA.toFixed(2) + "%"
    } as EdgeStats;
}

export function edgeStatsFromEdgeStatsResponse(
    edgeStatsResponseJson: EdgeHealthStats
): Array<EdgeStats> {
    if (!edgeStatsResponseJson?.link_stats) {
        return [];
    }
    return edgeStatsResponseJson.link_stats.map(edgeStatFromLink);
}
