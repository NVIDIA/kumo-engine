// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { edgeStatsFromEdgeStatsResponse } from "../edgeStatsConverter";

import {
    mockEdgeHealthStats,
    mockWeakEdgeHealthStats
} from "./MockResponse/mockResponses";

describe("edgeStatsFromEdgeStatsResponse", () => {
    it("Should parse edge stats correctly", () => {
        const result = edgeStatsFromEdgeStatsResponse(mockEdgeHealthStats);
        expect(result).toHaveLength(1);
        const stat = result[0];
        expect(stat.col_a).toBe("users.id");
        expect(stat.col_b).toBe("orders.user_id");
        expect(stat.col_a_table_name).toBe("users");
        expect(stat.col_b_table_name).toBe("orders");
        expect(stat.col_a_column_name).toBe("id");
        expect(stat.col_b_column_name).toBe("user_id");
        expect(stat.col_a_table_id).toBe("1");
        expect(stat.col_b_table_id).toBe("2");
        expect(stat.link_type).toBe("FK");
        expect(stat.matches_a_in_b).toBe("95");
        expect(stat.matches_b_in_a).toBe("90");
        expect(stat.percent_a_in_b).toBe("95.00%");
        expect(stat.percent_b_in_a).toBe("90.00%");
        expect(stat.weak_a_in_b).toBe(false);
        expect(stat.weak_b_in_a).toBe(false);
        expect(stat.weak_edge).toBe(false);
    });

    it("Should handle null link_stats", () => {
        const result = edgeStatsFromEdgeStatsResponse({
            is_ready: true,
            link_stats: undefined
        });
        expect(result).toEqual([]);
    });

    it("Should mark weak edges correctly based on threshold", () => {
        const weakResponse = {
            ...mockWeakEdgeHealthStats,
            link_stats: [
                {
                    ...mockWeakEdgeHealthStats.link_stats[0],
                    percents: {
                        a_in_b: 19.8,
                        b_in_a: 16.9
                    }
                }
            ]
        };
        const result = edgeStatsFromEdgeStatsResponse(weakResponse);
        const stat = result[0];
        expect(stat.weak_edge).toBe(true);
        expect(stat.weak_a_in_b).toBe(true);
        expect(stat.weak_b_in_a).toBe(true);
        expect(stat.percent_a_in_b).toBe("19.80%");
        expect(stat.percent_b_in_a).toBe("16.90%");
    });
});
