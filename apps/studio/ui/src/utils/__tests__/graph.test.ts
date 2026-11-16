// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { GraphTables } from "types/graphs";

import { getpKeys } from "utils/graph";

describe("getpKeys", () => {
    const allTables = {
        users: { pkey: "id" },
        posts: { pkey: "post_id" }
    } as unknown as GraphTables;

    it("should return matching primary key columns", () => {
        const cols = [
            { table_name: "users", col_name: "id" },
            { table_name: "posts", col_name: "post_id" },
            { table_name: "posts", col_name: "title" } // not a pkey
        ];

        const result = getpKeys(cols, allTables);
        expect(result).toHaveLength(2);
        expect(result).toEqual([
            { table_name: "users", col_name: "id" },
            { table_name: "posts", col_name: "post_id" }
        ]);
    });

    it.each([
        {
            description: "should return empty array if no keys match",
            cols: [
                { table_name: "users", col_name: "username" },
                { table_name: "posts", col_name: "title" }
            ],
            tables: allTables
        },
        {
            description: "should handle empty input array",
            cols: [],
            tables: allTables
        },
        {
            description: "should skip if table is not found in allTables",
            cols: [{ table_name: "non_existing", col_name: "id" }],
            tables: allTables
        },
        {
            description: "should skip if pkey is invalid",
            cols: [{ table_name: "invalid_table", col_name: "id" }],
            tables: {
                invalid_table: { pkey: null }
            } as unknown as GraphTables
        }
    ])("$description", ({ cols, tables }) => {
        const result = getpKeys(cols, tables);
        expect(result).toEqual([]);
    });
});
