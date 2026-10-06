// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CompletionContext } from "@codemirror/autocomplete";
import { EditorState } from "@codemirror/state";
import { describe, expect, it } from "vitest";

import { JSTable } from "types/tables";

import { autocompleteWithGraphs } from "../autocomplete";

const createMockTables = (): JSTable[] => [
    {
        name_alias: "users",
        columns: [{ name: "id" }, { name: "name" }, { name: "email" }]
    } as JSTable,
    {
        name_alias: "orders",
        columns: [{ name: "order_id" }, { name: "user_id" }, { name: "total" }]
    } as JSTable
];

describe("autocompleteWithGraphs", () => {
    const createContext = (docText: string, pos: number): CompletionContext => {
        const state = EditorState.create({ doc: docText });
        return {
            state,
            pos,
            explicit: false,
            matchBefore: (regex: RegExp) => {
                const textBefore = docText.substring(0, pos);
                const match = textBefore.match(new RegExp(regex.source + "$"));
                if (match && match[0]) {
                    return {
                        from: pos - match[0].length,
                        to: pos,
                        text: match[0]
                    };
                }
                return null;
            },
            tokenBefore: () => {
                return null;
            },
            aborted: false,
            addEventListener: () => {}
        } as CompletionContext;
    };

    const getAutocompleteResult = (
        tables: JSTable[],
        docText: string,
        pos: number
    ) => {
        const autocomplete = autocompleteWithGraphs(tables);
        const context = createContext(docText, pos);
        return autocomplete(context);
    };

    describe("Keyword completions", () => {
        it.each([
            { word: "PRED", expectedKeyword: "PREDICT" },
            { word: "WHE", expectedKeyword: "WHERE" },
            { word: "CLA", expectedKeyword: "CLASSIFY" },
            { word: "FORE", expectedKeyword: "FORECAST" },
            { word: "TIME", expectedKeyword: "TIMEFRAMES" }
        ])(
            "Should suggest $expectedKeyword for $word",
            ({ word, expectedKeyword }) => {
                const result = getAutocompleteResult([], word, word.length);

                expect(result).not.toBeNull();
                if (result) {
                    const labels = result.options.map((opt) => opt.label);
                    expect(labels).toContain(expectedKeyword);
                }
            }
        );
    });

    describe("Table name completions", () => {
        it("Should include table names in completions", () => {
            const tables = createMockTables();
            const result = getAutocompleteResult(tables, "u", 1);

            expect(result).not.toBeNull();
            if (result) {
                const tableNames = result.options
                    .filter((opt) => opt.type === "constant")
                    .map((opt) => opt.label);
                expect(tableNames).toContain("users");
                expect(tableNames).toContain("orders");
            }
        });

        it("Should combine keywords and table names", () => {
            const tables = createMockTables();
            const result = getAutocompleteResult(tables, "W", 1);

            expect(result).not.toBeNull();
            if (result) {
                const hasKeywords = result.options.some(
                    (opt) => opt.label === "WHERE"
                );
                const hasTables = result.options.some(
                    (opt) => opt.type === "constant"
                );
                expect(hasKeywords).toBe(true);
                expect(hasTables).toBe(true);
            }
        });
    });

    describe("Column completions after table name", () => {
        it.each([
            {
                description: "Should suggest columns for users table",
                docText: "users.i",
                pos: 7,
                expectedColumns: ["id", "name", "email"],
                notExpectedColumns: []
            },
            {
                description: "Should suggest columns for orders table",
                docText: "orders.o",
                pos: 8,
                expectedColumns: ["order_id", "user_id", "total"],
                notExpectedColumns: ["id", "email"]
            }
        ])(
            "$description",
            ({ docText, pos, expectedColumns, notExpectedColumns }) => {
                const tables = createMockTables();
                const result = getAutocompleteResult(tables, docText, pos);

                expect(result).not.toBeNull();
                if (result) {
                    const columnLabels = result.options.map((opt) => opt.label);
                    expectedColumns.forEach((col) => {
                        expect(columnLabels).toContain(col);
                    });
                    notExpectedColumns.forEach((col) => {
                        expect(columnLabels).not.toContain(col);
                    });
                }
            }
        );

        it("Should return empty completions for invalid table name", () => {
            const tables = createMockTables();
            const result = getAutocompleteResult(tables, "invalid.col", 11);

            expect(result).not.toBeNull();
            if (result) {
                expect(result.options).toHaveLength(0);
            }
        });
    });

    describe("Multi-word keyword completions", () => {
        it.each([
            {
                description: "Should suggest EACH after FOR",
                docText: "FOR e",
                pos: 5,
                expectedKeywords: ["EACH"]
            },
            {
                description: "Should suggest NULL and NOT after IS",
                docText: "IS n",
                pos: 4,
                expectedKeywords: ["NULL", "NOT"]
            },
            {
                description: "Should suggest WITH after STARTS",
                docText: "STARTS w",
                pos: 8,
                expectedKeywords: ["WITH"]
            },
            {
                description: "Should suggest WITH after ENDS",
                docText: "ENDS w",
                pos: 6,
                expectedKeywords: ["WITH"]
            },
            {
                description: "Should suggest TOP after RANK",
                docText: "RANK t",
                pos: 6,
                expectedKeywords: ["TOP"]
            },
            {
                description: "Should suggest LIKE and CONTAINS after NOT",
                docText: "NOT l",
                pos: 5,
                expectedKeywords: ["LIKE", "CONTAINS"]
            }
        ])("$description", ({ docText, pos, expectedKeywords }) => {
            const result = getAutocompleteResult([], docText, pos);

            expect(result).not.toBeNull();
            if (result) {
                const labels = result.options.map((opt) => opt.label);
                expectedKeywords.forEach((keyword) => {
                    expect(labels).toContain(keyword);
                });
            }
        });
    });

    describe("Edge cases", () => {
        it.each([
            {
                description: "Should handle tables without columns",
                tables: [{ name_alias: "empty_table" } as JSTable],
                docText: "e",
                pos: 1,
                expectNull: false
            },
            {
                description: "Should handle single word in document",
                tables: [],
                docText: "PREDICT",
                pos: 7,
                expectNull: false
            },
            {
                description:
                    "Should return null for whitespace-only cursor position",
                tables: [],
                docText: "PREDICT ",
                pos: 8,
                expectNull: true
            }
        ])("$description", ({ tables, docText, pos, expectNull }) => {
            const result = getAutocompleteResult(tables, docText, pos);

            if (expectNull) {
                expect(result).toBeNull();
            } else {
                expect(result).not.toBeNull();
            }
        });
    });

    it("Should return null when no word is being typed", () => {
        const result = getAutocompleteResult([], "", 0);

        expect(result).toBeNull();
    });
});
