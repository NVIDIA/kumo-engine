// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
    keywords,
    multiKeywordMap,
    multiWordKeywordStart
} from "../snippetsAndKeywords";

describe("snippetsAndKeywords", () => {
    describe("keywords", () => {
        it("Should have 35 keyword entries", () => {
            expect(keywords).toBeDefined();
            expect(keywords.length).toBe(35);
        });

        it("Should have uppercase labels for all keywords", () => {
            keywords.forEach((keyword) => {
                expect(keyword.label).toBe(keyword.label.toUpperCase());
            });
        });

        it("Should have type 'variable' for all keywords", () => {
            keywords.forEach((keyword) => {
                expect(keyword.type).toBe("variable");
            });
        });

        it("Should not offer the removed FOR form", () => {
            expect(keywords.map((keyword) => keyword.label)).not.toContain(
                "FOR"
            );
        });

        it.each([
            { label: "PREDICT" },
            { label: "WHERE" },
            { label: "CLASSIFY" },
            { label: "RANK TOP K" },
            { label: "FOR EACH" },
            { label: "AND" },
            { label: "OR" },
            { label: "NOT" },
            { label: "CONTAINS" },
            { label: "LIKE" },
            { label: "FORECAST" },
            { label: "TIMEFRAMES" }
        ])("Should include keyword $label", ({ label }) => {
            const labels = keywords.map((k) => k.label);
            expect(labels).toContain(label);
        });

        it.each([
            { label: "AVG" },
            { label: "COUNT" },
            { label: "COUNT_DISTINCT" },
            { label: "FIRST" },
            { label: "LAST" },
            { label: "MAX" },
            { label: "MIN" },
            { label: "SUM" }
        ])("Should include aggregation function $label", ({ label }) => {
            const labels = keywords.map((k) => k.label);
            expect(labels).toContain(label);
        });

        it.each([
            { label: "!=" },
            { label: "<" },
            { label: "<=" },
            { label: "=" },
            { label: ">" },
            { label: ">=" }
        ])("Should include comparison operator $label", ({ label }) => {
            const labels = keywords.map((k) => k.label);
            expect(labels).toContain(label);
        });
    });

    describe("multiWordKeywordStart", () => {
        it("Should have 7 multi-word keyword start entries", () => {
            expect(multiWordKeywordStart).toBeDefined();
            expect(multiWordKeywordStart.length).toBe(7);
            expect(multiWordKeywordStart).toEqual([
                "ends",
                "is",
                "not",
                "starts",
                "top",
                "rank",
                "for"
            ]);
        });
    });

    describe("multiKeywordMap", () => {
        it("Should have 6 entries in the map", () => {
            expect(multiKeywordMap).toBeDefined();
            expect(multiKeywordMap.size).toBe(6);
        });

        it.each([
            {
                description: "Should map 'ends' to ['WITH']",
                key: "ends",
                expectedLabels: ["WITH"]
            },
            {
                description: "Should map 'is' to ['NOT', 'NULL']",
                key: "is",
                expectedLabels: ["NOT", "NULL"]
            },
            {
                description: "Should map 'not' to ['NULL', 'LIKE', 'CONTAINS']",
                key: "not",
                expectedLabels: ["NULL", "LIKE", "CONTAINS"]
            },
            {
                description: "Should map 'starts' to ['WITH']",
                key: "starts",
                expectedLabels: ["WITH"]
            },
            {
                description: "Should map 'rank' to ['TOP']",
                key: "rank",
                expectedLabels: ["TOP"]
            },
            {
                description: "Should map 'for' to ['EACH']",
                key: "for",
                expectedLabels: ["EACH"]
            }
        ])("$description", ({ key, expectedLabels }) => {
            const completions = multiKeywordMap.get(key);

            expect(completions).toBeDefined();
            expect(completions?.length).toBe(expectedLabels.length);

            const labels = completions?.map((c) => c.label);
            expectedLabels.forEach((label) => {
                expect(labels).toContain(label);
            });
        });

        it("Should have type 'variable' for all multi-keyword map entries", () => {
            multiKeywordMap.forEach((completions) => {
                completions.forEach((completion) => {
                    expect(completion.type).toBe("variable");
                });
            });
        });

        it("Should have uppercase labels for all multi-keyword map entries", () => {
            multiKeywordMap.forEach((completions) => {
                completions.forEach((completion) => {
                    expect(completion.label).toBe(
                        completion.label.toUpperCase()
                    );
                });
            });
        });
    });
});
