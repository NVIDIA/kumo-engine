// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from "vitest";

import { HighlightTypes, HoverHintsType } from "types/codeEditor";

import { parseHintsFromMintlify } from "../parseHints";

vi.mock("utils/PQLParsing/pqDocs.json", () => ({
    default: {
        assuming: [
            "Assuming Keyword",
            "Used for `assumptions`",
            "Makes **assumptions** about data\nFilters results\n Description"
        ],
        rank: [
            "Rank Function",
            "Orders **data** by `value`",
            "This function ranks data\nby specified criteria"
        ],
        predict: [
            "Predict Function",
            "Makes `predictions`",
            "Prediction **functionality**"
        ],
        where: [
            "Where Clause",
            "Filters <data> with `conditions`",
            "Filters **data** based on conditions"
        ],
        and: [
            "And Operator",
            "Logical `AND`",
            "Combines **conditions** with AND"
        ],
        or: ["Or Operator", "Logical `OR`", "Combines **conditions** with OR"],
        not: ["Not Operator", "Logical `NOT`", "Negates **conditions**"],
        contains: [
            "Contains Operator",
            "String `matching`",
            "Tests if string **contains** substring"
        ],
        is_null: [
            "Is Null Operator",
            "Null `check`",
            "Tests for **null** values"
        ],
        is_not_null: [
            "Is Not Null Operator",
            "Not null `check`",
            "Tests for **non-null** values"
        ],
        like: [
            "Like Operator",
            "Pattern `matching`",
            "Pattern **matching** operator"
        ],
        starts_with: [
            "Starts With",
            "String `prefix`",
            "Tests string **prefix**"
        ],
        ends_with: ["Ends With", "String `suffix`", "Tests string **suffix**"],
        page: [
            "Page Operator",
            "Pagination `control`",
            "Controls **page** display"
        ],
        avg: [
            "Average Function",
            "Calculates `average`",
            "Computes **average** value"
        ],
        count: [
            "Count Function",
            "Counts `records`",
            "Counts **number** of records"
        ],
        sum: ["Sum Function", "Calculates `sum`", "Computes **total** sum"],
        max: ["Max Function", "Maximum `value`", "Finds **maximum** value"],
        min: ["Min Function", "Minimum `value`", "Finds **minimum** value"],
        first: ["First Function", "First `record`", "Gets **first** record"],
        last: ["Last Function", "Last `record`", "Gets **last** record"],
        model: [
            "Model Config",
            "Defines <model> parameters",
            "Model **configuration** options"
        ],
        network: [
            "Network Layer",
            "Neural `network` setup",
            "Network layer **configuration**"
        ],
        special: [
            "Special Function",
            "Special <operations>",
            "Special **function** operations\n\nAdditional details"
        ],
        node: ["Node Function", "Graph `node`", "Represents graph **nodes**"],
        column_processing: [
            "Column Processing",
            "Process `columns`",
            "Handles column **processing** operations"
        ],
        html_test: [
            "HTML Test",
            "Contains <tags> and `code`",
            "Body with <html> **tags** and `code`"
        ]
    }
}));

const expectValidHintStructure = (hint: HoverHintsType[string]) => {
    expect(hint).toHaveProperty("title");
    expect(hint).toHaveProperty("link");
    expect(hint).toHaveProperty("type");
};

describe("parseHintsFromMintlify", () => {
    describe("Language Processing", () => {
        it("Should process PQLANG_V2 keys with valid hint structure", () => {
            const result = parseHintsFromMintlify("PQLANG_V2", {});

            expect(typeof result).toBe("object");
            Object.values(result).forEach((hint) => {
                expectValidHintStructure(hint);
                expect(Object.values(HighlightTypes)).toContain(hint.type);
            });
        });

        it("Should process YAML keys with valid hint structure", () => {
            const result = parseHintsFromMintlify("YAML", {});

            expect(typeof result).toBe("object");
            Object.values(result).forEach((hint) => {
                expectValidHintStructure(hint);
                expect(hint.type).toBe(HighlightTypes.KEYWORD);
            });
        });

        it("Should return empty hints object for unknown language", () => {
            const result = parseHintsFromMintlify("UNKNOWN_LANG", {});
            expect(Object.keys(result)).toHaveLength(0);
        });

        it("Should handle existing hints object", () => {
            const existingHints: HoverHintsType = {
                existing: {
                    title: "Existing",
                    subtitle: "Pre-existing",
                    hint: "Already there",
                    link: "http://example.com",
                    type: HighlightTypes.KEYWORD
                }
            };

            const result = parseHintsFromMintlify("PQLANG_V2", existingHints);

            expect(result.existing).toBeDefined();
            expect(result.existing.title).toBe("Existing");
            expect(result).toBe(existingHints);
        });
    });

    describe("Highlight Types", () => {
        it("Should assign correct highlight types for PQLANG_V2", () => {
            const result = parseHintsFromMintlify("PQLANG_V2", {});

            Object.values(result).forEach((hint) => {
                expect([
                    HighlightTypes.KEYWORD,
                    HighlightTypes.OPERATOR,
                    HighlightTypes.TYPE_NAME
                ]).toContain(hint.type);
            });
        });

        it("Should always use KEYWORD highlight type for YAML", () => {
            const result = parseHintsFromMintlify("YAML", {});

            Object.values(result).forEach((hint) => {
                expect(hint.type).toBe(HighlightTypes.KEYWORD);
            });
        });

        it("Should handle rank special case", () => {
            const result = parseHintsFromMintlify("PQLANG_V2", {});

            expect(result.rank).toBeDefined();
            expect(result.rank.type).toBe(HighlightTypes.KEYWORD);
        });

        it("Should handle operators with underscore replacement", () => {
            const result = parseHintsFromMintlify("PQLANG_V2", {});

            expect(result["is null operator"]).toBeDefined();
            expect(result["is null operator"].type).toBe(
                HighlightTypes.OPERATOR
            );
        });
    });

    it("Should handle column_processing special case", () => {
        const result = parseHintsFromMintlify("YAML", {});

        expect(result["encoder_overrides"]).toBeDefined();
        expect(result["encoder_overrides"].title).toBe("Column Processing");
    });

    describe("Content Transformation", () => {
        it("Should escape HTML characters when processing strings", () => {
            const result = parseHintsFromMintlify("PQLANG_V2", {});

            expect(result["where clause"].subtitle).toContain("&lt");
            expect(result["where clause"].subtitle).toContain("&gt");
        });

        it("Should convert markdown code blocks to HTML", () => {
            const result = parseHintsFromMintlify("PQLANG_V2", {});

            expect(result["assuming keyword"].subtitle).toContain(
                "<code>assumptions</code>"
            );
        });

        it("Should convert markdown bold to HTML", () => {
            const result = parseHintsFromMintlify("PQLANG_V2", {});

            expect(result["assuming keyword"].hint).toContain(
                "<strong>assumptions</strong>"
            );
        });

        it("Should filter body content correctly", () => {
            const result = parseHintsFromMintlify("PQLANG_V2", {});

            expect(result["assuming keyword"].hint).not.toContain(
                " Description"
            );
            expect(result["assuming keyword"].hint).toContain("\r\n");
        });

        it("Should generate correct links", () => {
            const result = parseHintsFromMintlify("PQLANG_V2", {});

            expect(result["assuming keyword"].link).toBe(
                "https://kumo.ai/docs/reference/assuming"
            );
        });
    });
});
