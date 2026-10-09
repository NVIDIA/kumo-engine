// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TreeFragment } from "@lezer/common";
import { describe, expect, it } from "vitest";

import { ParserAdapter } from "../PQLParserAdapter";

describe("ParserAdapter", () => {
    const createAdapter = () => new ParserAdapter();

    describe("buildTree", () => {
        it.each([
            {
                description: "Should build a tree from a valid PQL document",
                input: "PREDICT target"
            },
            {
                description: "Should build tree with WHERE keyword",
                input: "WHERE"
            },
            {
                description: "Should build tree with multiple keywords",
                input: "PREDICT target WHERE"
            },
            {
                description: "Should build tree with CLASSIFY keyword",
                input: "CLASSIFY"
            },
            {
                description: "Should build tree with RANK keyword",
                input: "RANK"
            },
            {
                description: "Should build tree with FORECAST keyword",
                input: "FORECAST"
            }
        ])("$description", ({ input }) => {
            const adapter = createAdapter();
            const tree = adapter.buildTree(input);

            expect(tree).toBeDefined();
            expect(tree.length).toBeGreaterThan(0);
        });

        it("Should handle empty document", () => {
            const adapter = createAdapter();
            const tree = adapter.buildTree("");

            expect(tree).toBeDefined();
            expect(tree.length).toBe(0);
        });
    });

    describe("startParse", () => {
        it.each([
            {
                description: "Should parse PREDICT input",
                input: "PREDICT target"
            },
            {
                description: "Should handle CLASSIFY keyword",
                input: "CLASSIFY"
            },
            {
                description: "Should handle RANK keyword",
                input: "RANK"
            },
            {
                description: "Should handle FORECAST keyword",
                input: "FORECAST"
            }
        ])("$description", ({ input }) => {
            const adapter = createAdapter();
            const result = adapter.startParse(input);

            expect(result).toBeDefined();
            expect(result.parsedPos).toBe(input.length);
            expect(result.stoppedAt).toBe(input.length);
            expect(typeof result.stopAt).toBe("function");
            expect(typeof result.advance).toBe("function");

            const tree = result.advance();
            expect(tree).toBeDefined();
            if (tree) {
                expect(tree.length).toBeGreaterThan(0);
            }
        });

        it("Should handle empty string input", () => {
            const adapter = createAdapter();
            const result = adapter.startParse("");

            expect(result).toBeDefined();
            expect(result.parsedPos).toBe(0);
            expect(result.stoppedAt).toBe(0);

            const tree = result.advance();
            expect(tree).toBeDefined();
            if (tree) {
                expect(tree.length).toBe(0);
            }
        });

        it("Should parse Input object with read method", () => {
            const adapter = createAdapter();
            const inputText = "WHERE id = 123";
            const mockInput = {
                length: inputText.length,
                read: (from: number, to: number) =>
                    inputText.substring(from, to),
                chunk: () => "",
                lineChunks: false
            };

            const result = adapter.startParse(mockInput);

            expect(result).toBeDefined();
            expect(result.parsedPos).toBe(inputText.length);
            expect(result.stoppedAt).toBe(inputText.length);

            const tree = result.advance();
            expect(tree).toBeDefined();
            if (tree) {
                expect(tree.length).toBeGreaterThan(0);
            }
        });
    });

    describe("createParse", () => {
        it("Should delegate to startParse correctly", () => {
            const adapter = createAdapter();
            const inputText = "PREDICT";
            const mockInput = {
                length: inputText.length,
                read: (from: number, to: number) =>
                    inputText.substring(from, to),
                chunk: () => "",
                lineChunks: false
            };
            const fragments: TreeFragment[] = [];
            const ranges = [{ from: 0, to: inputText.length }];

            const result = adapter.createParse(mockInput, fragments, ranges);

            expect(result).toBeDefined();
            expect(result.parsedPos).toBe(inputText.length);
            expect(result.stoppedAt).toBe(inputText.length);

            const tree = result.advance();
            expect(tree).toBeDefined();
            if (tree) {
                expect(tree.length).toBeGreaterThan(0);
            }
        });
    });

    describe("DEFAULT_NODE_GROUP_SIZE", () => {
        it("Should have correct default value", () => {
            const adapter = createAdapter();
            expect(adapter.DEFAULT_NODE_GROUP_SIZE).toBe(4);
        });
    });
});
