// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { PQLGrammarLexer } from "../grammar/PQLGrammarLexer";
import { PQLServer } from "../PQLServer";

describe("PQLServer", () => {
    const createServer = () => new PQLServer();

    describe("getTokenStream", () => {
        it.each([
            {
                description: "Should tokenize PREDICT keyword",
                input: "PREDICT",
                expectedTokenText: "PREDICT"
            },
            {
                description: "Should tokenize WHERE keyword",
                input: "WHERE",
                expectedTokenText: "WHERE"
            },
            {
                description: "Should tokenize CLASSIFY keyword",
                input: "CLASSIFY",
                expectedTokenText: "CLASSIFY"
            },
            {
                description: "Should tokenize RANK keyword",
                input: "RANK",
                expectedTokenText: "RANK"
            },
            {
                description: "Should tokenize FORECAST keyword",
                input: "FORECAST",
                expectedTokenText: "FORECAST"
            },
            {
                description: "Should tokenize TIMEFRAMES keyword",
                input: "TIMEFRAMES",
                expectedTokenText: "TIMEFRAMES"
            }
        ])("$description", ({ input, expectedTokenText }) => {
            const server = createServer();
            const tokens = server.getTokenStream(input);

            expect(tokens).toBeDefined();
            expect(Array.isArray(tokens)).toBe(true);
            expect(tokens.length).toBeGreaterThan(0);

            const firstToken = tokens[0];
            expect(firstToken.text).toBe(expectedTokenText);
        });

        it("Should tokenize FOR EACH keywords", () => {
            const server = createServer();
            const tokens = server.getTokenStream("FOR EACH");

            expect(tokens).toBeDefined();
            expect(tokens.length).toBeGreaterThan(0);

            const tokenTexts = tokens.map((t) => t.text);
            expect(tokenTexts).toContain("FOR EACH");
        });

        it("Should tokenize complex query", () => {
            const server = createServer();
            const tokens = server.getTokenStream(
                "PREDICT target WHERE id = 123"
            );

            expect(tokens).toBeDefined();
            expect(tokens.length).toBeGreaterThan(5);

            const tokenTexts = tokens.map((t) => t.text);
            expect(tokenTexts).toContain("PREDICT");
            expect(tokenTexts).toContain("target");
            expect(tokenTexts).toContain("WHERE");
            expect(tokenTexts).toContain("id");
            expect(tokenTexts).toContain("=");
            expect(tokenTexts).toContain("123");
        });

        it("Should return empty array for empty string", () => {
            const server = createServer();
            const tokens = server.getTokenStream("");

            expect(tokens).toBeDefined();
            expect(Array.isArray(tokens)).toBe(true);
        });

        it("Should tokenize logical operators", () => {
            const server = createServer();
            const tokens = server.getTokenStream("AND OR NOT");

            expect(tokens).toBeDefined();
            expect(tokens.length).toBeGreaterThanOrEqual(3);

            const tokenTexts = tokens.map((t) => t.text);
            expect(tokenTexts).toContain("AND");
            expect(tokenTexts).toContain("OR");
            expect(tokenTexts).toContain("NOT");
        });

        it("Should tokenize numbers and strings", () => {
            const server = createServer();
            const tokens = server.getTokenStream("123 45.67");

            expect(tokens).toBeDefined();
            expect(tokens.length).toBeGreaterThanOrEqual(2);

            const tokenTexts = tokens.map((t) => t.text);
            expect(tokenTexts).toContain("123");
            expect(tokenTexts).toContain("45.67");
        });
    });

    describe("getTokenTypeForIndex", () => {
        it.each([
            {
                description: "Should return PREDICT for PREDICT token index",
                tokenIndex: PQLGrammarLexer.PREDICT,
                expected: "PREDICT"
            },
            {
                description: "Should return WHERE for WHERE token index",
                tokenIndex: PQLGrammarLexer.WHERE,
                expected: "WHERE"
            },
            {
                description: "Should return CLASSIFY for CLASSIFY token index",
                tokenIndex: PQLGrammarLexer.CLASSIFY,
                expected: "CLASSIFY"
            },
            {
                description: "Should return RANK for RANK token index",
                tokenIndex: PQLGrammarLexer.RANK,
                expected: "RANK"
            },
            {
                description: "Should return FORECAST for FORECAST token index",
                tokenIndex: PQLGrammarLexer.FORECAST,
                expected: "FORECAST"
            },
            {
                description:
                    "Should return TIMEFRAMES for TIMEFRAMES token index",
                tokenIndex: PQLGrammarLexer.TIMEFRAMES,
                expected: "TIMEFRAMES"
            },
            {
                description: "Should return AND for AND token index",
                tokenIndex: PQLGrammarLexer.AND,
                expected: "AND"
            },
            {
                description: "Should return OR for OR token index",
                tokenIndex: PQLGrammarLexer.OR,
                expected: "OR"
            },
            {
                description: "Should return NOT for NOT token index",
                tokenIndex: PQLGrammarLexer.NOT,
                expected: "NOT"
            },
            {
                description: "Should return INT for INT token index",
                tokenIndex: PQLGrammarLexer.INT,
                expected: "INT"
            },
            {
                description: "Should return DECIMAL for DECIMAL token index",
                tokenIndex: PQLGrammarLexer.DECIMAL,
                expected: "DECIMAL"
            },
            {
                description: "Should return ID for ID token index",
                tokenIndex: PQLGrammarLexer.ID,
                expected: "ID"
            }
        ])("$description", ({ tokenIndex, expected }) => {
            const server = createServer();
            const result = server.getTokenTypeForIndex(tokenIndex);

            expect(result).toBe(expected);
        });

        it("Should return unknown for invalid token index", () => {
            const server = createServer();
            const result = server.getTokenTypeForIndex(99999);

            expect(result).toBe("unknown");
        });

        it("Should return unknown for negative token index", () => {
            const server = createServer();
            const result = server.getTokenTypeForIndex(-1);

            expect(result).toBe("unknown");
        });
    });
});
