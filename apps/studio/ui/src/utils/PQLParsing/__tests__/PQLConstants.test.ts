// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { parserAdapterNodeSet, tokenToNodeType } from "../PQLConstants";

describe("PQLConstants", () => {
    describe("tokenToNodeType", () => {
        it.each([
            { token: "PREDICT", expectedName: "PREDICT" },
            { token: "WHERE", expectedName: "WHERE" },
            { token: "CLASSIFY", expectedName: "CLASSIFY" },
            { token: "RANK", expectedName: "RANK" },
            { token: "FOR_EACH", expectedName: "FOR_EACH" },
            { token: "ASSUMING", expectedName: "ASSUMING" },
            { token: "FORECAST", expectedName: "FORECAST" },
            { token: "TIMEFRAMES", expectedName: "TIMEFRAMES" }
        ])("Should have $token keyword node", ({ token, expectedName }) => {
            expect(
                tokenToNodeType[token as keyof typeof tokenToNodeType]
            ).toBeDefined();
            expect(
                tokenToNodeType[token as keyof typeof tokenToNodeType].name
            ).toBe(expectedName);
        });

        it.each([
            { token: "AND", expectedName: "AND" },
            { token: "OR", expectedName: "OR" },
            { token: "NOT", expectedName: "NOT" }
        ])("Should have $token operator node", ({ token, expectedName }) => {
            expect(
                tokenToNodeType[token as keyof typeof tokenToNodeType]
            ).toBeDefined();
            expect(
                tokenToNodeType[token as keyof typeof tokenToNodeType].name
            ).toBe(expectedName);
        });

        it.each([
            { token: "INT", expectedName: "INT" },
            { token: "DECIMAL", expectedName: "DECIMAL" },
            { token: "STR", expectedName: "STR" },
            { token: "ID", expectedName: "ID" },
            { token: "BOOL", expectedName: "BOOL" }
        ])("Should have $token data type node", ({ token, expectedName }) => {
            expect(
                tokenToNodeType[token as keyof typeof tokenToNodeType]
            ).toBeDefined();
            expect(
                tokenToNodeType[token as keyof typeof tokenToNodeType].name
            ).toBe(expectedName);
        });

        it("Should have all nodes defined with unique IDs and correct names", () => {
            const ids = Object.values(tokenToNodeType).map((node) => node.id);
            const uniqueIds = new Set(ids);

            expect(ids.length).toBe(uniqueIds.size);

            Object.entries(tokenToNodeType).forEach(([key, value]) => {
                expect(value).toBeDefined();
                expect(value.name).toBe(key);
            });
        });
    });

    describe("parserAdapterNodeSet", () => {
        it("Should create NodeSet with all node types", () => {
            expect(parserAdapterNodeSet).toBeDefined();
        });

        it("Should contain all node types from tokenToNodeType", () => {
            const nodeCount = Object.keys(tokenToNodeType).length;
            expect(parserAdapterNodeSet.types.length).toBe(nodeCount);
        });

        it("Should have unique IDs for all nodes in NodeSet", () => {
            const ids = parserAdapterNodeSet.types.map((node) => node.id);
            const uniqueIds = new Set(ids);

            expect(ids.length).toBe(uniqueIds.size);
            expect(ids.length).toBeGreaterThan(0);
        });

        it("Should include document node at correct position", () => {
            const documentNode = parserAdapterNodeSet.types.find(
                (node) => node.name === "document"
            );

            expect(documentNode).toBeDefined();
            expect(documentNode?.isTop).toBe(true);
        });
    });
});
