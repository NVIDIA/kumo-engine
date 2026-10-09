// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Node, PredictionItem } from "@kumo-ai/kumo-ui-catalog";
import { describe, expect, it } from "vitest";

import type { GraphNode } from "types/explorer";

import {
    resolvePredictionItemClick,
    synthesizePredictionItemData
} from "../subgraphPredictionItem";

const makeNode = (
    id: string,
    tableName: string,
    idName: string,
    value: string
): GraphNode =>
    ({
        id,
        nodeId: id,
        data: {
            tableName,
            value,
            idName,
            cols: [],
            totalScore: 0
        }
    }) as unknown as GraphNode;

const makePredictionItem = (
    overrides: Partial<PredictionItem> = {}
): PredictionItem =>
    ({
        id: "p1",
        rank: 1,
        tableName: "products",
        nodeType: "exploratory",
        color: "var(--color-cyan-600)",
        ...overrides
    }) as PredictionItem;

describe("synthesizePredictionItemData", () => {
    it("uses item.idName when set", () => {
        const item = makePredictionItem({ idName: "product_pkey" });
        const data = synthesizePredictionItemData(item, []);
        expect(data.idName).toBe("product_pkey");
    });

    it("falls back to same-table LHS node's idName when item.idName is unset", () => {
        const item = makePredictionItem({ idName: undefined });
        const lhsNode = makeNode(
            "n1",
            "products",
            "borrowed_pkey",
            "other-pkey"
        );
        const data = synthesizePredictionItemData(item, [lhsNode]);
        expect(data.idName).toBe("borrowed_pkey");
    });

    it("strips schema prefix when matching LHS node table (suffix match)", () => {
        // Backend may ship schema-prefixed table names; UI normalizes by suffix.
        const item = makePredictionItem({
            tableName: "schema_a.products",
            idName: undefined
        });
        const lhsNode = makeNode(
            "n1",
            "schema_b.products",
            "borrowed_pkey",
            "other"
        );
        const data = synthesizePredictionItemData(item, [lhsNode]);
        expect(data.idName).toBe("borrowed_pkey");
    });

    it("defaults idName to 'id' when no item.idName and no same-table LHS node", () => {
        const item = makePredictionItem({ idName: undefined });
        const lhsNode = makeNode("n1", "orders", "order_pkey", "o1");
        const data = synthesizePredictionItemData(item, [lhsNode]);
        expect(data.idName).toBe("id");
    });

    it("uses item.cols when present, empty array otherwise", () => {
        const withCols = makePredictionItem({
            cols: [
                {
                    name: "price",
                    value: 9.99,
                    stype: "numerical",
                    grad_score: 0.3
                }
            ]
        });
        expect(synthesizePredictionItemData(withCols, []).cols).toHaveLength(1);

        const withoutCols = makePredictionItem({ cols: undefined });
        expect(synthesizePredictionItemData(withoutCols, []).cols).toEqual([]);
    });

    it("passes through value, tableName, hopLevel, nodeType from the item", () => {
        const item = makePredictionItem({
            id: "p42",
            tableName: "products",
            hopLevel: 2,
            nodeType: "recurring_hit"
        });
        const data = synthesizePredictionItemData(item, []);
        expect(data.value).toBe("p42");
        expect(data.tableName).toBe("products");
        expect(data.hopLevel).toBe(2);
        expect(data.nodeType).toBe("recurring_hit");
        expect(data.totalScore).toBe(0); // synthesized items get score 0
    });
});

describe("resolvePredictionItemClick", () => {
    const item = makePredictionItem({ id: "p1" });

    const makeClickNode = (data: object): Node =>
        ({
            id: "__prediction-item-p1",
            type: "predictionItem",
            position: { x: 0, y: 0 },
            data
        }) as unknown as Node;

    it("returns null when node has no hoverKey", () => {
        const node = makeClickNode({ item });
        expect(resolvePredictionItemClick(node, [])).toBeNull();
    });

    it("returns LHS counterpart's data when present (preferred over synthesize)", () => {
        const lhsNode = makeNode("p1", "products", "real_pkey", "p1");
        const node = makeClickNode({ hoverKey: "p1", item });
        const result = resolvePredictionItemClick(node, [lhsNode]);
        expect(result?.hk).toBe("p1");
        expect(result?.data?.idName).toBe("real_pkey");
        // value/tableName come from the LHS node's data, not the item.
        expect(result?.data?.tableName).toBe("products");
    });

    it("synthesizes data from item when no LHS counterpart exists", () => {
        const node = makeClickNode({ hoverKey: "rhs-only-key", item });
        const result = resolvePredictionItemClick(node, []);
        expect(result?.hk).toBe("rhs-only-key");
        expect(result?.data?.value).toBe("p1");
        expect(result?.data?.idName).toBe("id"); // synthesized default
    });

    it("returns hk with data=null when neither LHS counterpart nor item present", () => {
        const node = makeClickNode({ hoverKey: "p1" });
        const result = resolvePredictionItemClick(node, []);
        expect(result?.hk).toBe("p1");
        expect(result?.data).toBeNull();
    });
});
