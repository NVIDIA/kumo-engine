// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HistoricalItems, NodeOutputData } from "gen/v0/types";
import { describe, expect, it } from "vitest";

import type { GraphNode, LinkPredictionEntityOutput } from "types/explorer";

import {
    buildNodeTypeMap,
    buildPkeyToLhsNodeInfo,
    deriveNodeType,
    toIdSet
} from "../subgraphNodeTypeBuild";

const makeNode = (
    id: string,
    tableName: string,
    value: string,
    hopLevel?: number
): GraphNode =>
    ({
        id,
        nodeId: id,
        data: {
            tableName,
            value,
            hopLevel,
            idName: "id",
            cols: [],
            totalScore: 0
        }
    }) as unknown as GraphNode;

describe("deriveNodeType", () => {
    // 5 categories from the issue's #28750 spec, indexed by (history, predictions, GT).
    it.each([
        { args: [true, false, false], expected: "organic" },
        { args: [true, true, false], expected: "recurring" },
        { args: [false, true, false], expected: "exploratory" },
        { args: [false, true, true], expected: "hit" },
        { args: [true, true, true], expected: "recurring_hit" }
    ])("returns $expected for $args", ({ args, expected }) => {
        expect(deriveNodeType(args[0], args[1], args[2])).toBe(expected);
    });

    it("returns 'organic' for any non-prediction combination", () => {
        // inHistory only and inGT-only (no prediction) both fall back to organic.
        expect(deriveNodeType(true, false, true)).toBe("organic");
        expect(deriveNodeType(false, false, true)).toBe("organic");
        expect(deriveNodeType(false, false, false)).toBe("organic");
    });
});

describe("toIdSet", () => {
    it("returns empty Set for undefined input", () => {
        expect(toIdSet(undefined).size).toBe(0);
    });

    it("returns empty Set for non-array input (runtime guard)", () => {
        // Defensive against task-type mismatch where caller cast a scalar.
        expect(toIdSet(42 as unknown as NodeOutputData[]).size).toBe(0);
        expect(toIdSet("abc" as unknown as NodeOutputData[]).size).toBe(0);
        expect(toIdSet({} as unknown as NodeOutputData[]).size).toBe(0);
    });

    it("coerces ids to strings and dedupes", () => {
        const set = toIdSet([
            { id: 1, cols: [] },
            { id: "1", cols: [] },
            { id: 2, cols: [] }
        ] as NodeOutputData[]);
        expect(set.size).toBe(2);
        expect(set.has("1")).toBe(true);
        expect(set.has("2")).toBe(true);
    });
});

describe("buildPkeyToLhsNodeInfo", () => {
    it("returns empty Map when targetTableName is undefined", () => {
        const result = buildPkeyToLhsNodeInfo({
            nodeArr: [makeNode("n1", "products", "p1")],
            targetTableName: undefined
        });
        expect(result.size).toBe(0);
    });

    it("only includes nodes whose tableName matches targetTableName", () => {
        const result = buildPkeyToLhsNodeInfo({
            nodeArr: [
                makeNode("n1", "products", "p1", 2),
                makeNode("n2", "orders", "o1", 1),
                makeNode("n3", "products", "p2", 2)
            ],
            targetTableName: "products"
        });
        expect(result.size).toBe(2);
        expect(result.get("p1")).toEqual({
            id: "n1",
            hopLevel: 2,
            halfSize: undefined
        });
        expect(result.get("p2")?.id).toBe("n3");
    });

    it("propagates halfSize from halfSizeByNodeId when provided", () => {
        const halfSizeMap = new Map([["n1", 5]]);
        const result = buildPkeyToLhsNodeInfo({
            nodeArr: [
                makeNode("n1", "products", "p1", 2),
                makeNode("n2", "products", "p2", 2)
            ],
            targetTableName: "products",
            halfSizeByNodeId: halfSizeMap
        });
        expect(result.get("p1")?.halfSize).toBe(5);
        // n2 wasn't in the halfSize map → undefined.
        expect(result.get("p2")?.halfSize).toBeUndefined();
    });

    it("defaults hopLevel to 0 when undefined on the node", () => {
        const result = buildPkeyToLhsNodeInfo({
            nodeArr: [makeNode("n1", "products", "p1")],
            targetTableName: "products"
        });
        expect(result.get("p1")?.hopLevel).toBe(0);
    });
});

describe("buildNodeTypeMap", () => {
    const entityOutput = {
        prediction: [
            { id: "p1", cols: [] },
            { id: "p2", cols: [] }
        ],
        target: [{ id: "p2", cols: [] }]
    } as unknown as LinkPredictionEntityOutput;

    const historicalItems = {
        nodes: [
            { id: "p1", cols: [] },
            { id: "p3", cols: [] }
        ]
    } as unknown as HistoricalItems;

    it("returns empty Map when targetTableName is undefined", () => {
        const result = buildNodeTypeMap({
            nodeArr: [makeNode("n1", "products", "p1")],
            entityOutput,
            historicalItems,
            targetTableName: undefined
        });
        expect(result.size).toBe(0);
    });

    it("skips nodes whose table doesn't match targetTableName", () => {
        const result = buildNodeTypeMap({
            nodeArr: [
                makeNode("n1", "orders", "p1"), // wrong table — skipped
                makeNode("n2", "products", "p1") // matches
            ],
            entityOutput,
            historicalItems,
            targetTableName: "products"
        });
        expect(result.size).toBe(1);
        expect(result.has("n1")).toBe(false);
        expect(result.has("n2")).toBe(true);
    });

    it("classifies each node by membership across the 3 sets", () => {
        // p1 in history + prediction → recurring
        // p2 in prediction + GT → hit
        // p3 in history only → not added (not in predictions, so the loop's
        //   "skip if no membership" guard does NOT trigger since history hits)
        //   Actually p3 IS in history, so inHistory=true → deriveNodeType returns
        //   organic. But p3 isn't in nodeArr here.
        const result = buildNodeTypeMap({
            nodeArr: [
                makeNode("n1", "products", "p1"),
                makeNode("n2", "products", "p2")
            ],
            entityOutput,
            historicalItems,
            targetTableName: "products"
        });
        expect(result.get("n1")).toBe("recurring");
        expect(result.get("n2")).toBe("hit");
    });

    it("skips nodes with no membership in any of the 3 sets", () => {
        const result = buildNodeTypeMap({
            nodeArr: [makeNode("n_unrelated", "products", "p_unknown")],
            entityOutput,
            historicalItems,
            targetTableName: "products"
        });
        expect(result.size).toBe(0);
    });

    it("handles null entityOutput and historicalItems gracefully", () => {
        const result = buildNodeTypeMap({
            nodeArr: [makeNode("n1", "products", "p1")],
            entityOutput: null,
            historicalItems: null,
            targetTableName: "products"
        });
        expect(result.size).toBe(0);
    });
});
