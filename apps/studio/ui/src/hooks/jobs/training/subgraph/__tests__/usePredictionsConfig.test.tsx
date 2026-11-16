// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from "@testing-library/react";
import type { HistoricalItems } from "gen/v0/types";
import { describe, expect, it } from "vitest";

import type { LinkPredictionEntityOutput } from "types/explorer";
import { TaskType } from "types/predictiveQueries";

import { type LhsNodeInfo } from "../subgraphNodeTypeBuild";
import { usePredictionsConfig } from "../usePredictionsConfig";

const baseEntityOutput = {
    prediction: [
        { id: "p1", id_name: "product_pk", cols: [{ name: "price" }] },
        { id: "p2", id_name: "product_pk", cols: [] }
    ],
    target: [{ id: "p2", cols: [] }]
} as unknown as LinkPredictionEntityOutput;

const baseHistoricalItems = {
    nodes: [{ id: "p1", cols: [] }]
} as unknown as HistoricalItems;

const baseProps = {
    taskType: TaskType.TEMPORAL_LINK_PREDICTION,
    entityOutput: baseEntityOutput,
    historicalItems: baseHistoricalItems,
    targetTableName: "products",
    targetColor: "var(--color-cyan-600)",
    hoveredId: null,
    selectedId: null,
    entityNodeId: "node_customer"
};

describe("usePredictionsConfig", () => {
    describe("returns undefined when", () => {
        it("task type is not link-prediction (e.g. regression)", () => {
            const { result } = renderHook(() =>
                usePredictionsConfig({
                    ...baseProps,
                    taskType: TaskType.REGRESSION
                })
            );
            expect(result.current).toBeUndefined();
        });

        it("task type is undefined", () => {
            const { result } = renderHook(() =>
                usePredictionsConfig({ ...baseProps, taskType: undefined })
            );
            expect(result.current).toBeUndefined();
        });

        it("entityOutput is null", () => {
            const { result } = renderHook(() =>
                usePredictionsConfig({ ...baseProps, entityOutput: null })
            );
            expect(result.current).toBeUndefined();
        });

        it("entityOutput.prediction is empty", () => {
            const { result } = renderHook(() =>
                usePredictionsConfig({
                    ...baseProps,
                    entityOutput: {
                        ...baseEntityOutput,
                        prediction: []
                    } as unknown as LinkPredictionEntityOutput
                })
            );
            expect(result.current).toBeUndefined();
        });

        it("targetTableName is missing", () => {
            const { result } = renderHook(() =>
                usePredictionsConfig({
                    ...baseProps,
                    targetTableName: undefined
                })
            );
            expect(result.current).toBeUndefined();
        });

        it("targetColor is missing", () => {
            const { result } = renderHook(() =>
                usePredictionsConfig({ ...baseProps, targetColor: undefined })
            );
            expect(result.current).toBeUndefined();
        });

        it("entityNodeId is missing (items built but config returns undefined)", () => {
            const { result } = renderHook(() =>
                usePredictionsConfig({ ...baseProps, entityNodeId: undefined })
            );
            expect(result.current).toBeUndefined();
        });
    });

    describe("supports all link-prediction task type variants", () => {
        it.each([
            TaskType.TEMPORAL_LINK_PREDICTION,
            TaskType.STATIC_LINK_PREDICTION,
            // Deprecated bare LINK_PREDICTION — legacy tasks may still report it.
            TaskType.LINK_PREDICTION
        ])("renders panel config for %s", (taskType) => {
            const { result } = renderHook(() =>
                usePredictionsConfig({ ...baseProps, taskType })
            );
            expect(result.current).toBeDefined();
            expect(result.current?.items).toHaveLength(2);
        });
    });

    describe("items mapping", () => {
        it("assigns rank by array index (1-based)", () => {
            const { result } = renderHook(() =>
                usePredictionsConfig(baseProps)
            );
            expect(result.current?.items[0].rank).toBe(1);
            expect(result.current?.items[1].rank).toBe(2);
        });

        it("classifies node types via deriveNodeType using history + GT intersections", () => {
            // p1 in history + prediction → recurring (no GT)
            // p2 in prediction + GT (not in history) → hit
            const { result } = renderHook(() =>
                usePredictionsConfig(baseProps)
            );
            expect(result.current?.items[0].nodeType).toBe("recurring");
            expect(result.current?.items[1].nodeType).toBe("hit");
        });

        it("passes through tableName, color, idName, cols from prediction", () => {
            const { result } = renderHook(() =>
                usePredictionsConfig(baseProps)
            );
            const first = result.current?.items[0];
            expect(first?.tableName).toBe("products");
            expect(first?.color).toBe("var(--color-cyan-600)");
            expect(first?.idName).toBe("product_pk");
            expect(first?.cols).toEqual([{ name: "price" }]);
        });

        it("falls back to pkey when no LHS counterpart in pkeyToLhsNodeInfo", () => {
            const { result } = renderHook(() =>
                usePredictionsConfig(baseProps)
            );
            expect(result.current?.items[0].id).toBe("p1");
            expect(result.current?.items[0].hopLevel).toBeUndefined();
            expect(result.current?.items[0].halfSize).toBeUndefined();
        });

        it("merges LHS counterpart's id/hopLevel/halfSize when present", () => {
            const pkeyToLhsNodeInfo = new Map<string, LhsNodeInfo>([
                ["p1", { id: "node_p1_on_lhs", hopLevel: 2, halfSize: 5 }]
            ]);
            const { result } = renderHook(() =>
                usePredictionsConfig({ ...baseProps, pkeyToLhsNodeInfo })
            );
            const first = result.current?.items[0];
            expect(first?.id).toBe("node_p1_on_lhs");
            expect(first?.hopLevel).toBe(2);
            expect(first?.halfSize).toBe(5);
        });
    });

    describe("config wiring", () => {
        it("forwards entityNodeId, hoveredId, selectedId to the returned config", () => {
            const { result } = renderHook(() =>
                usePredictionsConfig({
                    ...baseProps,
                    hoveredId: "h",
                    selectedId: "s"
                })
            );
            expect(result.current?.entityNodeId).toBe("node_customer");
            expect(result.current?.hoveredId).toBe("h");
            expect(result.current?.selectedId).toBe("s");
        });
    });
});
