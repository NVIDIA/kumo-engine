// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from "@testing-library/react";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { describe, expect, it } from "vitest";

import { TaskType } from "types/predictiveQueries";

import { useRestructureStats } from "../trainingDataStats";

const makeSplitStats = (overrides = {}) =>
    ({
        train_table_stats: { num_train_rows: 600, num_entities: 100 },
        val_table_stats: { num_train_rows: 200, num_entities: 50 },
        test_table_stats: { num_train_rows: 200, num_entities: 25 },
        ...overrides
    }) as never;

const getMetrics = (tableData: { metric: string }[]) =>
    tableData.map((row) => row.metric);

const renderWithQueryClient = <T>(hook: () => T) =>
    renderHook(hook, { wrapper: createWrapper(createQueryClient()) });

const renderStats = (taskType: TaskType, isTrainingEvaluation?: boolean) => {
    const stats = makeSplitStats();
    const { result } = renderWithQueryClient(() =>
        useRestructureStats(taskType, stats, isTrainingEvaluation)
    );
    return { result, stats };
};

describe("useRestructureStats", () => {
    it("Should return empty tableData and null allStats when splitStats is null", () => {
        const { result } = renderWithQueryClient(() =>
            useRestructureStats(TaskType.BINARY_CLASSIFICATION, null)
        );

        expect(result.current.tableData).toEqual([]);
        expect(result.current.allStats).toBeNull();
    });

    it("Should set allStats to the provided splitStats", () => {
        const { result, stats } = renderStats(TaskType.BINARY_CLASSIFICATION);

        expect(result.current.allStats).toBe(stats);
    });

    it("Should return 3 basic rows with correct values when isTrainingEvaluation is true", () => {
        const { result } = renderStats(TaskType.BINARY_CLASSIFICATION, true);
        const { tableData } = result.current;

        expect(getMetrics(tableData)).toEqual([
            "Prediction window",
            "Split distribution",
            "Total rows"
        ]);

        const totalRow = tableData.find((r) => r.metric === "Total rows");
        const splitRow = tableData.find(
            (r) => r.metric === "Split distribution"
        );

        expect(totalRow?.training).toBe(600);
        expect(totalRow?.validation).toBe(200);
        expect(totalRow?.holdout).toBe(200);

        expect(splitRow?.training).toBe("60.00%");
        expect(splitRow?.validation).toBe("20.00%");
        expect(splitRow?.holdout).toBe("20.00%");
    });

    it("Should include Unique Entities row when isTrainingEvaluation is false", () => {
        const { result } = renderStats(TaskType.NONE, false);

        expect(getMetrics(result.current.tableData)).toContain(
            "Unique entities"
        );
    });

    it("Should include binary classification rows for BINARY_CLASSIFICATION task type", () => {
        const { result } = renderStats(TaskType.BINARY_CLASSIFICATION);

        const metrics = getMetrics(result.current.tableData);
        const expectedMetrics = ["Positive samples", "Negative samples"];
        expectedMetrics.forEach((m) => expect(metrics).toContain(m));
    });

    it.each([
        ["REGRESSION", TaskType.REGRESSION],
        ["FORECASTING", TaskType.FORECASTING]
    ])(
        "Should include regression stats rows for %s task type",
        (_, taskType) => {
            const { result } = renderStats(taskType);

            const metrics = getMetrics(result.current.tableData);
            const expectedMetrics = ["Mean", "Min", "Max", "Median"];
            expectedMetrics.forEach((m) => expect(metrics).toContain(m));
        }
    );
});
