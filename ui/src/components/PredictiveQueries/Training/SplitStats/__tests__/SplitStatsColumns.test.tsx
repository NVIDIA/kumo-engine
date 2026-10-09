// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext } from "@kumo-ai/kumo-ui-catalog";
import { describe, expect, it } from "vitest";

import { TransposedRow } from "components/PredictiveQueries/Training/SplitStats/trainingDataStats";

import { SplitStatsColumns } from "../SplitStatsColumns";

type ColumnDefinition = ReturnType<typeof SplitStatsColumns>[number];

const createMockRow = (
    overrides?: Partial<TransposedRow>
): { original: TransposedRow } => ({
    original: {
        metric: "Total Rows",
        training: 1000,
        validation: 500,
        holdout: 300,
        ...overrides
    }
});

const getColumn = (columns: ColumnDefinition[], index: number) => {
    const column = columns[index];
    if (!column) {
        throw new Error(`Column at index ${index} not found`);
    }
    return column;
};

const renderCell = (
    column: ColumnDefinition,
    row: { original: TransposedRow }
) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    return column.cell({ row } as CellContext<TransposedRow, unknown>);
};

const renderHeader = (column: ColumnDefinition) => {
    if (typeof column?.header !== "function") {
        throw new Error("Column header is not a function");
    }
    return column.header({} as never);
};

describe("SplitStatsColumns", () => {
    describe("Metric Column", () => {
        it("Should have correct header", () => {
            const columns = SplitStatsColumns({ windowWidth: 1920 });
            const metricColumn = getColumn(columns, 0);

            const header = renderHeader(metricColumn);

            expect(header.type).toBe("span");
            expect(header.props.className).toBe(
                "text-title-secondary text-base"
            );
            expect(header.props.children).toBe("Data Statistics");
        });

        it.each([
            { metricName: "Prediction Window" },
            { metricName: "Split Distribution" },
            { metricName: "Total Rows" },
            { metricName: "Unique Entities" },
            { metricName: "Some Metric" }
        ])("Should pass $metricName to MetricCell", ({ metricName }) => {
            const columns = SplitStatsColumns({ windowWidth: 1920 });
            const metricColumn = getColumn(columns, 0);
            const mockRow = createMockRow({ metric: metricName });

            const cell = renderCell(metricColumn, mockRow);

            expect(cell).toBeDefined();
            expect(cell.props.metricName).toBe(metricName);
        });
    });

    describe.each([
        {
            name: "Holdout",
            index: 1,
            header: "Holdout",
            className: "text-pink-600 text-base",
            field: "holdout" as keyof TransposedRow,
            numericValue: 300,
            stringValue: "20%"
        },
        {
            name: "Validation",
            index: 2,
            header: "Validation",
            className: "text-title-secondary text-base",
            field: "validation" as keyof TransposedRow,
            numericValue: 500,
            stringValue: "30%"
        },
        {
            name: "Training",
            index: 3,
            header: "Training",
            className: "text-title-secondary text-base",
            field: "training" as keyof TransposedRow,
            numericValue: 1000,
            stringValue: "50%"
        }
    ])(
        "$name Column",
        ({ index, header, className, field, numericValue, stringValue }) => {
            it("Should have correct header", () => {
                const columns = SplitStatsColumns({ windowWidth: 1920 });
                const column = getColumn(columns, index);

                const headerResult = renderHeader(column);

                expect(headerResult.type).toBe("span");
                expect(headerResult.props.className).toBe(className);
                expect(headerResult.props.children).toBe(header);
            });

            it("Should render numeric value", () => {
                const columns = SplitStatsColumns({ windowWidth: 1920 });
                const column = getColumn(columns, index);
                const mockRow = createMockRow({ [field]: numericValue });

                const cell = renderCell(column, mockRow);

                expect(cell.type).toBe("div");
                expect(cell.props.className).toBe(
                    "text-title-primary text-base"
                );
                expect(cell.props.children).toBe(numericValue);
            });

            it("Should render string value", () => {
                const columns = SplitStatsColumns({ windowWidth: 1920 });
                const column = getColumn(columns, index);
                const mockRow = createMockRow({ [field]: stringValue });

                const cell = renderCell(column, mockRow);

                expect(cell.props.children).toBe(stringValue);
            });
        }
    );

    describe("Return Value", () => {
        it("Should return array with 4 columns", () => {
            const columns = SplitStatsColumns({ windowWidth: 1920 });

            expect(Array.isArray(columns)).toBe(true);
            expect(columns).toHaveLength(4);
        });

        it("Should return consistent results on multiple calls", () => {
            const columns1 = SplitStatsColumns({ windowWidth: 1920 });
            const columns2 = SplitStatsColumns({ windowWidth: 1920 });

            expect(columns1).toHaveLength(columns2.length);
        });
    });
});
