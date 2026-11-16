// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { MultiClassificationPredictionColumns } from "../MultiClassificationPrediction";

type ColumnDefinition = ReturnType<
    typeof MultiClassificationPredictionColumns
>[number];

const getColumn = (columns: ColumnDefinition[], id: string) => {
    const column = columns.find((col) => col.id === id);
    if (!column) {
        throw new Error(`Column with id "${id}" not found`);
    }
    return column;
};

describe("MultiClassificationPredictionColumns", () => {
    describe("Column Configuration", () => {
        it("Should generate two columns", () => {
            const columns = MultiClassificationPredictionColumns();

            expect(columns).toHaveLength(2);
        });

        it("Should have label and value column ids", () => {
            const columns = MultiClassificationPredictionColumns();

            const columnIds = columns.map((col) => col.id);
            expect(columnIds).toContain("label");
            expect(columnIds).toContain("value");
        });
    });

    describe("Label Column", () => {
        it("Should have correct configuration for label column", () => {
            const columns = MultiClassificationPredictionColumns();
            const labelColumn = getColumn(columns, "label");

            expect(labelColumn.id).toBe("label");
            expect(labelColumn.header).toBe("Label");
        });

        it("Should be the first column", () => {
            const columns = MultiClassificationPredictionColumns();

            expect(columns[0].id).toBe("label");
        });
    });

    describe("Value Column", () => {
        it("Should have correct configuration for value column", () => {
            const columns = MultiClassificationPredictionColumns();
            const valueColumn = getColumn(columns, "value");

            expect(valueColumn.id).toBe("value");
            expect(valueColumn.header).toBe("Value");
        });

        it("Should be the second column", () => {
            const columns = MultiClassificationPredictionColumns();

            expect(columns[1].id).toBe("value");
        });
    });

    describe("Return Value", () => {
        it("Should return an array", () => {
            const columns = MultiClassificationPredictionColumns();

            expect(Array.isArray(columns)).toBe(true);
        });

        it("Should return consistent results on multiple calls", () => {
            const columns1 = MultiClassificationPredictionColumns();
            const columns2 = MultiClassificationPredictionColumns();

            expect(columns1).toHaveLength(columns2.length);
            expect(columns1[0].id).toBe(columns2[0].id);
            expect(columns1[1].id).toBe(columns2[1].id);
        });
    });
});
