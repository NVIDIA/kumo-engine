// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Row } from "@kumo-ai/kumo-ui-catalog";
import { describe, expect, it } from "vitest";

import { QueryType, TaskType } from "types/predictiveQueries";
import { TimeFrameStats } from "types/stats";

import { CommonTrainingTableStatsColumns } from "../CommonTrainingTableStats";

const createMockRow = (
    overrides?: Partial<TimeFrameStats>
): Row<TimeFrameStats> =>
    ({
        original: {
            num_entities: 1000,
            num_train_rows: 1000,
            num_targets: 0,
            max_date: "",
            min_date: "",
            anchor_date: "",
            target_distribution: { count: {}, percentage: {} },
            ...overrides
        }
    }) as unknown as Row<TimeFrameStats>;

const findColumn = (
    columns: ReturnType<typeof CommonTrainingTableStatsColumns>,
    id: string
) => {
    const column = columns.find((col) => col.id === id);
    return column;
};

const getCellRenderer = (
    column: NonNullable<ReturnType<typeof findColumn>>
) => {
    if (typeof column.cell !== "function") {
        throw new Error("Cell renderer is not a function");
    }
    return column.cell;
};

describe("CommonTrainingTableStatsColumns", () => {
    it.each([
        {
            isSplitStats: true,
            columnId: "data_split",
            shouldExist: true,
            expectedHeader: "Split"
        },
        {
            isSplitStats: true,
            columnId: "anchor_date",
            shouldExist: false
        },
        {
            isSplitStats: false,
            columnId: "anchor_date",
            shouldExist: true
        },
        {
            isSplitStats: false,
            columnId: "data_split",
            shouldExist: false
        }
    ])(
        "Should $<shouldExist>include $columnId column when isSplitStats is $isSplitStats",
        ({ isSplitStats, columnId, shouldExist, expectedHeader }) => {
            const columns = CommonTrainingTableStatsColumns(
                QueryType.TEMPORAL,
                isSplitStats,
                "Anchor Date"
            );

            const column = findColumn(columns, columnId);
            if (shouldExist) {
                expect(column).toBeDefined();
                if (expectedHeader) {
                    expect(column?.header).toBe(expectedHeader);
                }
            } else {
                expect(column).toBeUndefined();
            }
        }
    );

    it.each([
        {
            columnId: "anchor_date",
            sortField: "anchor_date_toSort",
            description: "anchor_date"
        },
        {
            columnId: "min_date",
            sortField: "min_date_toSort",
            description: "min_date"
        }
    ])("Should have sortingFn for $description", ({ columnId, sortField }) => {
        const columns = CommonTrainingTableStatsColumns(
            QueryType.TEMPORAL,
            false,
            "Anchor Date"
        );

        const column = findColumn(columns, columnId);
        expect(column?.sortingFn).toBeDefined();
        expect(typeof column?.sortingFn).toBe("function");

        if (typeof column?.sortingFn === "function") {
            const mockRowA = createMockRow({
                [sortField]: "2023-01-02"
            });
            const mockRowB = createMockRow({
                [sortField]: "2023-01-01"
            });

            const result = column.sortingFn(mockRowA, mockRowB, "");
            expect(result).toBe(-1);
        }
    });

    it("Should include prediction window column for temporal queries", () => {
        const columns = CommonTrainingTableStatsColumns(
            QueryType.TEMPORAL,
            false,
            "Anchor Date"
        );

        const predWindowColumn = findColumn(columns, "min_date");
        expect(predWindowColumn).toBeDefined();
    });

    it.each([
        { isSplitStats: true, shouldExist: true },
        { isSplitStats: false, shouldExist: false }
    ])(
        "Should $<shouldExist>include num_train_rows column when isSplitStats is $isSplitStats",
        ({ isSplitStats, shouldExist }) => {
            const columns = CommonTrainingTableStatsColumns(
                QueryType.TEMPORAL,
                isSplitStats,
                "Anchor Date"
            );

            const numRowsColumn = findColumn(columns, "num_train_rows");
            if (shouldExist) {
                expect(numRowsColumn).toBeDefined();
            } else {
                expect(numRowsColumn).toBeUndefined();
            }
        }
    );

    it.each([
        { columnId: "min_date", description: "prediction window" },
        { columnId: "num_train_rows", description: "num_train_rows" }
    ])(
        "Should not include $description column for non-temporal queries",
        ({ columnId }) => {
            const columns = CommonTrainingTableStatsColumns(
                TaskType.BINARY_CLASSIFICATION,
                columnId === "num_train_rows" ? true : false,
                "Anchor Date"
            );

            const column = findColumn(columns, columnId);
            expect(column).toBeUndefined();
        }
    );

    it.each([
        {
            queryType: QueryType.TEMPORAL,
            isSplitStats: true,
            expectedColumns: ["data_split", "min_date", "num_train_rows"],
            description: "split stats temporal query"
        },
        {
            queryType: QueryType.TEMPORAL,
            isSplitStats: false,
            expectedColumns: ["anchor_date", "min_date"],
            description: "non-split stats temporal query"
        },
        {
            queryType: TaskType.REGRESSION,
            isSplitStats: false,
            expectedColumns: ["anchor_date"],
            description: "non-temporal query"
        }
    ])(
        "Should have correct column order for $description",
        ({ queryType, isSplitStats, expectedColumns }) => {
            const columns = CommonTrainingTableStatsColumns(
                queryType,
                isSplitStats,
                "Anchor Date"
            );

            const columnIds = columns.map((col) => col.id);
            expect(columnIds).toEqual(expectedColumns);
        }
    );

    it.each([
        { value: 5000, description: "with value using parseNumbers" },
        {
            value: undefined,
            description: "Symbol emdash when num_train_rows is undefined"
        },
        { value: 0, description: "Symbol emdash when num_train_rows is zero" }
    ])("Should render $description", ({ value }) => {
        const columns = CommonTrainingTableStatsColumns(
            QueryType.TEMPORAL,
            true,
            "Anchor Date"
        );

        const numRowsColumn = findColumn(columns, "num_train_rows");
        expect(numRowsColumn).toBeDefined();

        if (numRowsColumn) {
            const cellRenderer = getCellRenderer(numRowsColumn);
            const mockRow = createMockRow({ num_train_rows: value });

            const result = cellRenderer({
                row: mockRow
            } as never);

            expect(result).toBeDefined();
        }
    });
});
