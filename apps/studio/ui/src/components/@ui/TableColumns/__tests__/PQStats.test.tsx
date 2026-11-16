// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row } from "@kumo-ai/kumo-ui-catalog";
import { describe, expect, it, vi } from "vitest";

import { QueryType } from "types/predictiveQueries";
import { TimeFrameStats } from "types/stats";

import { PQStatsColumns } from "../PQStats";

type ColumnDefinition = ReturnType<typeof PQStatsColumns>[number];

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
        },
        getIsExpanded: () => false,
        toggleExpanded: vi.fn()
    }) as unknown as Row<TimeFrameStats>;

const renderCell = (column: ColumnDefinition, row: Row<TimeFrameStats>) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    return column.cell({ row } as CellContext<TimeFrameStats, number>);
};

const getColumn = (columns: ColumnDefinition[], id: string) => {
    const column = columns.find((col) => col.id === id);
    if (!column) {
        throw new Error(`Column with id "${id}" not found`);
    }
    return column;
};

const getButton = (cellContent: JSX.Element) => {
    const tooltip = cellContent.props.children;
    const tooltipTrigger = Array.isArray(tooltip.props.children)
        ? tooltip.props.children[0]
        : tooltip.props.children;
    const span = tooltipTrigger.props.children;
    return span.props.children;
};

describe("PQStatsColumns", () => {
    describe("Column Configuration", () => {
        it("Should generate columns with num_entities and expand", () => {
            const columns = PQStatsColumns(false, false, QueryType.TEMPORAL);

            const columnIds = columns.map((col) => col.id);
            expect(columnIds).toContain("num_entities");
            expect(columnIds).toContain("expand");
        });

        it("Should configure expand column with correct header and size", () => {
            const columns = PQStatsColumns(true, false, QueryType.TEMPORAL);

            const expandColumn = getColumn(columns, "expand");
            expect(expandColumn.header).toBe("Target Distribution");
            expect(expandColumn.size).toBe(150);
        });

        it.each([
            {
                isSplitStats: true,
                expectedColumns: ["data_split", "min_date", "num_train_rows"]
            },
            {
                isSplitStats: false,
                expectedColumns: ["anchor_date", "min_date"]
            }
        ])(
            "Should include correct columns when isSplitStats is $isSplitStats",
            ({ isSplitStats, expectedColumns }) => {
                const columns = PQStatsColumns(
                    false,
                    isSplitStats,
                    QueryType.TEMPORAL
                );
                const columnIds = columns.map((col) => col.id);

                expectedColumns.forEach((colId) => {
                    expect(columnIds).toContain(colId);
                });
            }
        );
    });

    describe("Num Entities Column", () => {
        const columns = PQStatsColumns(false, false, QueryType.TEMPORAL);
        const entitiesColumn = getColumn(columns, "num_entities");

        it("Should have num_entities id", () => {
            expect(entitiesColumn.id).toBe("num_entities");
        });

        it("Should render header as function", () => {
            expect(typeof entitiesColumn.header).toBe("function");
        });

        it.each([{ value: 12345 }, { value: 9999999 }, { value: 0 }])(
            "Should render cell with num_entities=$value",
            ({ value }) => {
                const mockRow = createMockRow({ num_entities: value });
                const cellValue = renderCell(entitiesColumn, mockRow);

                expect(cellValue).toBeDefined();
            }
        );
    });

    describe("Expand Column Button", () => {
        it.each([
            { isExpanded: false, expectedText: "See Details" },
            { isExpanded: true, expectedText: "Close Details" }
        ])(
            "Should render '$expectedText' when isExpanded=$isExpanded",
            ({ isExpanded, expectedText }) => {
                const columns = PQStatsColumns(true, false, QueryType.TEMPORAL);
                const expandColumn = getColumn(columns, "expand");
                const mockRow = {
                    original: createMockRow().original,
                    getIsExpanded: () => isExpanded,
                    toggleExpanded: vi.fn()
                } as unknown as Row<TimeFrameStats>;

                const cellContent = renderCell(expandColumn, mockRow);
                const button = getButton(cellContent);

                expect(button.props.children).toBe(expectedText);
            }
        );

        it.each([
            { showDistribution: true, expectedDisabled: false },
            { showDistribution: false, expectedDisabled: true }
        ])(
            "Should set disabled=$expectedDisabled when showDistribution=$showDistribution",
            ({ showDistribution, expectedDisabled }) => {
                const columns = PQStatsColumns(
                    showDistribution,
                    false,
                    QueryType.TEMPORAL
                );
                const expandColumn = getColumn(columns, "expand");
                const mockRow = createMockRow();

                const cellContent = renderCell(expandColumn, mockRow);
                const button = getButton(cellContent);

                expect(button.props.disabled).toBe(expectedDisabled);
            }
        );

        it("Should call toggleExpanded and stopPropagation on click", () => {
            const columns = PQStatsColumns(true, false, QueryType.TEMPORAL);
            const expandColumn = getColumn(columns, "expand");
            const toggleExpandedMock = vi.fn();
            const mockRow = {
                original: createMockRow().original,
                getIsExpanded: () => false,
                toggleExpanded: toggleExpandedMock
            } as unknown as Row<TimeFrameStats>;

            const cellContent = renderCell(expandColumn, mockRow);
            const button = getButton(cellContent);
            const mockEvent = {
                stopPropagation: vi.fn()
            } as unknown as React.MouseEvent<HTMLButtonElement>;

            button.props.onClick(mockEvent);

            expect(mockEvent.stopPropagation).toHaveBeenCalled();
            expect(toggleExpandedMock).toHaveBeenCalled();
        });
    });
});
