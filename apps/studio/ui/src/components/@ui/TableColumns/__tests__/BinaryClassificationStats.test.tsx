// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row, Symbol } from "@kumo-ai/kumo-ui-catalog";
import { describe, expect, it, vi } from "vitest";

import { QueryType } from "types/predictiveQueries";
import { TimeFrameStats } from "types/stats";

import { BinaryClassificationStatsColumns } from "../BinaryClassificationStats";

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

const createMockContext = (
    value: number | null | undefined,
    rowOverrides?: Partial<TimeFrameStats>
): CellContext<TimeFrameStats, number> =>
    ({
        getValue: () => value,
        row: createMockRow(rowOverrides)
    }) as CellContext<TimeFrameStats, number>;

const findColumn = (
    columns: ReturnType<typeof BinaryClassificationStatsColumns>,
    id: string
) => {
    const column = columns.find((col) => col.id === id);
    if (!column) {
        throw new Error(`Column with id "${id}" not found`);
    }
    return column;
};

const getCellRenderer = (column: ReturnType<typeof findColumn>) => {
    if (typeof column.cell !== "function") {
        throw new Error("Cell renderer is not a function");
    }
    return column.cell;
};

describe("BinaryClassificationStatsColumns", () => {
    it("Should generate base columns without distribution available", () => {
        const columns = BinaryClassificationStatsColumns(
            false,
            QueryType.TEMPORAL,
            false
        );

        expect(columns).toHaveLength(5);

        const columnIds = columns.map((col) => col.id);
        expect(columnIds).toContain("num_entities");
        expect(columnIds).toContain("target_distribution.count.True");
        expect(columnIds).toContain("target_distribution.count.False");
    });

    it("Should include expand column when distributionAvailable is true", () => {
        const columns = BinaryClassificationStatsColumns(
            false,
            QueryType.TEMPORAL,
            true
        );

        expect(columns).toHaveLength(6);
        expect(columns[5].id).toBe("expand");
        expect(columns[5].header).toBe("Target Distribution");
    });

    it("Should handle undefined isSplitStats parameter", () => {
        const columns = BinaryClassificationStatsColumns(
            undefined,
            QueryType.TEMPORAL,
            false
        );

        expect(columns).toBeDefined();
        expect(columns.length).toBeGreaterThan(0);
    });

    it("Should include common training table stats columns", () => {
        const columns = BinaryClassificationStatsColumns(
            true,
            QueryType.TEMPORAL,
            false
        );

        const columnIds = columns.map((col) => col.id);
        expect(columnIds).toContain("data_split");
        expect(columnIds).toContain("min_date");
        expect(columnIds).toContain("num_train_rows");
    });

    it("Should render num_entities column with correct id", () => {
        const columns = BinaryClassificationStatsColumns(
            false,
            QueryType.TEMPORAL,
            false
        );

        const entitiesColumn = findColumn(columns, "num_entities");

        expect(entitiesColumn.id).toBe("num_entities");
    });

    it("Should render num_entities value correctly", () => {
        const columns = BinaryClassificationStatsColumns(
            false,
            QueryType.TEMPORAL,
            false
        );
        const entitiesColumn = findColumn(columns, "num_entities");
        const cellRenderer = getCellRenderer(entitiesColumn);
        const mockRow = createMockRow({ num_entities: 12345 });

        const cellValue = cellRenderer({
            row: mockRow
        } as CellContext<TimeFrameStats, number>);

        expect(cellValue).toBeDefined();
    });

    it("Should handle large numbers in num_entities column", () => {
        const columns = BinaryClassificationStatsColumns(
            false,
            QueryType.TEMPORAL,
            false
        );
        const entitiesColumn = findColumn(columns, "num_entities");
        const cellRenderer = getCellRenderer(entitiesColumn);
        const mockRow = createMockRow({ num_entities: 9999999 });

        const cellValue = cellRenderer({
            row: mockRow
        } as CellContext<TimeFrameStats, number>);

        expect(cellValue).toBeDefined();
    });

    it("Should have correct header and id for positive samples column", () => {
        const columns = BinaryClassificationStatsColumns(
            false,
            QueryType.TEMPORAL,
            false
        );

        const positiveColumn = findColumn(
            columns,
            "target_distribution.count.True"
        );

        expect(positiveColumn.header).toBe("Positive samples");
        expect(positiveColumn.id).toBe("target_distribution.count.True");
    });

    it.each([
        { value: null, description: "null" },
        { value: undefined, description: "undefined" },
        {
            value: 0,
            description: "zero",
            rowOverrides: {
                target_distribution: { count: { True: 0 }, percentage: {} }
            }
        }
    ])(
        "Should render emdash in positive samples column when value is $description",
        ({ value, rowOverrides }) => {
            const columns = BinaryClassificationStatsColumns(
                false,
                QueryType.TEMPORAL,
                false
            );
            const positiveColumn = findColumn(
                columns,
                "target_distribution.count.True"
            );
            const cellRenderer = getCellRenderer(positiveColumn);
            const mockContext = createMockContext(value, rowOverrides);

            const cellValue = cellRenderer(mockContext);

            expect(cellValue.props.children.type).toBe(Symbol);
            expect(cellValue.props.children.props.type).toBe("emdash");
        }
    );

    it.each([
        {
            description: "non-split stats",
            isSplitStats: false,
            value: 300,
            rowOverrides: {
                num_entities: 1000,
                target_distribution: {
                    count: { True: 300 },
                    percentage: {}
                }
            }
        },
        {
            description: "split stats using num_train_rows",
            isSplitStats: true,
            value: 400,
            rowOverrides: {
                num_entities: 500,
                num_train_rows: 1000,
                target_distribution: {
                    count: { True: 400 },
                    percentage: {}
                }
            }
        }
    ])(
        "Should render count with percentage for positive samples in $description",
        ({ isSplitStats, value, rowOverrides }) => {
            const columns = BinaryClassificationStatsColumns(
                isSplitStats,
                QueryType.TEMPORAL,
                false
            );
            const positiveColumn = findColumn(
                columns,
                "target_distribution.count.True"
            );
            const cellRenderer = getCellRenderer(positiveColumn);
            const mockContext = createMockContext(value, rowOverrides);

            const cellValue = cellRenderer(mockContext);

            expect(cellValue).toBeDefined();
        }
    );

    it("Should calculate percentage correctly for positive samples", () => {
        const columns = BinaryClassificationStatsColumns(
            false,
            QueryType.TEMPORAL,
            false
        );
        const positiveColumn = findColumn(
            columns,
            "target_distribution.count.True"
        );
        const cellRenderer = getCellRenderer(positiveColumn);
        const mockContext = createMockContext(250, {
            num_entities: 1000,
            num_train_rows: 2000,
            target_distribution: {
                count: { True: 250 },
                percentage: {}
            }
        });

        const cellValue = cellRenderer(mockContext);
        expect(cellValue).toBeDefined();
    });

    it("Should have correct header and id for negative samples column", () => {
        const columns = BinaryClassificationStatsColumns(
            false,
            QueryType.TEMPORAL,
            false
        );

        const negativeColumn = findColumn(
            columns,
            "target_distribution.count.False"
        );

        expect(negativeColumn.header).toBe("Negative samples");
        expect(negativeColumn.id).toBe("target_distribution.count.False");
    });

    it("Should render emdash in negative samples column when value is null or undefined", () => {
        const columns = BinaryClassificationStatsColumns(
            false,
            QueryType.TEMPORAL,
            false
        );
        const negativeColumn = findColumn(
            columns,
            "target_distribution.count.False"
        );
        const cellRenderer = getCellRenderer(negativeColumn);
        const mockContext = createMockContext(undefined);

        const cellValue = cellRenderer(mockContext);

        expect(cellValue.props.children.type).toBe(Symbol);
        expect(cellValue.props.children.props.type).toBe("emdash");
    });

    it.each([
        {
            description: "non-split stats",
            isSplitStats: false,
            value: 700,
            rowOverrides: {
                num_entities: 1000,
                target_distribution: {
                    count: { False: 700 },
                    percentage: {}
                }
            }
        },
        {
            description: "split stats using num_train_rows",
            isSplitStats: true,
            value: 600,
            rowOverrides: {
                num_entities: 500,
                num_train_rows: 1000,
                target_distribution: {
                    count: { False: 600 },
                    percentage: {}
                }
            }
        }
    ])(
        "Should render count with percentage for negative samples in $description",
        ({ isSplitStats, value, rowOverrides }) => {
            const columns = BinaryClassificationStatsColumns(
                isSplitStats,
                QueryType.TEMPORAL,
                false
            );
            const negativeColumn = findColumn(
                columns,
                "target_distribution.count.False"
            );
            const cellRenderer = getCellRenderer(negativeColumn);
            const mockContext = createMockContext(value, rowOverrides);

            const cellValue = cellRenderer(mockContext);

            expect(cellValue).toBeDefined();
        }
    );

    it("Should have correct size for expand column", () => {
        const columns = BinaryClassificationStatsColumns(
            false,
            QueryType.TEMPORAL,
            true
        );

        const expandColumn = findColumn(columns, "expand");

        expect(expandColumn.size).toBe(150);
    });

    it.each([
        { isExpanded: false, expectedText: "See Details" },
        { isExpanded: true, expectedText: "Close Details" }
    ])(
        "Should render '$expectedText' button when row is expanded=$isExpanded",
        ({ isExpanded, expectedText }) => {
            const columns = BinaryClassificationStatsColumns(
                false,
                QueryType.TEMPORAL,
                true
            );
            const expandColumn = findColumn(columns, "expand");
            const cellRenderer = getCellRenderer(expandColumn);
            const mockRow = {
                ...createMockRow(),
                getIsExpanded: () => isExpanded
            } as unknown as Row<TimeFrameStats>;

            const button = cellRenderer({
                row: mockRow
            } as CellContext<TimeFrameStats, number>);

            expect(button.props.children).toBe(expectedText);
        }
    );

    it("Should call toggleExpanded and stopPropagation when expand button is clicked", () => {
        const columns = BinaryClassificationStatsColumns(
            false,
            QueryType.TEMPORAL,
            true
        );
        const expandColumn = findColumn(columns, "expand");
        const cellRenderer = getCellRenderer(expandColumn);
        const toggleExpandedMock = vi.fn();
        const mockRow = {
            ...createMockRow(),
            getIsExpanded: () => false,
            toggleExpanded: toggleExpandedMock
        } as unknown as Row<TimeFrameStats>;
        const mockEvent = {
            stopPropagation: vi.fn()
        } as unknown as React.MouseEvent<HTMLButtonElement>;

        const button = cellRenderer({
            row: mockRow
        } as CellContext<TimeFrameStats, number>);
        button.props.onClick(mockEvent);

        expect(mockEvent.stopPropagation).toHaveBeenCalled();
        expect(toggleExpandedMock).toHaveBeenCalled();
    });
});
