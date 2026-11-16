// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row } from "@kumo-ai/kumo-ui-catalog";
import { describe, expect, it } from "vitest";

import { SubgraphSummary } from "types/training_job";

import { SubgraphColumns } from "../Subgraph";

const createMockRow = (
    overrides?: Partial<SubgraphSummary>
): Row<SubgraphSummary> =>
    ({
        original: {
            property: "test_property",
            current: "test_current",
            global: "test_global",
            ...overrides
        }
    }) as unknown as Row<SubgraphSummary>;

const findColumn = (
    columns: ReturnType<typeof SubgraphColumns>,
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

describe("SubgraphColumns", () => {
    it("Should generate all columns", () => {
        const columns = SubgraphColumns();

        expect(columns).toHaveLength(3);
        expect(columns.map((col) => col.id)).toEqual([
            "property",
            "current",
            "global"
        ]);
    });

    it.each([
        { columnId: "property", expectedHeader: "Property" },
        { columnId: "current", expectedHeader: "Current Entity" },
        { columnId: "global", expectedHeader: "Global" }
    ])(
        "Should have correct header for $columnId column",
        ({ columnId, expectedHeader }) => {
            const columns = SubgraphColumns();
            const column = findColumn(columns, columnId);

            expect(column.header).toBe(expectedHeader);
        }
    );

    it.each([
        { columnId: "property", expectedAccessorKey: "property" },
        { columnId: "current", expectedAccessorKey: "current" },
        { columnId: "global", expectedAccessorKey: "global" }
    ])(
        "Should have correct accessorKey for $columnId column",
        ({ columnId, expectedAccessorKey }) => {
            const columns = SubgraphColumns();
            const column = findColumn(columns, columnId);

            expect("accessorKey" in column && column.accessorKey).toBe(
                expectedAccessorKey
            );
        }
    );

    describe("Current Entity and Global columns", () => {
        it.each([
            {
                columnId: "current",
                value: "test_value",
                expectedResult: "test_value"
            },
            {
                columnId: "global",
                value: "global_value",
                expectedResult: "global_value"
            },
            { columnId: "current", value: 123, expectedResult: "123" },
            { columnId: "global", value: 456, expectedResult: "456" },
            { columnId: "current", value: 0, expectedResult: "0" },
            { columnId: "global", value: 0, expectedResult: "0" },
            { columnId: "current", value: -100, expectedResult: "-100" },
            {
                columnId: "global",
                value: 999999999,
                expectedResult: "999999999"
            },
            { columnId: "current", value: 3.14159, expectedResult: "3.14159" },
            { columnId: "current", value: "", expectedResult: "" },
            { columnId: "global", value: "", expectedResult: "" }
        ])(
            "Should render $value correctly in $columnId column",
            ({ columnId, value, expectedResult }) => {
                const columns = SubgraphColumns();
                const column = findColumn(columns, columnId);
                const cellRenderer = getCellRenderer(column);
                const mockRow = createMockRow({ [columnId]: value });

                const cellValue = cellRenderer({
                    row: mockRow
                } as CellContext<SubgraphSummary, string | number>);

                expect(cellValue).toBe(expectedResult);
            }
        );

        it.each([
            { columnId: "current", value: null as unknown as undefined },
            { columnId: "global", value: null as unknown as undefined }
        ])(
            "Should render null value correctly in $columnId column",
            ({ columnId, value }) => {
                const columns = SubgraphColumns();
                const column = findColumn(columns, columnId);
                const cellRenderer = getCellRenderer(column);
                const mockRow = createMockRow({ [columnId]: value });

                const cellValue = cellRenderer({
                    row: mockRow
                } as CellContext<SubgraphSummary, string | number>);

                expect(cellValue).toBeNull();
            }
        );

        it.each([
            { columnId: "current", value: undefined },
            { columnId: "global", value: undefined }
        ])(
            "Should render undefined value correctly in $columnId column",
            ({ columnId, value }) => {
                const columns = SubgraphColumns();
                const column = findColumn(columns, columnId);
                const cellRenderer = getCellRenderer(column);
                const mockRow = createMockRow({ [columnId]: value });

                const cellValue = cellRenderer({
                    row: mockRow
                } as CellContext<SubgraphSummary, string | number>);

                expect(cellValue).toBeUndefined();
            }
        );

        it.each([
            { columnId: "current", globalValue: "some_global" },
            { columnId: "global", globalValue: "<MISSING>" }
        ])(
            "Should render <MISSING> with tooltip in $columnId column",
            ({ columnId, globalValue }) => {
                const columns = SubgraphColumns();
                const column = findColumn(columns, columnId);
                const cellRenderer = getCellRenderer(column);
                const mockRow = createMockRow({
                    [columnId]: "<MISSING>",
                    global: globalValue
                });

                const result = cellRenderer({
                    row: mockRow
                } as CellContext<SubgraphSummary, string | number>);

                expect(result).toBeDefined();
                expect(result.type).toBe("div");
                expect(result.props.className).toBe(
                    "flex items-center justify-between"
                );
            }
        );

        it.each([
            {
                columnId: "current",
                globalValue: "test_global_value",
                expectedTooltipId: "tooltip-missing-current-test_global_value"
            },
            {
                columnId: "global",
                globalValue: "<MISSING>",
                expectedTooltipId: "tooltip-missing-global-<MISSING>"
            }
        ])(
            "Should have correct tooltip id for <MISSING> in $columnId column",
            ({ columnId, globalValue, expectedTooltipId }) => {
                const columns = SubgraphColumns();
                const column = findColumn(columns, columnId);
                const cellRenderer = getCellRenderer(column);
                const mockRow = createMockRow({
                    [columnId]: "<MISSING>",
                    global: globalValue
                });

                const result = cellRenderer({
                    row: mockRow
                } as CellContext<SubgraphSummary, string | number>);

                const tooltip = result.props.children;
                expect(tooltip.props.id).toBe(expectedTooltipId);
            }
        );

        it.each([{ columnId: "current" }, { columnId: "global" }])(
            "Should have correct tooltip content for <MISSING> in $columnId column",
            ({ columnId }) => {
                const columns = SubgraphColumns();
                const column = findColumn(columns, columnId);
                const cellRenderer = getCellRenderer(column);
                const mockRow = createMockRow({ [columnId]: "<MISSING>" });

                const result = cellRenderer({
                    row: mockRow
                } as CellContext<SubgraphSummary, string | number>);

                const tooltip = result.props.children;
                const tooltipContent = tooltip.props.children[1];
                expect(tooltipContent.props.children).toBe(
                    "Cannot extract corresponding value due to the limitation of graph structure or the value is unsupported for the current task."
                );
            }
        );
    });
});
