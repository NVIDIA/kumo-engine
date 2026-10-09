// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Row, Table } from "@kumo-ai/kumo-ui-catalog";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ColumnArgs } from "types/tables";

import {
    PreprocessingColumns,
    PreprocessingColumnsProps
} from "../PreprocessingColumns";

type ColumnDefinition = ReturnType<typeof PreprocessingColumns>[number];

const createMockRow = (data: ColumnArgs, index = 0): Row<ColumnArgs> =>
    ({
        original: data,
        index,
        getIsSelected: () => false,
        toggleSelected: vi.fn()
    }) as unknown as Row<ColumnArgs>;

const createMockTable = (): Table<ColumnArgs> =>
    ({
        getIsAllRowsSelected: () => false,
        getIsSomeRowsSelected: () => false,
        toggleAllRowsSelected: vi.fn()
    }) as unknown as Table<ColumnArgs>;

const renderCell = (
    column: ColumnDefinition,
    row: Row<ColumnArgs>,
    getValue?: () => string
) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    return render(<div>{column.cell({ row, getValue } as never)}</div>);
};

const renderHeader = (column: ColumnDefinition, table?: Table<ColumnArgs>) => {
    const headerFn = column.header;
    const context = table ? { table } : {};
    return render(
        <div>
            {typeof headerFn === "function"
                ? headerFn(context as never)
                : headerFn}
        </div>
    );
};

const defaultProps: PreprocessingColumnsProps = {
    columnDTypes: [
        { label: "String", value: "string" },
        { label: "Integer", value: "integer" }
    ],
    columnSTypes: [
        { label: "Categorical", value: "categorical" },
        { label: "Numerical", value: "numerical" }
    ],
    dTypeOptions: [
        { label: "String", value: "string" },
        { label: "Integer", value: "integer" },
        { label: "Float", value: "float" }
    ],
    onCreateTimeColSelect: vi.fn(),
    onEndTimeColSelect: vi.fn(),
    onSelectPrimaryKey: vi.fn(),
    onSetColumnDataType: vi.fn(),
    onSetColumnSType: vi.fn(),
    preservedPageIndex: 0,
    defaultPageSize: 10,
    selected: ["true", "true"],
    showPreview: true,
    sTypeOptions: [
        [
            { label: "Categorical", value: "categorical" },
            { label: "Numerical", value: "numerical" }
        ],
        [
            { label: "Categorical", value: "categorical" },
            { label: "Text", value: "text" }
        ]
    ]
};

const mockColumnData: ColumnArgs = {
    name: "customer_id",
    dtype: "string",
    stype: "categorical",
    isPrimary: false,
    isTimeColumn: false,
    isEndTimeColumn: false,
    disableSelection: false,
    isSampleAvailable: true,
    sample_rows: ["sample1", "sample2", "sample3"]
};

describe("PreprocessingColumns", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        window.innerWidth = 1920;
    });

    describe("Column Configuration", () => {
        it("Should return 6 column definitions with correct IDs", () => {
            const columns = PreprocessingColumns(defaultProps);

            expect(columns).toHaveLength(6);
            expect(columns.map((col) => col.id)).toEqual([
                "select",
                "name",
                "data_type",
                "semantic_type",
                "properties",
                "preview"
            ]);
        });

        it.each([
            { id: "select", index: 0, expectedSize: 30 },
            {
                id: "name",
                index: 1,
                expectedSize: Math.floor((1920 * 16) / 100)
            },
            {
                id: "data_type",
                index: 2,
                expectedSize: Math.floor((1920 * 8) / 100)
            },
            {
                id: "semantic_type",
                index: 3,
                expectedSize: Math.floor((1920 * 8) / 100)
            },
            {
                id: "properties",
                index: 4,
                expectedSize: Math.floor((1920 * 8) / 100)
            }
        ])(
            "Should set $id column size to $expectedSize",
            ({ index, expectedSize }) => {
                const columns = PreprocessingColumns(defaultProps);

                expect(columns[index].size).toBe(expectedSize);
            }
        );

        it.each([
            { width: 2000, expectedSize: Math.floor((2000 * 37) / 100) },
            { width: 1800, expectedSize: Math.floor((1800 * 30) / 100) }
        ])(
            "Should adjust preview column size based on window width $width",
            ({ width, expectedSize }) => {
                window.innerWidth = width;
                const columns = PreprocessingColumns(defaultProps);

                expect(columns[5].size).toBe(expectedSize);
            }
        );
    });

    describe("Select Column", () => {
        it("Should render header checkbox", () => {
            const columns = PreprocessingColumns(defaultProps);
            const mockTable = createMockTable();

            renderHeader(columns[0], mockTable);

            expect(
                screen.getByRole("checkbox", { name: "Select all" })
            ).toBeInTheDocument();
        });

        it("Should render cell checkbox", () => {
            const columns = PreprocessingColumns(defaultProps);
            const mockRow = createMockRow(mockColumnData);

            renderCell(columns[0], mockRow);

            expect(
                screen.getByRole("checkbox", { name: "Select row" })
            ).toBeInTheDocument();
        });

        it("Should handle indeterminate state in header", () => {
            const columns = PreprocessingColumns(defaultProps);
            const mockTable = {
                ...createMockTable(),
                getIsAllRowsSelected: () => false,
                getIsSomeRowsSelected: () => true
            } as unknown as Table<ColumnArgs>;

            renderHeader(columns[0], mockTable);

            expect(
                screen.getByRole("checkbox", { name: "Select all" })
            ).toBeInTheDocument();
        });
    });

    describe("Name Column", () => {
        it("Should render column name in cell", () => {
            const columns = PreprocessingColumns(defaultProps);
            const mockRow = createMockRow(mockColumnData);

            const { container } = renderCell(
                columns[1],
                mockRow,
                () => "customer_id"
            );

            expect(screen.getByText("customer_id")).toBeInTheDocument();
            expect(
                container.querySelector('[data-testid="column-name-0"]')
            ).toBeInTheDocument();
        });
    });

    describe("Data Type Column", () => {
        it("Should render ChangeColumnType component", () => {
            const columns = PreprocessingColumns(defaultProps);
            const mockRow = createMockRow(mockColumnData, 0);

            const { container } = renderCell(columns[2], mockRow);

            expect(
                container.querySelector('[data-testid="testid-select-d-type"]')
            ).toBeInTheDocument();
        });

        it.each([
            {
                scenario: "row is not selected",
                props: { ...defaultProps, selected: ["false", "false"] },
                rowData: mockColumnData
            },
            {
                scenario: "disableSelection is true",
                props: defaultProps,
                rowData: { ...mockColumnData, disableSelection: true }
            }
        ])(
            "Should disable ChangeColumnType when $scenario",
            ({ props, rowData }) => {
                const columns = PreprocessingColumns(props);
                const mockRow = createMockRow(rowData, 0);

                const { container } = renderCell(columns[2], mockRow);

                expect(
                    container.querySelector(
                        '[data-testid="testid-select-d-type"]'
                    )
                ).toBeDisabled();
            }
        );

        it("Should sort by dtype correctly", () => {
            const columns = PreprocessingColumns(defaultProps);
            const sortingFn = columns[2].sortingFn;

            if (typeof sortingFn === "function") {
                const rowA = createMockRow({
                    ...mockColumnData,
                    dtype: "string"
                });
                const rowB = createMockRow({
                    ...mockColumnData,
                    dtype: "integer"
                });

                const result = sortingFn(
                    rowA as never,
                    rowB as never,
                    "data_type"
                );

                expect(result).toBeGreaterThan(0);
            }
        });
    });

    describe("Semantic Type Column", () => {
        it("Should render semantic type label", () => {
            const columns = PreprocessingColumns(defaultProps);
            const mockRow = createMockRow(mockColumnData, 0);

            renderCell(columns[3], mockRow);

            expect(screen.getByText("Categorical")).toBeInTheDocument();
        });

        it("Should disable ChangeColumnType when row is not selected", () => {
            const props = { ...defaultProps, selected: ["false"] };
            const columns = PreprocessingColumns(props);
            const mockRow = createMockRow(mockColumnData, 0);

            const { container } = renderCell(columns[3], mockRow);

            expect(
                container.querySelector('[data-testid="testid-select-s-type"]')
            ).toBeDisabled();
        });

        it("Should sort by stype correctly", () => {
            const columns = PreprocessingColumns(defaultProps);
            const sortingFn = columns[3].sortingFn;

            if (typeof sortingFn === "function") {
                const rowA = createMockRow({
                    ...mockColumnData,
                    stype: "text"
                });
                const rowB = createMockRow({
                    ...mockColumnData,
                    stype: "categorical"
                });

                const result = sortingFn(
                    rowA as never,
                    rowB as never,
                    "semantic_type"
                );

                expect(result).toBeGreaterThan(0);
            }
        });
    });

    describe("Properties Column", () => {
        it.each([
            { property: "isPrimary", value: true, expectedText: "Primary Key" },
            {
                property: "isTimeColumn",
                value: true,
                expectedText: "Create date"
            },
            {
                property: "isEndTimeColumn",
                value: true,
                expectedText: "End date"
            }
        ])(
            "Should render $expectedText when $property is $value",
            ({ property, value, expectedText }) => {
                const columns = PreprocessingColumns(defaultProps);
                const mockRow = createMockRow(
                    { ...mockColumnData, [property]: value },
                    0
                );

                renderCell(columns[4], mockRow);

                expect(screen.getByText(expectedText)).toBeInTheDocument();
            }
        );

        it.each([
            {
                scenario: "row is not selected",
                props: { ...defaultProps, selected: ["false"] },
                rowData: mockColumnData
            },
            {
                scenario: "disableSelection is true",
                props: defaultProps,
                rowData: { ...mockColumnData, disableSelection: true }
            }
        ])("Should disable Select when $scenario", ({ props, rowData }) => {
            const columns = PreprocessingColumns(props);
            const mockRow = createMockRow(rowData, 0);

            const { container } = renderCell(columns[4], mockRow);

            expect(
                container.querySelector(
                    '[data-testid="testid-properties-select-0"]'
                )
            ).toBeDisabled();
        });
    });

    describe("Preview Column", () => {
        it("Should render SampleCell component with data", () => {
            const columns = PreprocessingColumns(defaultProps);
            const mockRow = createMockRow(mockColumnData, 0);

            const { container } = renderCell(columns[5], mockRow);

            expect(
                container.querySelector('[data-testid="sample-cell"]')
            ).toBeInTheDocument();
        });
    });

    describe("Edge Cases", () => {
        it("Should handle missing column data gracefully", () => {
            const columns = PreprocessingColumns(defaultProps);
            const mockRow = createMockRow({} as ColumnArgs, 0);

            expect(() =>
                renderCell(columns[1], mockRow, () => "")
            ).not.toThrow();
        });

        it("Should handle different page indices correctly", () => {
            const props = {
                ...defaultProps,
                preservedPageIndex: 2,
                defaultPageSize: 10
            };
            const columns = PreprocessingColumns(props);
            const mockRow = createMockRow(mockColumnData, 5);

            const { container } = renderCell(columns[2], mockRow);

            expect(
                container.querySelector('[data-testid="testid-select-d-type"]')
            ).toBeInTheDocument();
        });
    });
});
