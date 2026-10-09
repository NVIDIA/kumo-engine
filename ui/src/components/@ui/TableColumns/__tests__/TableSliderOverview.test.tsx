// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row } from "@kumo-ai/kumo-ui-catalog";
import { render } from "@testing-library/react";
import { ColStatsResponse } from "gen/v0/types";
import { describe, expect, it } from "vitest";

import { ColumnArgs, TableColData, TableDetails } from "types/tables";

import { TableSliderOverviewColumns } from "../TableSliderOverview";

const mockTable: TableDetails = {
    cols: [],
    source_table: {} as TableDetails["source_table"],
    pkey: "id",
    time_col: "created_at",
    end_time_col: undefined
};

const mockColumnStats: ColStatsResponse = {
    statistics_available: true,
    approx: false,
    rows_used: 100,
    columns: [
        {
            name: "id",
            sample_rows: ["1", "2", "3", "4", "5"],
            stype: "number",
            dtype: "int64"
        },
        {
            name: "name",
            sample_rows: ["Alice", "Bob", "Charlie"],
            stype: "text",
            dtype: "string"
        },
        {
            name: "email",
            sample_rows: [
                "alice@example.com",
                "bob@example.com",
                "charlie@example.com",
                "david@example.com",
                "eve@example.com"
            ],
            stype: "text",
            dtype: "string"
        },
        {
            name: "description",
            sample_rows: [
                "This is a very long description that exceeds 75 characters and Should be truncated with a tooltip"
            ],
            stype: "text",
            dtype: "string"
        },
        {
            name: "image",
            sample_rows: [
                "photo1.jpg",
                "photo2.png",
                "photo3.jpeg",
                "photo4.gif",
                "photo5.jpg"
            ],
            stype: "image",
            dtype: "string"
        },
        {
            name: "notes",
            sample_rows: [null, "", "Some note"],
            stype: "text",
            dtype: "string"
        }
    ]
};

type ColumnDefinition = ReturnType<typeof TableSliderOverviewColumns>[number];

const renderCell = (
    column: ColumnDefinition,
    rowData: Partial<TableColData>
) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    const mockRow: Row<TableColData> = {
        original: rowData as TableColData,
        id: rowData.name || "test-row"
    } as Row<TableColData>;
    return column.cell({ row: mockRow } as CellContext<TableColData, unknown>);
};

describe("TableSliderOverviewColumns", () => {
    describe("Column Configuration", () => {
        it.each([
            {
                id: "name",
                header: "Column Name",
                accessorKey: "name",
                size: 25
            },
            {
                id: "dtype",
                header: "Data type",
                accessorKey: "dtype",
                size: 25
            },
            {
                id: "stype",
                header: "Semantic Type",
                accessorKey: "stype",
                size: 25
            },
            {
                id: "props",
                header: "Properties",
                accessorKey: "props",
                size: 25
            },
            {
                id: "sampleRows",
                header: "Preview",
                accessorKey: "sampleRows",
                size: 200
            }
        ])(
            "Should configure $id column correctly",
            ({ id, header, accessorKey, size }) => {
                const columns = TableSliderOverviewColumns(
                    mockTable,
                    false,
                    mockColumnStats
                );
                const column = columns.find((col) => col.id === id);

                expect(column?.id).toBe(id);
                expect(column?.header).toBe(header);
                expect(column?.accessorKey).toBe(accessorKey);
                expect(column?.size).toBe(size);
            }
        );
    });

    describe("Properties Column", () => {
        it.each([
            { columnName: "id", expectedText: "Primary key" },
            { columnName: "created_at", expectedText: "Time Column" }
        ])(
            "Should render $expectedText for $columnName",
            ({ columnName, expectedText }) => {
                const columns = TableSliderOverviewColumns(
                    mockTable,
                    false,
                    mockColumnStats
                );
                const propsColumn = columns.find((col) => col.id === "props")!;

                const result = renderCell(propsColumn, { name: columnName });
                const { getByText } = render(<div>{result}</div>);

                expect(getByText(expectedText)).toBeInTheDocument();
            }
        );

        it("Should render emdash for regular columns", () => {
            const columns = TableSliderOverviewColumns(
                mockTable,
                false,
                mockColumnStats
            );
            const propsColumn = columns.find((col) => col.id === "props")!;

            const result = renderCell(propsColumn, { name: "email" });

            expect(result.props.type).toBe("emdash");
        });

        it.each([
            { field: "pkey", columnName: "id" },
            { field: "time_col", columnName: "created_at" }
        ])(
            "Should render emdash when $field is undefined",
            ({ field, columnName }) => {
                const tableConfig = { ...mockTable, [field]: undefined };
                const columns = TableSliderOverviewColumns(
                    tableConfig,
                    false,
                    mockColumnStats
                );
                const propsColumn = columns.find((col) => col.id === "props")!;

                const result = renderCell(propsColumn, { name: columnName });

                expect(result.props.type).toBe("emdash");
            }
        );
    });

    describe("Preview Column - Loading State", () => {
        it("Should render skeleton when fetching samples", () => {
            const columns = TableSliderOverviewColumns(
                mockTable,
                true,
                mockColumnStats
            );
            const previewColumn = columns.find(
                (col) => col.id === "sampleRows"
            )!;

            const result = renderCell(previewColumn, { name: "id" });

            expect(result.props.children.props.className).toBe("h-4 w-40");
            expect(result.props.className).toBe("min-w-175 w-175");
        });
    });

    describe("Preview Column - Text Samples", () => {
        it("Should render short text samples without tooltip", () => {
            const columns = TableSliderOverviewColumns(
                mockTable,
                false,
                mockColumnStats
            );
            const previewColumn = columns.find(
                (col) => col.id === "sampleRows"
            )!;

            const result = renderCell(previewColumn, { name: "name" });
            const { container } = render(<div>{result}</div>);

            expect(container.textContent).toBe("Alice, Bob, Charlie");
            expect(container.querySelector('[role="tooltip"]')).toBeNull();
        });

        it("Should truncate long text samples with ellipsis", () => {
            const columns = TableSliderOverviewColumns(
                mockTable,
                false,
                mockColumnStats
            );
            const previewColumn = columns.find(
                (col) => col.id === "sampleRows"
            )!;

            const result = renderCell(previewColumn, { name: "description" });
            const { container } = render(<div>{result}</div>);

            expect(container.textContent).toContain("...");
            expect(container.textContent?.length).toBeLessThan(200);
        });

        it("Should replace null and empty strings with N/A", () => {
            const columns = TableSliderOverviewColumns(
                mockTable,
                false,
                mockColumnStats
            );
            const previewColumn = columns.find(
                (col) => col.id === "sampleRows"
            )!;

            const result = renderCell(previewColumn, { name: "notes" });
            const { container } = render(<div>{result}</div>);

            expect(container.textContent).toBe("N/A, N/A, Some note");
        });

        it("Should render image samples with ellipsis", () => {
            const columns = TableSliderOverviewColumns(
                mockTable,
                false,
                mockColumnStats
            );
            const previewColumn = columns.find(
                (col) => col.id === "sampleRows"
            )!;

            const result = renderCell(previewColumn, { name: "image" });
            const { container } = render(<div>{result}</div>);

            expect(container.textContent).toContain("...");
        });
    });

    describe("Preview Column - Edge Cases", () => {
        it.each([
            {
                scenario: "column not in columnStats",
                columnName: "nonexistent_column",
                customStats: mockColumnStats
            },
            {
                scenario: "empty sample_rows array",
                columnName: "empty",
                customStats: {
                    ...mockColumnStats,
                    columns: [{ name: "empty", sample_rows: [] }]
                }
            },
            {
                scenario: "undefined sample_rows",
                columnName: "no_samples",
                customStats: {
                    ...mockColumnStats,
                    columns: [{ name: "no_samples" } as ColumnArgs]
                }
            },
            {
                scenario: "null sample_rows",
                columnName: "null_samples",
                customStats: {
                    ...mockColumnStats,
                    columns: [
                        {
                            name: "null_samples",
                            sample_rows: null
                        } as ColumnArgs
                    ]
                }
            }
        ])(
            "Should render empty string for $scenario",
            ({ columnName, customStats }) => {
                const columns = TableSliderOverviewColumns(
                    mockTable,
                    false,
                    customStats as ColStatsResponse
                );
                const previewColumn = columns.find(
                    (col) => col.id === "sampleRows"
                )!;

                const result = renderCell(previewColumn, { name: columnName });
                const { container } = render(<div>{result}</div>);

                expect(container.textContent).toBe("");
            }
        );

        it("Should handle null columnStats", () => {
            const columns = TableSliderOverviewColumns(mockTable, false, null);
            const previewColumn = columns.find(
                (col) => col.id === "sampleRows"
            )!;

            const result = renderCell(previewColumn, { name: "id" });
            const { container } = render(<div>{result}</div>);

            expect(container.textContent).toBe("");
        });
    });
});
