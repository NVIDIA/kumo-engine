// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row, Table } from "@kumo-ai/kumo-ui-catalog";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TableListResponse } from "types/tables";

// Force the Custom SQL feature flag OFF for this suite so the picker keeps its
// prior six columns. Mirrors the enable-mock in GraphSelectTablesCustomSql.test.tsx.
vi.mock("utils/Constants", async (importOriginal) => {
    const actual = await importOriginal<typeof import("utils/Constants")>();
    return { ...actual, ENABLE_CUSTOM_SQL_TABLES: false };
});

import { GraphSelectTablesColumns } from "../GraphSelectTables";

const mockSetShowSlider = vi.fn();
const mockSetPreviewName = vi.fn();

const mockTable: TableListResponse = {
    id: "table-1",
    name: "customers",
    dateCreated: "2024-01-15T10:30:00Z",
    connector_id: "connector-123",
    source_table_name: "raw_customers",
    timestamps: {
        created_at: "2024-01-15T10:30:00Z",
        updated_at: "2024-01-20T14:45:00Z"
    }
};

type ColumnDefinition = ReturnType<typeof GraphSelectTablesColumns>[number];

const renderCell = (
    column: ColumnDefinition,
    table: Partial<TableListResponse>
) => {
    if (typeof column?.cell !== "function") {
        return column?.cell;
    }
    const mockRow: Row<TableListResponse> = {
        original: table as TableListResponse,
        id: table.id || "test-row",
        index: 0
    } as Row<TableListResponse>;
    return column.cell({
        row: mockRow
    } as CellContext<TableListResponse, unknown>);
};

const renderHeader = (column: ColumnDefinition) => {
    if (typeof column?.header !== "function") {
        return column?.header;
    }
    const mockTableInstance = {
        getIsAllRowsSelected: vi.fn().mockReturnValue(false),
        getIsSomeRowsSelected: vi.fn().mockReturnValue(false),
        toggleAllRowsSelected: vi.fn()
    } as unknown as Table<TableListResponse>;

    const headerFn = column.header as unknown as (ctx: never) => never;
    return headerFn({
        table: mockTableInstance,
        column,
        header: {} as never
    } as never);
};

describe("GraphSelectTablesColumns", () => {
    let columns: ReturnType<typeof GraphSelectTablesColumns>;

    beforeEach(() => {
        vi.clearAllMocks();
        columns = GraphSelectTablesColumns(
            false,
            mockSetShowSlider,
            mockSetPreviewName
        );
    });

    const getColumn = (id: string) => columns.find((col) => col.id === id)!;

    // Custom SQL columns are gated behind ENABLE_CUSTOM_SQL_TABLES, forced off
    // for this suite (see the vi.mock above), so the picker keeps its prior six
    // columns here. The flag-on behavior is covered in
    // GraphSelectTablesCustomSql.test.tsx.
    it("Should return the prior six columns when the feature flag is off", () => {
        const columnIds = columns.map((col) => col.id);

        expect(columnIds).toEqual([
            "select",
            "name",
            "dateCreated",
            "connector_id",
            "source_table_name",
            "preview"
        ]);
    });

    it("Should not include the Custom SQL columns when the flag is off", () => {
        const columnIds = columns.map((col) => col.id);
        expect(columnIds).not.toContain("table_kind");
        expect(columnIds).not.toContain("definition_updated_at");
    });

    describe("Select Column", () => {
        it("Should have correct size configuration", () => {
            const column = getColumn("select");

            expect(column.size).toBe(30);
        });

        it("Should render header checkbox with correct props when no rows selected", () => {
            const column = getColumn("select");
            const result = renderHeader(column) as unknown as {
                props: Record<string, unknown>;
            };

            expect(result.props.checked).toBe(false);
            expect(result.props["aria-label"]).toBe("Select all");
            expect(result.props.className).toBe("cursor-pointer");
        });
    });

    describe.each([
        { id: "name", header: "Table Name", accessorKey: "name" },
        {
            id: "dateCreated",
            header: "Created date",
            accessorKey: "dateCreated"
        },
        {
            id: "connector_id",
            header: "Connector",
            accessorKey: "connector_id"
        },
        {
            id: "source_table_name",
            header: "Source Table",
            accessorKey: "source_table_name"
        }
    ])("$id Column", ({ id, header, accessorKey }) => {
        it("Should have correct header and accessorKey", () => {
            const column = getColumn(id);

            expect(column.header).toBe(header);
            expect("accessorKey" in column && column.accessorKey).toBe(
                accessorKey
            );
        });
    });

    describe("Preview Column", () => {
        it("Should have correct header and empty accessorKey", () => {
            const column = getColumn("preview");

            expect(column.header).toBe("Preview");
            expect("accessorKey" in column && column.accessorKey).toBe("");
        });

        it("Should render preview icon with correct styling", () => {
            const column = getColumn("preview");

            const result = renderCell(column, mockTable);

            expect(result.props["data-testid"]).toBe("preview-0");
            expect(result.props.className).toBe(
                "text-title-primary cursor-pointer"
            );
            expect(result.props.children.props.className).toBe("size-4.5");
        });

        it.each([
            { showSlider: false, expectedShowSlider: true },
            { showSlider: true, expectedShowSlider: false }
        ])(
            "Should toggle showSlider to $expectedShowSlider when clicked with showSlider=$showSlider",
            ({ showSlider, expectedShowSlider }) => {
                const columnsWithSlider = GraphSelectTablesColumns(
                    showSlider,
                    mockSetShowSlider,
                    mockSetPreviewName
                );
                const column = columnsWithSlider.find(
                    (col) => col.id === "preview"
                )!;

                const result = renderCell(column, mockTable);
                result.props.onClick();

                expect(mockSetShowSlider).toHaveBeenCalledWith(
                    expectedShowSlider
                );
                expect(mockSetPreviewName).toHaveBeenCalledWith("customers");
            }
        );

        it("Should handle missing table name gracefully", () => {
            const column = getColumn("preview");
            const testTable = { ...mockTable, name: undefined };

            const result = renderCell(column, testTable);
            result.props.onClick();

            expect(mockSetShowSlider).toHaveBeenCalledWith(true);
            expect(mockSetPreviewName).toHaveBeenCalledWith(undefined);
        });
    });
});
