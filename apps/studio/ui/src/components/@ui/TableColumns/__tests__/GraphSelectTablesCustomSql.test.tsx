// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row } from "@kumo-ai/kumo-ui-catalog";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TableKind, TableListResponse } from "types/tables";

// Enable the Custom SQL feature flag for this suite so the picker exposes its
// Kind + Definition-updated columns.
vi.mock("utils/Constants", async (importOriginal) => {
    const actual = await importOriginal<typeof import("utils/Constants")>();
    return { ...actual, ENABLE_CUSTOM_SQL_TABLES: true };
});

import { GraphSelectTablesColumns } from "../GraphSelectTables";

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
    const mockRow = {
        original: table as TableListResponse,
        id: table.id || "test-row",
        index: 0
    } as Row<TableListResponse>;
    return column.cell({
        row: mockRow
    } as CellContext<TableListResponse, unknown>);
};

describe("GraphSelectTablesColumns (Custom SQL flag on)", () => {
    let columns: ReturnType<typeof GraphSelectTablesColumns>;

    beforeEach(() => {
        vi.clearAllMocks();
        columns = GraphSelectTablesColumns(false, vi.fn(), vi.fn());
    });

    const getColumn = (id: string) => columns.find((col) => col.id === id)!;

    it("Should include the Kind and Definition-updated columns", () => {
        expect(columns.map((col) => col.id)).toEqual([
            "select",
            "name",
            "table_kind",
            "dateCreated",
            "connector_id",
            "source_table_name",
            "definition_updated_at",
            "preview"
        ]);
    });

    it("Should render a SQL badge for SQL tables", () => {
        const result = renderCell(getColumn("table_kind"), {
            ...mockTable,
            table_kind: TableKind.Sql
        });
        expect(result.props.children).toBe("SQL");
        expect(result.props.variant).toBe("pink");
    });

    it("Should render a Raw badge for raw tables", () => {
        const result = renderCell(getColumn("table_kind"), {
            ...mockTable,
            table_kind: TableKind.Raw
        });
        expect(result.props.children).toBe("Raw");
        expect(result.props.variant).toBe("default");
    });

    it("Should render a dash when there is no updated timestamp", () => {
        expect(
            renderCell(getColumn("definition_updated_at"), {
                ...mockTable,
                timestamps: {
                    created_at: "2024-01-15T10:30:00Z",
                    updated_at: ""
                }
            })
        ).toBe("-");
    });

    it("Should format the standard updated timestamp when present (Q3)", () => {
        // The column reuses the entry's `timestamps.updated_at`, not a distinct
        // definition field.
        expect(
            renderCell(getColumn("definition_updated_at"), mockTable)
        ).not.toBe("-");
    });
});
