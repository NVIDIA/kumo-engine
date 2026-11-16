// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row } from "@kumo-ai/kumo-ui-catalog";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { JSTable } from "types/tables";

import { ConnectedTablesColumns } from "../ConnectedTables";

const mockOnTableDeleteClick = vi.fn();

const mockTable: JSTable = {
    id: "table-1",
    name_alias: "customers",
    num_cols: "10",
    num_rows: 1000,
    size: "50 MB",
    timeRange: "2024-01-01 to 2024-12-31",
    updated: "2 days ago",
    updatedOrigin: "2024-01-20T10:30:00Z",
    table: {} as JSTable["table"],
    source_table: {} as JSTable["source_table"],
    s3Connector: {} as JSTable["s3Connector"],
    description: "",
    sourceTable: "",
    connector: null,
    connectorId: null,
    table_location: null,
    name_link: null,
    graphs_original: [],
    graphs: [],
    sortCreated: "",
    sortUpdated: "",
    pkey: null,
    timeCol: null,
    endTimeCol: null
};

type ColumnDefinition = ReturnType<typeof ConnectedTablesColumns>[number];

const renderCell = (column: ColumnDefinition, table: Partial<JSTable>) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    const mockRow: Row<JSTable> = {
        original: table as JSTable,
        id: table.id || "test-row"
    } as Row<JSTable>;
    return column.cell({ row: mockRow } as CellContext<JSTable, unknown>);
};

describe("ConnectedTablesColumns", () => {
    let columns: ReturnType<typeof ConnectedTablesColumns>;

    beforeEach(() => {
        vi.clearAllMocks();
        mockNavigateCall();
        columns = ConnectedTablesColumns(
            mockOnTableDeleteClick,
            "project-test"
        );
    });

    const getColumn = (id: string) => columns.find((col) => col.id === id)!;

    describe("Table Name Column", () => {
        it("Should render table name as TextLink with correct props", () => {
            const column = getColumn("name_alias");

            const result = renderCell(column, mockTable);

            expect(result.props.id).toBe("connected-table-cols-customers");
            expect(result.props.pageLink).toBe(
                "/project/project-test/tables/customers"
            );
            expect(result.props.title).toBe("customers");
        });

        it("Should navigate when clicked", () => {
            const column = getColumn("name_alias");
            const result = renderCell(column, mockTable);
            const mockEvent = { preventDefault: vi.fn() } as unknown as Event;

            result.props.onClick(mockEvent);

            expect(mockEvent.preventDefault).toHaveBeenCalled();
            expect(mockUseNavigate).toHaveBeenCalledWith(
                "/project/project-test/tables/customers"
            );
        });
    });

    describe("Dimensions Column", () => {
        it("Should render all dimensions when values are present", () => {
            const column = getColumn("num_cols");

            const result = renderCell(column, mockTable);

            expect(result.props.children[0].props.children).toBe("10 Col, ");
            expect(result.props.children[1].props.children).toBe("1000 Rows, ");
            expect(result.props.children[2]).toBe("50 MB");
        });

        it.each([
            { field: "num_cols", index: 0, expected: "" },
            { field: "num_rows", index: 1, expected: "" }
        ])(
            "Should render empty string for missing $field",
            ({ field, index, expected }) => {
                const column = getColumn("num_cols");
                const testTable = { ...mockTable, [field]: undefined };

                const result = renderCell(column, testTable);

                expect(result.props.children[index]).toBe(expected);
            }
        );

        it("Should render undefined for missing size", () => {
            const column = getColumn("num_cols");
            const testTable = { ...mockTable, size: undefined };

            const result = renderCell(column, testTable);

            expect(result.props.children[2]).toBeUndefined();
        });
    });

    describe("Fact Column", () => {
        it.each([{ timeRange: undefined }, { timeRange: "" }])(
            "Should render emdash when timeRange is $timeRange",
            ({ timeRange }) => {
                const column = columns.find((col) => col.header === "Fact")!;
                const testTable = { ...mockTable, timeRange } as JSTable;

                const result = renderCell(column, testTable);

                expect(result.props.children.props.type).toBe("emdash");
            }
        );
    });

    describe("Updated Column", () => {
        it("Should render updated time with tooltip", () => {
            const column = getColumn("Updated");

            const result = renderCell(column, mockTable);

            expect(result.props.id).toBe("updated-table-1");
            expect(result.props.children[0].props.children.props.children).toBe(
                "2 days ago"
            );
        });

        it("Should render dash when updated is missing", () => {
            const column = getColumn("Updated");
            const testTable = { ...mockTable, updated: undefined };

            const result = renderCell(column, testTable);

            expect(result).toBe("-");
        });
    });

    describe("Action Column", () => {
        it("Should render delete icon with correct styling", () => {
            const column = getColumn("action");

            const result = renderCell(column, mockTable);

            expect(result.props.className).toBe("cursor-pointer");
            expect(result.props.children.props.className).toBe("size-5.5");
        });

        it("Should call delete handler when clicked", () => {
            const column = getColumn("action");
            const result = renderCell(column, mockTable);

            result.props.onClick();

            expect(mockOnTableDeleteClick).toHaveBeenCalledWith(mockTable);
        });
    });
});
