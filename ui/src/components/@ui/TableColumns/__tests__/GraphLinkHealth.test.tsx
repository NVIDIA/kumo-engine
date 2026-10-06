// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row } from "@kumo-ai/kumo-ui-catalog";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { EdgeStats } from "types/graphs";

import { GraphLinkHealthColumns } from "../GraphLinkHealth";

const createMockRow = (overrides?: Partial<EdgeStats>): Row<EdgeStats> =>
    ({
        original: {
            col_a: "table_a.id",
            col_a_table_name: "table_a",
            col_a_column_name: "id",
            col_a_table_id: "table_a_id",
            col_b: "table_b.foreign_id",
            col_b_table_name: "table_b",
            col_b_column_name: "foreign_id",
            col_b_table_id: "table_b_id",
            link_type: "one_to_many",
            weak_edge: false,
            matches_a_in_b: "1000",
            matches_a_in_b_original: 1000,
            percent_a_in_b: "95.5%",
            weak_a_in_b: false,
            matches_b_in_a: "950",
            matches_b_in_a_original: 950,
            percent_b_in_a: "90.0%",
            weak_b_in_a: false,
            ...overrides
        }
    }) as unknown as Row<EdgeStats>;

const renderCell = (
    column: ReturnType<typeof GraphLinkHealthColumns>[number],
    row: Row<EdgeStats>
) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    return render(
        <div>{column.cell({ row } as CellContext<EdgeStats, unknown>)}</div>
    );
};

const findColumn = (
    columns: ReturnType<typeof GraphLinkHealthColumns>,
    accessorKey: string
) => {
    const column = columns.find(
        (col) => "accessorKey" in col && col.accessorKey === accessorKey
    );
    if (!column) {
        throw new Error(`Column with accessorKey "${accessorKey}" not found`);
    }
    return column;
};

describe("GraphLinkHealthColumns", () => {
    it("Should return 5 column definitions", () => {
        const columns = GraphLinkHealthColumns();

        expect(columns).toHaveLength(5);
    });

    it("Should have correct column accessorKeys", () => {
        const columns = GraphLinkHealthColumns();
        const accessorKeys = columns.map((col) =>
            "accessorKey" in col ? col.accessorKey : undefined
        );

        expect(accessorKeys).toEqual([
            "col_a",
            "col_b",
            "link_type",
            "matches_a_in_b",
            "matches_b_in_a"
        ]);
    });

    it.each([
        {
            accessorKey: "col_a",
            tableNameField: "col_a_table_name" as const,
            columnNameField: "col_a_column_name" as const,
            tableName: "users",
            columnName: "user_id"
        },
        {
            accessorKey: "col_b",
            tableNameField: "col_b_table_name" as const,
            columnNameField: "col_b_column_name" as const,
            tableName: "orders",
            columnName: "customer_id"
        }
    ])(
        "Should render table and column name for $accessorKey",
        ({
            accessorKey,
            tableNameField,
            columnNameField,
            tableName,
            columnName
        }) => {
            const columns = GraphLinkHealthColumns();
            const column = findColumn(columns, accessorKey);
            const mockRow = createMockRow({
                [tableNameField]: tableName,
                [columnNameField]: columnName
            });

            const { container } = renderCell(column, mockRow);

            expect(screen.getByText(tableName)).toBeInTheDocument();
            expect(screen.getByText(`.${columnName}`)).toBeInTheDocument();
            const tableNameLabel = container.querySelector(
                "label:not(.text-title-secondary)"
            );
            expect(tableNameLabel?.textContent).toBe(tableName);
        }
    );

    it.each([
        { accessorKey: "col_a", columnNameField: "col_a_column_name" as const },
        { accessorKey: "col_b", columnNameField: "col_b_column_name" as const }
    ])(
        "Should handle empty table name for $accessorKey",
        ({ accessorKey, columnNameField }) => {
            const columns = GraphLinkHealthColumns();
            const column = findColumn(columns, accessorKey);
            const mockRow = createMockRow({
                [`${accessorKey}_table_name`]: undefined as never,
                [columnNameField]: "id"
            });

            renderCell(column, mockRow);

            expect(screen.getByText(".id")).toBeInTheDocument();
        }
    );

    it("Should have accessorKey for link_type column", () => {
        const columns = GraphLinkHealthColumns();
        const linkTypeColumn = findColumn(columns, "link_type");

        expect("accessorKey" in linkTypeColumn).toBe(true);
        if ("accessorKey" in linkTypeColumn) {
            expect(linkTypeColumn.accessorKey).toBe("link_type");
        }
    });

    it("Should have custom sorting function for matches_a_in_b", () => {
        const columns = GraphLinkHealthColumns();
        const matchesColumn = findColumn(columns, "matches_a_in_b");

        expect(matchesColumn.sortingFn).toBeDefined();
        expect(typeof matchesColumn.sortingFn).toBe("function");
    });

    it.each([
        {
            description: "matches_a_in_b when link is healthy",
            accessorKey: "matches_a_in_b",
            matches: "1000",
            percent: "95.5%",
            weak_a_in_b: false,
            weak_b_in_a: false,
            expectOrange: false
        },
        {
            description: "matches_b_in_a when link is healthy",
            accessorKey: "matches_b_in_a",
            matches: "950",
            percent: "90.0%",
            weak_a_in_b: false,
            weak_b_in_a: false,
            expectOrange: false
        },
        {
            description: "matches_a_in_b with weak link",
            accessorKey: "matches_a_in_b",
            matches: "500",
            percent: "45.0%",
            weak_a_in_b: true,
            weak_b_in_a: true,
            expectOrange: true
        },
        {
            description: "matches_b_in_a with weak link",
            accessorKey: "matches_b_in_a",
            matches: "400",
            percent: "40.0%",
            weak_a_in_b: true,
            weak_b_in_a: true,
            expectOrange: true
        }
    ])(
        "Should render $description",
        ({
            accessorKey,
            matches,
            percent,
            weak_a_in_b,
            weak_b_in_a,
            expectOrange
        }) => {
            const columns = GraphLinkHealthColumns();
            const column = findColumn(columns, accessorKey);
            const mockRow = createMockRow({
                [accessorKey]: matches,
                [`percent_${accessorKey.split("_").slice(1).join("_")}`]:
                    percent,
                weak_a_in_b,
                weak_b_in_a
            });

            const { container } = renderCell(column, mockRow);

            expect(screen.getByText(matches)).toBeInTheDocument();
            expect(screen.getByText(`(${percent})`)).toBeInTheDocument();

            const orangeDiv = container.querySelector("div.text-orange-500");
            if (expectOrange) {
                expect(orangeDiv).toBeInTheDocument();
                expect(orangeDiv?.textContent).toBe(`(${percent})`);
            } else {
                expect(orangeDiv).not.toBeInTheDocument();
            }
        }
    );

    it.each([
        {
            description: "only weak_a_in_b is true",
            accessorKey: "matches_a_in_b",
            weak_a_in_b: true,
            weak_b_in_a: false
        },
        {
            description: "only weak_b_in_a is true",
            accessorKey: "matches_b_in_a",
            weak_a_in_b: false,
            weak_b_in_a: true
        }
    ])(
        "Should highlight percent when $description",
        ({ accessorKey, weak_a_in_b, weak_b_in_a }) => {
            const columns = GraphLinkHealthColumns();
            const column = findColumn(columns, accessorKey);
            const mockRow = createMockRow({
                weak_a_in_b,
                weak_b_in_a
            });

            const { container } = renderCell(column, mockRow);

            const orangeDiv = container.querySelector("div.text-orange-500");
            expect(orangeDiv).toBeInTheDocument();
        }
    );

    it("Should handle numeric values for matches", () => {
        const columns = GraphLinkHealthColumns();
        const matchesAInBColumn = findColumn(columns, "matches_a_in_b");
        const mockRow = createMockRow({
            matches_a_in_b: 5000,
            percent_a_in_b: "100.0%"
        });

        renderCell(matchesAInBColumn, mockRow);

        expect(screen.getByText("5000")).toBeInTheDocument();
    });
});
