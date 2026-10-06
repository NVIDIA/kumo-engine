// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Row } from "@kumo-ai/kumo-ui-catalog";
import { describe, expect, it } from "vitest";

import { ColumnRow, OutputTableColumns } from "../OutputTable";

type ColumnDefinition = ReturnType<typeof OutputTableColumns>[number];

const createRow = (data: ColumnRow | null): Row<ColumnRow> => {
    return { original: data } as Row<ColumnRow>;
};

const getSortingFn = (
    column: ReturnType<typeof OutputTableColumns>[number]
) => {
    if (!column.sortingFn || typeof column.sortingFn !== "function") {
        throw new Error("sortingFn should be a function");
    }
    return column.sortingFn;
};

const renderCell = (column: ColumnDefinition, row: Row<ColumnRow>) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    return column.cell({ row } as never);
};

describe("OutputTableColumns", () => {
    describe("Column Configuration", () => {
        it.each([
            {
                hasSamples: false,
                expectedLength: 2,
                expectedIds: ["name", "colType"],
                expectedHeaders: ["Column name", "Data type"],
                description: "without samples"
            },
            {
                hasSamples: true,
                expectedLength: 3,
                expectedIds: ["name", "colType", "samples"],
                expectedHeaders: ["Column name", "Data type", "Column samples"],
                description: "with samples"
            }
        ])(
            "Should return correct columns $description",
            ({ hasSamples, expectedLength, expectedIds, expectedHeaders }) => {
                const columns = OutputTableColumns(hasSamples);

                expect(columns).toHaveLength(expectedLength);
                expectedIds.forEach((id, index) => {
                    expect(columns[index].id).toBe(id);
                    expect(columns[index].header).toBe(expectedHeaders[index]);
                });
            }
        );
    });

    describe("Name Column Sorting", () => {
        const columns = OutputTableColumns(false);
        const sortingFn = getSortingFn(columns[0]);

        it.each([
            {
                description: "names in ascending order",
                rowA: { name: "customer_id", colType: "integer" },
                rowB: { name: "order_date", colType: "timestamp" },
                expected: -1
            },
            {
                description: "names in descending order",
                rowA: { name: "total_amount", colType: "decimal" },
                rowB: { name: "customer_id", colType: "integer" },
                expected: 1
            },
            {
                description: "0 for equal names",
                rowA: { name: "user_id", colType: "integer" },
                rowB: { name: "user_id", colType: "integer" },
                expected: 0
            }
        ])("Should sort $description", ({ rowA, rowB, expected }) => {
            expect(sortingFn(createRow(rowA), createRow(rowB), "name")).toBe(
                expected
            );
        });

        it("Should handle null original gracefully", () => {
            const rowA = createRow(null);
            const rowB = createRow({ name: "product_name", colType: "string" });

            expect(sortingFn(rowA, rowB, "name")).toBe(1);
        });
    });

    describe("Samples Column Sorting", () => {
        const columns = OutputTableColumns(true);
        const sortingFn = getSortingFn(columns[2]);

        it.each([
            {
                description: "by name in descending order",
                rowA: {
                    name: "customer_id",
                    colType: "integer",
                    samples: [1, 2, 3]
                },
                rowB: {
                    name: "order_total",
                    colType: "decimal",
                    samples: [100.5, 200.75, 300.25]
                },
                expected: 1
            },
            {
                description: "by name in ascending order",
                rowA: {
                    name: "total_price",
                    colType: "decimal",
                    samples: [99.99, 149.99, 199.99]
                },
                rowB: {
                    name: "customer_id",
                    colType: "integer",
                    samples: [1001, 1002, 1003]
                },
                expected: -1
            },
            {
                description: "0 for equal names",
                rowA: {
                    name: "transaction_id",
                    colType: "string",
                    samples: ["TXN001", "TXN002", "TXN003"]
                },
                rowB: {
                    name: "transaction_id",
                    colType: "string",
                    samples: ["TXN004", "TXN005", "TXN006"]
                },
                expected: 0
            }
        ])("Should sort $description", ({ rowA, rowB, expected }) => {
            expect(sortingFn(createRow(rowA), createRow(rowB), "samples")).toBe(
                expected
            );
        });

        it("Should handle null original gracefully", () => {
            const rowA = createRow(null);
            const rowB = createRow({
                name: "status_code",
                colType: "integer",
                samples: [200, 404, 500]
            });

            expect(sortingFn(rowA, rowB, "samples")).toBe(1);
        });
    });

    describe("Cell Rendering", () => {
        describe("Name Column", () => {
            const columns = OutputTableColumns(false);
            const nameColumn = columns[0];

            it("Should render column name", () => {
                const row = createRow({ name: "user_id", colType: "integer" });
                const result = renderCell(nameColumn, row);

                expect(result.props.children).toBe("user_id");
            });

            it("Should handle null row gracefully", () => {
                const row = createRow(null);
                const result = renderCell(nameColumn, row);

                expect(result).toBeDefined();
            });
        });

        describe("ColType Column", () => {
            const columns = OutputTableColumns(false);
            const colTypeColumn = columns[1];

            it("Should render column type", () => {
                const row = createRow({ name: "amount", colType: "decimal" });
                const result = renderCell(colTypeColumn, row);

                expect(result.props.children).toBe("decimal");
            });

            it("Should handle null row gracefully", () => {
                const row = createRow(null);
                const result = renderCell(colTypeColumn, row);

                expect(result).toBeDefined();
            });
        });

        describe("Samples Column", () => {
            const columns = OutputTableColumns(true);
            const samplesColumn = columns[2];

            it.each([
                {
                    description: "samples array with numbers",
                    rowData: {
                        name: "price",
                        colType: "float",
                        samples: [10.5, 20.75, 30.25]
                    },
                    expected: "10.5, 20.75, 30.25"
                },
                {
                    description: "samples array with strings",
                    rowData: {
                        name: "category",
                        colType: "string",
                        samples: ["electronics", "books", "clothing"]
                    },
                    expected: "electronics, books, clothing"
                },
                {
                    description: "empty samples array",
                    rowData: {
                        name: "count",
                        colType: "integer",
                        samples: []
                    },
                    expected: ""
                }
            ])("Should render $description", ({ rowData, expected }) => {
                const row = createRow(rowData);
                const result = renderCell(samplesColumn, row);

                expect(result.props.children).toBe(expected);
            });
        });
    });
});
