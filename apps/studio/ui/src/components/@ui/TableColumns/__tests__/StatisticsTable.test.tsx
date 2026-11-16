// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { StatisticsTableColumns } from "../StatisticsTable";

const renderCell = (
    column: ReturnType<typeof StatisticsTableColumns>[number],
    row: {
        original: Record<string, unknown>;
        getValue?: () => unknown;
    }
) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    return render(
        <div>{column.cell({ row, getValue: row.getValue } as never)}</div>
    );
};

const renderHeader = (
    column: ReturnType<typeof StatisticsTableColumns>[number]
) => {
    const headerFn = column.header;
    return render(
        <div>
            {typeof headerFn === "function" ? headerFn({} as never) : headerFn}
        </div>
    );
};

const createMockRow = (data: Record<string, unknown>) => ({
    original: data
});

const createMockRowWithGetter = (
    data: Record<string, unknown>,
    getValue: () => unknown
) => ({
    original: data,
    getValue
});

describe("StatisticsTableColumns", () => {
    describe("Column Configuration", () => {
        it("Should return an array of 6 column definitions", () => {
            const columns = StatisticsTableColumns();

            expect(columns).toBeInstanceOf(Array);
            expect(columns).toHaveLength(6);
        });

        it("Should have correct column IDs", () => {
            const columns = StatisticsTableColumns();
            const columnIds = columns.map((col) => col.id);

            expect(columnIds).toEqual([
                "name",
                "data_type",
                "stype",
                "statistics.counts.missing_percentage",
                "statistics.counts.invalid_percentage",
                "statistics.counts.total"
            ]);
        });

        it("Should have correct headers", () => {
            const columns = StatisticsTableColumns();

            expect(columns[0].header).toBeDefined();
            expect(columns[1].header).toBeDefined();
            expect(columns[2].header).toBeDefined();
            expect(columns[5].header).toBeDefined();

            const { container: missingContainer } = renderHeader(columns[3]);
            expect(missingContainer.textContent).toContain("Missing");

            const { container: invalidContainer } = renderHeader(columns[4]);
            expect(invalidContainer.textContent).toContain("Invalid");
        });

        it("Should have enableHiding set for Total Count column", () => {
            const columns = StatisticsTableColumns();

            expect(columns[5].enableHiding).toBe(true);
        });

        it("Should show tooltip content for Missing column header on hover", async () => {
            const columns = StatisticsTableColumns();
            renderHeader(columns[3]);

            const tooltipTrigger = document.querySelector(
                '[data-slot="tooltip-trigger"]'
            );
            await userEvent.hover(tooltipTrigger!);
            await waitFor(() => {
                const tooltipContent = screen.queryByRole("tooltip");
                expect(tooltipContent).toBeInTheDocument();
                expect(tooltipContent?.textContent).toContain(
                    "Missing values (NULL or empty) in the column"
                );
            });
        });

        it("Should show tooltip content for Invalid column header on hover", async () => {
            const columns = StatisticsTableColumns();
            renderHeader(columns[4]);

            const tooltipTrigger = document.querySelector(
                '[data-slot="tooltip-trigger"]'
            );
            await userEvent.hover(tooltipTrigger!);
            await waitFor(() => {
                const tooltipContent = screen.queryByRole("tooltip");
                expect(tooltipContent).toBeInTheDocument();
                expect(tooltipContent?.textContent).toContain(
                    "Invalid rows may be dropped due to"
                );
            });
        });
    });

    describe("Name Column", () => {
        const columns = StatisticsTableColumns();
        const nameColumn = columns[0];

        it.each([
            {
                description: "primary key column with Key icon",
                rowData: { name: "id", isPrimary: true },
                expectedName: "id",
                expectedIcon: "primary-key-icon"
            },
            {
                description: "time column with CalendarAdd icon",
                rowData: { name: "created_at", isTimeColumn: true },
                expectedName: "created_at",
                expectedIcon: "time-column-icon"
            },
            {
                description: "end time column with StopCircle icon",
                rowData: { name: "ended_at", isEndTimeColumn: true },
                expectedName: "ended_at",
                expectedIcon: "end-time-icon"
            }
        ])(
            "Should render $description",
            ({ rowData, expectedName, expectedIcon }) => {
                const mockRow = createMockRow(rowData);
                renderCell(nameColumn, mockRow);

                expect(screen.getByText(expectedName)).toBeInTheDocument();
                const icon = screen.getByTestId(expectedIcon);
                expect(icon).toBeInTheDocument();
            }
        );

        it("Should render regular column without icon", () => {
            const mockRow = createMockRow({ name: "email" });

            renderCell(nameColumn, mockRow);

            expect(screen.getByText("email")).toBeInTheDocument();
        });

        it("Should render header with margin", () => {
            const { container } = renderHeader(nameColumn);

            expect(screen.getByText("Column")).toBeInTheDocument();
            const span = container.querySelector("span");
            expect(span).toHaveClass("ml-2.5");
        });
    });

    describe("Data Type and Semantic Type Columns", () => {
        const columns = StatisticsTableColumns();
        const dtypeColumn = columns[1];
        const stypeColumn = columns[2];

        it("Should render dtype value", () => {
            const mockRow = createMockRowWithGetter({}, () => "String");

            renderCell(dtypeColumn, mockRow);

            expect(screen.getByText("String")).toBeInTheDocument();
        });

        it("Should replace 'Sequence' with 'Embedding' for both dtype and stype", () => {
            const mockRowDtype = createMockRowWithGetter({}, () => "Sequence");
            const { container: container1 } = renderCell(
                dtypeColumn,
                mockRowDtype
            );
            expect(screen.getByText("Embedding")).toBeInTheDocument();
            container1.remove();

            const mockRowStype = createMockRowWithGetter({}, () => "Sequence");
            renderCell(stypeColumn, mockRowStype);
            expect(screen.getByText("Embedding")).toBeInTheDocument();
            expect(screen.queryByText("Sequence")).not.toBeInTheDocument();
        });

        it("Should have sorting functions for dtype and stype columns", () => {
            expect(dtypeColumn.sortingFn).toBeDefined();
            expect(typeof dtypeColumn.sortingFn).toBe("function");
            expect(stypeColumn.sortingFn).toBeDefined();
            expect(typeof stypeColumn.sortingFn).toBe("function");
        });

        it.each([
            {
                description: "dtype",
                column: "dtypeColumn",
                rowAData: { dtype: "String" },
                rowAValue: "String",
                rowBData: { dtype: "Integer" },
                rowBValue: "Integer"
            },
            {
                description: "stype",
                column: "stypeColumn",
                rowAData: { stype: "Categorical" },
                rowAValue: "Categorical",
                rowBData: { stype: "Numerical" },
                rowBValue: "Numerical"
            }
        ])(
            "Should execute $description sorting function correctly",
            ({ column, rowAData, rowAValue, rowBData, rowBValue }) => {
                const mockRowA = createMockRowWithGetter(
                    rowAData,
                    () => rowAValue
                );
                const mockRowB = createMockRowWithGetter(
                    rowBData,
                    () => rowBValue
                );

                const targetColumn =
                    column === "dtypeColumn" ? dtypeColumn : stypeColumn;
                const sortingFn = targetColumn.sortingFn;
                if (typeof sortingFn === "function") {
                    const result = sortingFn(
                        mockRowA as never,
                        mockRowB as never,
                        ""
                    );
                    expect(typeof result).toBe("number");
                }
            }
        );

        it("Should handle different data types correctly", () => {
            const testCases = [
                { input: "String", expected: "String" },
                { input: "Integer", expected: "Integer" },
                { input: "Float", expected: "Float" },
                { input: "Boolean", expected: "Boolean" }
            ];

            testCases.forEach(({ input, expected }) => {
                const mockRow = createMockRowWithGetter({}, () => input);
                const { container } = renderCell(dtypeColumn, mockRow);
                expect(screen.getByText(expected)).toBeInTheDocument();
                container.remove();
            });
        });
    });

    describe("Missing Percentage Column", () => {
        const columns = StatisticsTableColumns(true);
        const missingColumn = columns[3];

        it.each([
            {
                description: "green color when < 10%",
                value: 5,
                expectedColor: "#059669"
            },
            {
                description: "yellow color when >= 10% and < 51%",
                value: 25,
                expectedColor: "#D97706"
            },
            {
                description: "red color when >= 51%",
                value: 75,
                expectedColor: "#DC2626"
            }
        ])("Should render with $description", ({ value, expectedColor }) => {
            const mockRow = createMockRow({
                statistics: { counts: { missing_percentage: value } }
            });

            const { container } = renderCell(missingColumn, mockRow);

            const coloredDiv = container.querySelector("span > div");
            expect(coloredDiv).toHaveStyle({ color: expectedColor });
            expect(coloredDiv?.textContent).toContain("%");
        });

        it("Should render emdash when statistics unavailable", () => {
            const columnsWithoutStats = StatisticsTableColumns(false);
            const missingColumnWithoutStats = columnsWithoutStats[3];

            const mockRow = createMockRow({
                statistics: { counts: { missing_percentage: 5 } }
            });

            const { container } = renderCell(
                missingColumnWithoutStats,
                mockRow
            );

            expect(container.querySelector("label")).not.toBeInTheDocument();
        });

        it("Should render emdash when value is null", () => {
            const mockRow = createMockRow({
                statistics: { counts: { missing_percentage: null } }
            });

            const { container } = renderCell(missingColumn, mockRow);

            expect(container.querySelector("label")).not.toBeInTheDocument();
        });

        it("Should handle boundary values (0%, 10%, 51%)", () => {
            const cellFn = missingColumn.cell;

            const testCases = [
                { value: 0, expectedColor: "#059669" },
                { value: 10, expectedColor: "#D97706" },
                { value: 51, expectedColor: "#DC2626" }
            ];

            testCases.forEach(({ value, expectedColor }) => {
                const mockRow = createMockRow({
                    statistics: { counts: { missing_percentage: value } }
                });
                const { container } = render(
                    <div>
                        {typeof cellFn === "function"
                            ? cellFn({ row: mockRow } as never)
                            : null}
                    </div>
                );
                const coloredDiv = container.querySelector("span > div");
                expect(coloredDiv).toHaveStyle({ color: expectedColor });
                container.remove();
            });
        });

        it("Should have sorting function", () => {
            expect(missingColumn.sortingFn).toBeDefined();
            expect(typeof missingColumn.sortingFn).toBe("function");
        });

        it("Should execute sorting function correctly", () => {
            const mockRowA = createMockRowWithGetter(
                { statistics: { counts: { missing_percentage: 25 } } },
                () => 25
            );
            const mockRowB = createMockRowWithGetter(
                { statistics: { counts: { missing_percentage: 50 } } },
                () => 50
            );

            const sortingFn = missingColumn.sortingFn;
            if (typeof sortingFn === "function") {
                const result = sortingFn(
                    mockRowA as never,
                    mockRowB as never,
                    ""
                );
                expect(typeof result).toBe("number");
            }
        });
    });

    describe("Invalid Percentage Column", () => {
        const columns = StatisticsTableColumns(true);
        const invalidColumn = columns[4];

        it.each([
            {
                description: "green color when < 10%",
                value: 3,
                expectedColor: "#059669"
            },
            {
                description: "yellow color when >= 10% and < 51%",
                value: 30,
                expectedColor: "#D97706"
            },
            {
                description: "red color when >= 51%",
                value: 80,
                expectedColor: "#DC2626"
            }
        ])("Should render with $description", ({ value, expectedColor }) => {
            const mockRow = createMockRow({
                statistics: { counts: { invalid_percentage: value } }
            });

            const { container } = renderCell(invalidColumn, mockRow);

            const coloredDiv = container.querySelector("span > div");
            expect(coloredDiv).toHaveStyle({ color: expectedColor });
            expect(coloredDiv?.textContent).toContain("%");
        });

        it("Should render emdash when statistics unavailable", () => {
            const columnsWithoutStats = StatisticsTableColumns(false);
            const invalidColumnWithoutStats = columnsWithoutStats[4];

            const mockRow = createMockRow({
                statistics: { counts: { invalid_percentage: 5 } }
            });

            const { container } = renderCell(
                invalidColumnWithoutStats,
                mockRow
            );

            expect(container.querySelector("label")).not.toBeInTheDocument();
        });

        it("Should render emdash when value is undefined", () => {
            const mockRow = createMockRow({
                statistics: { counts: { invalid_percentage: undefined } }
            });

            const { container } = renderCell(invalidColumn, mockRow);

            expect(container.querySelector("label")).not.toBeInTheDocument();
        });

        it("Should have sorting function", () => {
            expect(invalidColumn.sortingFn).toBeDefined();
            expect(typeof invalidColumn.sortingFn).toBe("function");
        });

        it("Should execute sorting function correctly", () => {
            const mockRowA = createMockRowWithGetter(
                { statistics: { counts: { invalid_percentage: 15 } } },
                () => 15
            );
            const mockRowB = createMockRowWithGetter(
                { statistics: { counts: { invalid_percentage: 35 } } },
                () => 35
            );

            const sortingFn = invalidColumn.sortingFn;
            if (typeof sortingFn === "function") {
                const result = sortingFn(
                    mockRowA as never,
                    mockRowB as never,
                    ""
                );
                expect(typeof result).toBe("number");
            }
        });
    });

    describe("Total Count Column", () => {
        const columns = StatisticsTableColumns(true);
        const totalColumn = columns[5];

        it("Should render total count when statistics available and value is valid", () => {
            const mockRow = createMockRow({
                statistics: { counts: { total: 1000 } }
            });

            renderCell(totalColumn, mockRow);

            expect(screen.getByText("1000")).toBeInTheDocument();
        });

        it("Should render emdash when statistics unavailable", () => {
            const columnsWithoutStats = StatisticsTableColumns(false);
            const totalColumnWithoutStats = columnsWithoutStats[5];

            const mockRow = createMockRow({
                statistics: { counts: { total: 1000 } }
            });

            renderCell(totalColumnWithoutStats, mockRow);

            expect(screen.queryByText("1000")).not.toBeInTheDocument();
        });

        it("Should render emdash when value is null", () => {
            const mockRow = createMockRow({
                statistics: { counts: { total: null } }
            });

            const { container } = renderCell(totalColumn, mockRow);

            expect(container.querySelector("label")).not.toBeInTheDocument();
        });
    });

    describe("Edge Cases", () => {
        const columns = StatisticsTableColumns(true);
        const missingColumn = columns[3];

        it("Should handle missing statistics object", () => {
            const mockRow = createMockRow({});

            const { container } = renderCell(missingColumn, mockRow);

            expect(container.querySelector("label")).not.toBeInTheDocument();
        });

        it("Should handle empty counts object", () => {
            const mockRow = createMockRow({
                statistics: { counts: {} }
            });

            const { container } = renderCell(missingColumn, mockRow);

            expect(container.querySelector("label")).not.toBeInTheDocument();
        });
    });
});
