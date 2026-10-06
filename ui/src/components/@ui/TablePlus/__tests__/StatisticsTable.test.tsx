// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { hoverOverStatus, verifyTextExists } from "tests/testHelpers";

import { ExtendedColStatsResponse } from "types/snapshots";

import StatisticsTable, { StatisticsTableProps } from "../StatisticsTable";

import {
    column_stats,
    sampleRowsWithInvalidValue
} from "./MockResponse/ColumnStatsResponse";

const defaultProps: StatisticsTableProps = {
    columnStats: null,
    loading: false,
    search: false,
    gotError: false,
    hiddenColumns: [],
    primaryKey: null,
    timeCol: null,
    endTimeCol: null,
    isSnapShot: false
};

describe("StatisticsTable", () => {
    const renderStatisticsTableNew = async (props: StatisticsTableProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <StatisticsTable {...props} />
                </MemoryRouter>
            );
        });
    };

    const getTable = () => {
        const table = document.getElementById("column-stats-table");
        expect(table).toBeInTheDocument();
        return table!;
    };

    const getDataRows = (table: HTMLElement) => {
        return table.querySelectorAll('tbody [data-slot="table-row"]');
    };

    const getRowByColumnName = (columnName: string) => {
        const row = screen.getByText(columnName).closest("tr");
        expect(row).toBeInTheDocument();
        return row!;
    };

    const clickRowAndVerifyStats = async (
        columnName: string,
        expectedCount: string | number,
        expectedUnique: string | number
    ) => {
        const row = getRowByColumnName(columnName);
        await userEvent.click(row);

        expect(screen.getByTestId("stats")).toBeInTheDocument();
        expect(screen.getByTestId("count")).toHaveTextContent(
            String(expectedCount)
        );
        expect(screen.getByTestId("unique")).toHaveTextContent(
            String(expectedUnique)
        );
    };

    it("Should render expected column headers in the statistics table", async () => {
        await renderStatisticsTableNew({
            ...defaultProps
        });

        getTable();

        const header = screen.getByTestId("table-header");
        const headerCells = within(header).getAllByRole("columnheader");

        const headerText = headerCells.map((th) => th.textContent?.trim());
        const expectedHeaders = [
            "Column",
            "Data Type",
            "Semantic Type",
            "Missing",
            "Invalid",
            "Total Count"
        ];
        expect(headerText).toEqual(expect.arrayContaining(expectedHeaders));
    });

    it("Should display tooltip message for data type and semantic column", async () => {
        await renderStatisticsTableNew({
            ...defaultProps,
            columnStats: column_stats,
            hiddenColumns: ["statistics.counts.total"]
        });

        getTable();

        await hoverOverStatus("Data Type");

        await waitFor(() =>
            expect(
                screen.getAllByText(
                    "The Data Type represents how the data is physically stored"
                ).length
            ).toBeGreaterThan(0)
        );

        await hoverOverStatus("Semantic Type");

        await waitFor(() =>
            expect(
                screen.getAllByText(
                    "The Semantic Type determines how data is processed and encoded"
                ).length
            ).toBeGreaterThan(0)
        );
    });

    it("Should display 'No data found.' message when columnStats are empty", async () => {
        await renderStatisticsTableNew({
            ...defaultProps,
            columnStats: null
        });

        getTable();
        await verifyTextExists("No data found.");
    });

    it("Should render correct values for each column in the table", async () => {
        await renderStatisticsTableNew({
            ...defaultProps,
            columnStats: column_stats
        });

        const table = getTable();
        const dataRows = getDataRows(table);

        expect(dataRows.length).toBe(column_stats.columns.length);

        dataRows.forEach((row, rowIndex) => {
            const cells = row.querySelectorAll('[data-slot="table-cell"]');
            const col = column_stats.columns[rowIndex];

            expect(cells[0]).toHaveTextContent(col.name);
            expect(cells[1]).toHaveTextContent(col.dtype);
            expect(cells[2]).toHaveTextContent(col.stype);

            const missing = col?.statistics?.counts.missing_percentage;
            expect(cells[3]).toHaveTextContent(String(missing ?? ""));

            const invalid = col?.statistics?.counts.invalid_percentage;
            expect(cells[4]).toHaveTextContent(String(invalid ?? ""));

            const total = col?.statistics?.counts.total;
            expect(cells[5]).toHaveTextContent(String(total ?? ""));
        });
    });

    it.each([
        {
            search: false,
            shouldShow: false,
            description:
                "Should not display the search box if search prop is false"
        },
        {
            search: true,
            shouldShow: true,
            description: "Should display the search box if search prop is true"
        }
    ])("$description", async ({ search, shouldShow }) => {
        await renderStatisticsTableNew({
            ...defaultProps,
            columnStats: column_stats,
            search
        });

        const inputByPlaceholder = screen.queryByPlaceholderText("Search");
        const inputByDataSlot = document.querySelector('[data-slot="input"]');

        if (shouldShow) {
            expect(inputByPlaceholder).toBeInTheDocument();
            expect(inputByDataSlot).not.toBeNull();
        } else {
            expect(inputByPlaceholder).not.toBeInTheDocument();
            expect(inputByDataSlot).toBeNull();
        }
    });

    it("Should show loading skeletons inside each table cell when loading is true", async () => {
        await renderStatisticsTableNew({
            ...defaultProps,
            loading: true
        });

        const table = getTable();
        const dataCells = table.querySelectorAll(
            'tbody [data-slot="table-cell"]'
        );

        dataCells.forEach((cell) => {
            const skeleton = cell.querySelector('[data-slot="skeleton"]');
            expect(skeleton).toBeInTheDocument();
        });
    });

    it("Should show an error when the gotError prop is set to true", async () => {
        await renderStatisticsTableNew({
            ...defaultProps,
            gotError: true
        });

        await verifyTextExists(
            "Statistics for this table are not available. To fix this, try running a new set of predictions. If this doesn't work, you may contact your POC at Kumo to investigate further."
        );
    });

    it("Should display approximate statistics message", async () => {
        await renderStatisticsTableNew({
            ...defaultProps,
            columnStats: {
                ...column_stats,
                approx: true,
                rows_used: 10000
            }
        });

        await verifyTextExists(
            "Statistics are approximate and based on 10,000 sampled rows."
        );
    });

    it.each([
        {
            description:
                "Should render the primary key icon when primaryKey is set",
            propName: "primaryKey" as const,
            testId: "primary-key-icon"
        },
        {
            description:
                "Should render the time column icon when timeCol is set",
            propName: "timeCol" as const,
            testId: "time-column-icon"
        },
        {
            description:
                "Should render the end-time column icon when endTimeCol is set",
            propName: "endTimeCol" as const,
            testId: "end-time-icon"
        }
    ])("$description", async ({ propName, testId }) => {
        await renderStatisticsTableNew({
            ...defaultProps,
            columnStats: column_stats,
            [propName]: "CustomerID"
        });

        const row = getRowByColumnName("CustomerID");

        expect(
            row.querySelector(`[data-testid="${testId}"]`)
        ).toBeInTheDocument();
    });

    it.each([
        {
            description:
                "Should not render the primary key icon when primaryKey does not match",
            propName: "primaryKey" as const,
            testId: "primary-key-icon"
        },
        {
            description:
                "Should not render the time column icon when timeCol does not match",
            propName: "timeCol" as const,
            testId: "time-column-icon"
        },
        {
            description:
                "Should not render the end-time column icon when endTimeCol does not match",
            propName: "endTimeCol" as const,
            testId: "end-time-icon"
        }
    ])("$description", async ({ propName, testId }) => {
        await renderStatisticsTableNew({
            ...defaultProps,
            columnStats: column_stats,
            [propName]: "NonMatchingColumn"
        });

        const icon = screen.queryByTestId(testId);
        expect(icon).not.toBeInTheDocument();
    });

    it("Should hide the 'Column' header when hiddenColumns is set", async () => {
        await renderStatisticsTableNew({
            ...defaultProps,
            columnStats: column_stats,
            hiddenColumns: ["name"]
        });

        const headers = screen.queryAllByRole("columnheader");

        const columnHeader = headers.find((header) =>
            header.textContent?.trim().includes("Column")
        );

        expect(columnHeader).toBeUndefined();
    });

    it("Should display list of valid samples values", async () => {
        await renderStatisticsTableNew({
            ...defaultProps,
            columnStats: column_stats
        });

        await clickRowAndVerifyStats(
            "CustomerID",
            column_stats.columns[0]?.statistics?.counts?.total ?? "",
            column_stats.columns[0]?.statistics?.counts?.num_unique ?? ""
        );

        expect(screen.getByTestId("sample")).toBeInTheDocument();
        expect(screen.getByTestId("sample")).not.toHaveTextContent("N/A");

        const expectedText =
            "Sample" +
            column_stats.columns[0].sample_rows
                ?.map((v) => (v === null || v === undefined ? "N/A" : v))
                .join("");

        expect(screen.getByTestId("sample")).toHaveTextContent(expectedText);
    });

    it("Should not display samples values when `isSnapshot` is true", async () => {
        await renderStatisticsTableNew({
            ...defaultProps,
            columnStats: column_stats,
            isSnapShot: true
        });

        const row = getRowByColumnName("CustomerID");
        await userEvent.click(row);

        expect(screen.queryByTestId("sample")).not.toBeInTheDocument();
    });

    it("Should display null values as N/A under Samples", async () => {
        await renderStatisticsTableNew({
            ...defaultProps,
            columnStats:
                sampleRowsWithInvalidValue as unknown as ExtendedColStatsResponse
        });

        await clickRowAndVerifyStats(
            "CustomerID",
            sampleRowsWithInvalidValue.columns[0].statistics.counts.total,
            sampleRowsWithInvalidValue.columns[0].statistics.counts.num_unique
        );

        expect(screen.getByTestId("sample")).toBeInTheDocument();
        expect(screen.getByTestId("sample")).toHaveTextContent("N/A");
    });
});
