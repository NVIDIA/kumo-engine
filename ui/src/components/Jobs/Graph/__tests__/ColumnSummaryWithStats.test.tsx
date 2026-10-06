// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import mockGraphSnapshotReponse from "tests/fixtures/GraphSnapshot.json";
import { mockAxiosInstance } from "tests/mockAxios";
import { verifyTextExists } from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { setColStatsResponseType } from "pages/Tables/TablesList/__tests__/MockFunction/Colstats.test.mock";

import { ColumnStatsdownOption } from "types/inputs";

import { tableIdIsSnapshot } from "utils/Helpers/typeGuards";

import ColumnSummaryWithStats, {
    ColumnSummaryWithStatsProps
} from "../ColumnSummaryWithStats";

const defaultProps: ColumnStatsdownOption[] = [
    {
        value: mockGraphSnapshotReponse.snapshot_resource.table_ids
            .FINANCE_TRANS,
        label: "FINANCE_TRANS",
        pkey: mockGraphSnapshotReponse.definition.tables.FINANCE_TRANS.pkey,
        timeCol:
            mockGraphSnapshotReponse.definition.tables.FINANCE_TRANS.time_col,
        endTimeCol:
            mockGraphSnapshotReponse.definition.tables.FINANCE_TRANS
                .end_time_col
    },
    {
        value: mockGraphSnapshotReponse.snapshot_resource.table_ids
            .FINANCE_ORDERS,
        label: "FINANCE_ORDERS",
        pkey: mockGraphSnapshotReponse.definition.tables.FINANCE_ORDERS.pkey,
        timeCol:
            mockGraphSnapshotReponse.definition.tables.FINANCE_ORDERS.time_col,
        endTimeCol:
            mockGraphSnapshotReponse.definition.tables.FINANCE_ORDERS
                .end_time_col
    }
];

describe("ColumnSummaryWithStats", () => {
    const queryClient = createQueryClient();

    const renderColumnSummaryWithStats = async (
        props: ColumnSummaryWithStatsProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <ColumnSummaryWithStats {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const selectTableFromDropdown = async (tableName: string) => {
        const dropdown = screen.getByTestId("testid-select-table");
        await userEvent.click(dropdown);
        const option = await screen.findByText(tableName);
        await userEvent.click(option);
    };

    const expandRowByIndex = async (index: number) => {
        const allRowExpand = await screen.findAllByTestId("chevron-right");
        await userEvent.click(allRowExpand[index]);
    };

    const closeExpandedRow = async (index: number) => {
        const allCloseButtons = screen.getAllByTestId("chevron-down");
        await userEvent.click(allCloseButtons[index]);
    };

    const verifyStatsCard = async () => {
        const statsCard = screen.getByTestId("stats");
        await waitFor(() => {
            expect(statsCard).toHaveTextContent("Stats");
        });
        return statsCard;
    };

    const verifyDistributionCard = async () => {
        const distributionCard = screen.getByTestId("distribution");
        await waitFor(() => {
            expect(distributionCard).toHaveTextContent("Distribution");
        });
        return distributionCard;
    };

    const waitForApiCall = async (tableId: string) => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/snapshots/table/${tableId}/col_stats`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    it("Should not render `ColumnSummaryWithStats` when the prop is set Empty", async () => {
        await renderColumnSummaryWithStats({ tables: [] });

        expect(screen.queryByText("Column Statistics")).not.toBeInTheDocument();
    });

    it("Should make API call with correct table ID when component mounts", async () => {
        await renderColumnSummaryWithStats({
            tables: defaultProps
        });

        const tableId = "table-bc150313117da0324bfe31ca4ff499e5@1664307208";
        await waitForApiCall(tableId);

        await verifyTextExists("Column Statistics");
    });

    it("Should display correct column headers in statistics table", async () => {
        await renderColumnSummaryWithStats({
            tables: defaultProps
        });

        const header = screen.getByTestId("table-header");
        const headerCells = within(header).getAllByRole("columnheader");

        const headerText = headerCells.map((th) => th.textContent?.trim());
        const expectedHeaders = [
            "Column",
            "Data Type",
            "Semantic Type",
            "Missing",
            "Invalid"
        ];
        expect(headerText).toEqual(expect.arrayContaining(expectedHeaders));
    });

    it("Should display table selection dropdown with correct options", async () => {
        await renderColumnSummaryWithStats({
            tables: defaultProps
        });

        const dropdown = screen.getByTestId("testid-select-table");
        expect(dropdown).toBeInTheDocument();
        await verifyTextExists("FINANCE_TRANS");
    });

    it("Should make new API call when selecting different table from dropdown", async () => {
        await renderColumnSummaryWithStats({
            tables: defaultProps
        });

        await verifyTextExists("FINANCE_TRANS");

        await selectTableFromDropdown("FINANCE_ORDERS");

        const tableId = "NA.NA-ccaf9a4a904985c04cd0be0c96d6857c";
        await waitForApiCall(tableId);
    });

    it("Should render expanded column stats for ID column and hides sample rows for snapshot table IDs", async () => {
        await renderColumnSummaryWithStats({
            tables: defaultProps
        });

        const tableId = "table-bc150313117da0324bfe31ca4ff499e5@1664307208";
        const isSnapshotId = tableIdIsSnapshot(tableId);

        await expandRowByIndex(0);

        const statsCard = await verifyStatsCard();
        await waitFor(() => {
            expect(screen.getByTestId("count")).toHaveTextContent("10,000");
            expect(screen.getByTestId("unique")).toHaveTextContent("10,000");
            expect(screen.getByTestId("missing")).toHaveTextContent("0%");
        });

        if (isSnapshotId) {
            expect(screen.queryByTestId("sample")).not.toBeInTheDocument();
        } else {
            const sampleCard = screen.getByTestId("sample");
            expect(sampleCard).toBeInTheDocument();

            expect(within(sampleCard).getByText("Sample")).toBeInTheDocument();

            const sampleValues = within(sampleCard).queryAllByText(/\d+/);
            expect(sampleValues.length).toBe(0);
        }

        await closeExpandedRow(1);

        await waitFor(() => {
            expect(statsCard).not.toBeInTheDocument();
        });
    });

    it("Should display expanded statistics view for numerical column with distribution chart", async () => {
        await renderColumnSummaryWithStats({
            tables: defaultProps
        });

        await expandRowByIndex(5);

        const statsCard = await verifyStatsCard();
        const distributionCard = await verifyDistributionCard();

        await closeExpandedRow(1);

        await waitFor(() => {
            expect(statsCard).not.toBeInTheDocument();
            expect(distributionCard).not.toBeInTheDocument();
        });
    });

    it("Should display expanded statistics view for categorical column with distribution data", async () => {
        await renderColumnSummaryWithStats({
            tables: defaultProps
        });

        await expandRowByIndex(4);

        const statsCard = await verifyStatsCard();
        const distributionCard = await verifyDistributionCard();

        await closeExpandedRow(1);

        await waitFor(() => {
            expect(statsCard).not.toBeInTheDocument();
            expect(distributionCard).not.toBeInTheDocument();
        });
    });

    it("Should handle empty sample rows by showing empty sample section", async () => {
        await renderColumnSummaryWithStats({
            tables: defaultProps
        });

        await selectTableFromDropdown("FINANCE_ORDERS");

        const tableId = "NA.NA-ccaf9a4a904985c04cd0be0c96d6857c";
        await waitForApiCall(tableId);

        await expandRowByIndex(4);

        await verifyStatsCard();

        await waitFor(() => {
            const sampleCard = screen.getByTestId("sample");
            expect(sampleCard).toBeInTheDocument();

            expect(within(sampleCard).getByText("Sample")).toBeInTheDocument();

            const sampleValues = within(sampleCard).queryAllByText(/\d+/);
            expect(sampleValues.length).toBe(0);
        });
    });

    it("Should display basic stats without distribution when statistics_available is false", async () => {
        await renderColumnSummaryWithStats({
            tables: defaultProps
        });

        await selectTableFromDropdown("FINANCE_ORDERS");

        const tableId = "NA.NA-ccaf9a4a904985c04cd0be0c96d6857c";
        await waitForApiCall(tableId);

        await expandRowByIndex(4);

        await verifyStatsCard();

        const distributionCard = screen.queryByTestId("distribution");
        expect(distributionCard).not.toBeInTheDocument();
    });

    it("Should show sample section but empty when statistics_available is false", async () => {
        await renderColumnSummaryWithStats({
            tables: defaultProps
        });

        await selectTableFromDropdown("FINANCE_ORDERS");

        const tableId = "NA.NA-ccaf9a4a904985c04cd0be0c96d6857c";
        const isSnapshotId = tableIdIsSnapshot(tableId);

        await waitForApiCall(tableId);

        await expandRowByIndex(2);

        await verifyStatsCard();

        const distributionCard = screen.queryByTestId("distribution");
        expect(distributionCard).not.toBeInTheDocument();

        if (isSnapshotId) {
            expect(screen.queryByTestId("sample")).not.toBeInTheDocument();
        } else {
            const sampleCard = screen.getByTestId("sample");
            expect(sampleCard).toBeInTheDocument();

            expect(within(sampleCard).getByText("Sample")).toBeInTheDocument();

            const sampleValues = within(sampleCard).queryAllByText(/\d+/);
            expect(sampleValues.length).toBe(0);
        }
    });

    it("Should display 'No data found' message when API returns an error response", async () => {
        setColStatsResponseType(testResponseType.ERROR);
        await renderColumnSummaryWithStats({
            tables: defaultProps
        });

        await verifyTextExists("Column Statistics");
        await verifyTextExists("No data found.");
        setColStatsResponseType(testResponseType.ALL);
    });
});
