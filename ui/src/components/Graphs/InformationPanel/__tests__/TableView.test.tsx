// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { waitFor } from "@testing-library/dom";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { tables } from "tests/fixtures/MockGraphInformationPanel";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import {
    verifyElementDoesNotExist,
    verifyElementExists
} from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { setTestResponseType } from "pages/Tables/TablesList/__tests__/MockFunction/TableDetail.test.mock";
import { mockGetTablesDetailStock } from "pages/Tables/TablesList/__tests__/MockResponse/TableDetail.test.response";

import { JSTable } from "types/tables";

import { getProjectPath } from "utils/Constants";
import { formatToShortLocalDateString } from "utils/timeConverter";

import TableView, { TableViewProps } from "../TableView";

const defaultProps: TableViewProps = {
    tables: [],
    selectedTable: "",
    navigateToGraph: () => {},
    open: true,
    tableIds: {}
};

describe("TableView", () => {
    const queryClient = createQueryClient();
    const renderTableView = async (props: TableViewProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <TableView {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
        mockNavigateCall();
    });

    afterEach(async () => {
        vi.clearAllMocks();
    });

    const checkApiCalls = async ({
        selectedTable,
        connectorName
    }: {
        selectedTable: string;
        connectorName: string;
    }) => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/tables/${selectedTable}`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/tables/${selectedTable}/col_stats`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/projects/project-tests/connectors/${connectorName}`
            );
        });
    };

    const findHoverTriggerByAttribute = (text: string) => {
        return screen
            .getAllByText(text)
            .find(
                (el) => el.getAttribute("data-slot") === "hover-card-trigger"
            );
    };

    const hoverAndVerifyState = async (
        element: Element,
        expectedState: "open" | "closed"
    ) => {
        await userEvent.hover(element);
        await waitFor(() => {
            const state = element.getAttribute("data-state");
            // Accept both "open" and "delayed-open" as valid open states for tooltips
            if (expectedState === "open") {
                expect(state === "open" || state === "delayed-open").toBe(true);
            } else {
                expect(element).toHaveAttribute("data-state", expectedState);
            }
        });
    };

    it("Should render TableView elements when open is true", async () => {
        await renderTableView(defaultProps);
        const backButton = screen.getByRole("button", {
            name: "Back"
        });
        verifyElementExists(backButton);
        const labelText = screen.getByText("Column Statistics");
        verifyElementExists(labelText);
    });

    it("Should not render TableView elements when open is false", async () => {
        await renderTableView({ ...defaultProps, open: false });

        verifyElementDoesNotExist(screen.queryByText("Column Statistics"));
        expect(document.body.innerHTML).toBe("<div></div>");
    });

    it("Should render column-stats-table with correct headings", async () => {
        await renderTableView(defaultProps);
        const labelText = screen.getByText("Column Statistics");
        verifyElementExists(labelText);

        verifyElementExists(
            screen.getByText(
                "The following stats are computed from a sample of table data."
            )
        );

        const statsTable = document.getElementById("column-stats-table");
        verifyElementExists(statsTable);

        const tableHeadings = [
            "Column",
            "Data Type",
            "Semantic Type",
            "Missing",
            "Invalid"
        ];

        for (const heading of tableHeadings) {
            verifyElementExists(screen.getByText(heading));
        }
    });

    it("Should display 'No data found.' when tables prop is empty", async () => {
        setTestResponseType(testResponseType.ERROR);
        await renderTableView(defaultProps);

        const noDataMessage = screen.getByText("No data found.");
        verifyElementExists(noDataMessage);

        const tbody = noDataMessage.closest("tbody");
        verifyElementExists(tbody);
        expect(tbody).toContainElement(noDataMessage);
        setTestResponseType();
    });

    it("Should display hover card content when hovering over month info", async () => {
        const selectedTable = mockGetTablesDetailStock.name_alias;
        const connectorName = mockGetTablesDetailStock.connector.id;
        const monthUpdate = formatToShortLocalDateString(
            mockGetTablesDetailStock?.timestamps?.created_at
        );

        await renderTableView({
            ...defaultProps,
            tables: tables,
            selectedTable: selectedTable,
            tableIds: {
                stock: "stock"
            }
        });

        await checkApiCalls({ selectedTable, connectorName });
        const hoverTrigger = screen.getByText(monthUpdate);
        expect(hoverTrigger).toHaveAttribute("data-state", "closed");
        expect(hoverTrigger).toHaveAttribute("data-testid", "Created");
        await hoverAndVerifyState(hoverTrigger, "open");

        const tooltipContent = await screen.findByRole(
            "tooltip",
            {},
            { timeout: 3000 }
        );
        expect(tooltipContent).toBeInTheDocument();

        expect(tooltipContent).toHaveTextContent("Last edited");
        expect(tooltipContent).toHaveTextContent("Created");
    });

    it("Should display column info without hover card", async () => {
        const selectedTable = "stock";
        const connectorName = "ltv_test";
        const expectedSize = "3.6 MB";
        await renderTableView({
            ...defaultProps,
            tables,
            selectedTable,
            tableIds: { stock: "stock" }
        });

        await checkApiCalls({ selectedTable, connectorName });

        await waitFor(
            () => {
                const columnInfoElements = screen.getAllByText(
                    (_content, element) => {
                        const text = element?.textContent ?? "";
                        return (
                            text.includes("Columns") &&
                            text.includes(expectedSize) &&
                            Boolean(
                                element?.classList?.contains("items-center")
                            )
                        );
                    }
                );
                expect(columnInfoElements.length).toBeGreaterThan(0);
            },
            { timeout: 3000 }
        );
    });

    it("Should call snapshot API when tableId consists '@'", async () => {
        const tableId = "abc@123";
        const snapshotTable = {
            ...tables[0],
            id: tableId
        };

        await renderTableView({
            ...defaultProps,
            tables: [snapshotTable],
            selectedTable: tableId
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/snapshots/table/${tableId}/col_stats`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });
    });

    it("Should display hover card content when hovering over connector info", async () => {
        const selectedTable = "stock";
        const connectorName = "ltv_test";
        const sourceTable = "transaction";

        await renderTableView({
            ...defaultProps,
            tables,
            selectedTable: "stock",
            tableIds: { stock: "stock" }
        });

        await checkApiCalls({ selectedTable, connectorName });

        const connectorTrigger = findHoverTriggerByAttribute(connectorName);
        expect(connectorTrigger).toBeDefined();

        await userEvent.hover(connectorTrigger!);

        await waitFor(() => {
            expect(screen.getByText("CONNECTOR")).toBeVisible();
        });

        verifyElementExists(screen.getByText("CONNECTOR"));
        expect(
            screen
                .getAllByText(connectorName)
                .some((el) => el.tagName.toLowerCase() === "a")
        ).toBe(true);

        verifyElementExists(screen.getByText("Source Table"));
        verifyElementExists(screen.getByText(sourceTable));
    });

    it("Should display the date range", async () => {
        const dateRange = "2010-12-01 to 2011-12-09";
        await renderTableView({
            ...defaultProps,
            tables,
            selectedTable: "stock",
            tableIds: { stock: "stock" }
        });

        const dateText = await screen.findByText(dateRange);
        verifyElementExists(dateText);
    });

    it("Should display column stats in the table", async () => {
        const columnName = "StockCode";
        const stype = "ID";
        const dtype = "String";
        const selectedTable = "stock";
        await renderTableView({
            ...defaultProps,
            tables: tables,
            selectedTable: selectedTable
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/tables/${selectedTable}/col_stats`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        await waitFor(() => {
            const tbody = document.querySelector('[data-slot="table-body"]');
            verifyElementExists(tbody);

            const row = tbody!.querySelector('[data-slot="table-row"]');
            verifyElementExists(row);

            const cells = row!.querySelectorAll('[data-slot="table-cell"]');

            expect(cells.length).toBeGreaterThanOrEqual(4);

            expect(cells[0].textContent).toContain(columnName);
            expect(cells[1].textContent).toContain(dtype);
            expect(cells[2].textContent).toContain(stype);
            expect(cells[3].textContent).toContain("—");
            expect(cells[4].textContent).toContain("—");
        });
    });

    it("Should call navigateToGraph on back button click", async () => {
        const navigateMock = vi.fn();

        await renderTableView({
            ...defaultProps,
            navigateToGraph: navigateMock
        });

        const backButton = screen.getByRole("button", {
            name: "Back"
        });
        await act(async () => {
            backButton.click();
        });

        expect(navigateMock).toHaveBeenCalled();
    });

    it("Should display a clickable connector link with proper styling inside the tooltip", async () => {
        const selectedTable = "stock";
        const connectorName = "ltv_test";

        await renderTableView({
            ...defaultProps,
            tables,
            selectedTable,
            tableIds: { stock: "stock" }
        });

        await checkApiCalls({ selectedTable, connectorName });

        const connectorTrigger = findHoverTriggerByAttribute(connectorName);

        expect(connectorTrigger).toBeDefined();

        await userEvent.hover(connectorTrigger!);

        const tooltip = await screen.findByText("CONNECTOR");
        verifyElementExists(tooltip);
        const link = screen.getByRole("link", { name: `${connectorName}` });

        verifyElementExists(link);
        expect(link).toHaveAttribute(
            "href",
            `${getProjectPath("project-tests").Connectors}${connectorName}`
        );
        expect(link).toHaveAttribute("id", `connecter-${connectorName}`);

        await userEvent.click(link!);
        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                `${getProjectPath("project-tests").Connectors}${connectorName}`
            );
        });
    });

    it("Should display `Something went wrong` for `Invalid` tables list", async () => {
        await renderTableView({
            ...defaultProps,
            tables: { name: false } as unknown as JSTable[]
        });

        const alert = screen.queryByRole("alert");
        verifyElementExists(alert);
        expect(alert).toHaveTextContent(/Something went wrong in/i);
        expect(alert).toHaveTextContent(/TableView/i);
        expect(alert).toHaveTextContent(/Please contact Kumo support/i);
    });
});
