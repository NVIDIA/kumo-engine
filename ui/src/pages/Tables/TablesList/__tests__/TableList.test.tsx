// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { setDeploymentTypeInStore } from "stores/__tests__/debugInfoStore";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import { testResponseType } from "tests/testResponseType";
import { deploymentType } from "tests/testType";
import { createQueryClient } from "tests/testUtil";

import ConnectTables from "components/Tables/Connect/ConnectTables";

import { setConnectorListResponseType } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorList.mock";

import {
    CreateTableArgs,
    FlowType,
    TableConnector,
    TableListResponse
} from "types/tables";

import { getProjectPath, storageKeys } from "utils/Constants";
import { getConnectorIDFromTableSource } from "utils/Converters/tablesConverter";
import { parseDateTime } from "utils/timeConverter";

// Force the Custom SQL feature flag OFF for this suite so the list keeps its
// prior layout (no Raw/SQL "Kind" filter). The flag-on behavior is covered in
// TableListKindFilter.test.tsx.
vi.mock("utils/Constants", async (importOriginal) => {
    const actual = await importOriginal<typeof import("utils/Constants")>();
    return { ...actual, ENABLE_CUSTOM_SQL_TABLES: false };
});

import TablesList, { resolveTableConnectorDetails } from "../TablesList";

import { expectedColStatsResponse } from "./MockFunction/Colstats.test.mock";
import {
    expectedTableDetailsResponse,
    setTableUsesProjectUploadConnector
} from "./MockFunction/TableDetail.test.mock";
import {
    expectedTableListResponse,
    setTableListResponseType
} from "./MockFunction/TableList.test.mock";

interface PageLoadTestArg {
    checkLoading?: boolean;
    checkApiCall?: boolean;
    checkTableList?: boolean;
    checkEmptyList?: boolean;
    checkPageTitle?: boolean;
}

describe("resolveTableConnectorDetails", () => {
    it.each([
        {
            connectorId: "project-upload",
            uploadConnectorId: "project-upload",
            expectedId: "",
            isUpload: true
        },
        {
            connectorId: "warehouse-connector",
            uploadConnectorId: "project-upload",
            expectedId: "warehouse-connector",
            isUpload: false
        },
        {
            connectorId: "warehouse-connector",
            uploadConnectorId: "",
            expectedId: "",
            isUpload: false
        },
        {
            connectorId: "",
            uploadConnectorId: "project-upload",
            expectedId: "",
            isUpload: false
        }
    ])(
        "resolves connector details for connector=$connectorId upload=$uploadConnectorId",
        ({ connectorId, uploadConnectorId, expectedId, isUpload }) => {
            expect(
                resolveTableConnectorDetails(connectorId, uploadConnectorId)
            ).toEqual({
                connectorDetailsId: expectedId,
                isProjectUploadConnector: isUpload
            });
        }
    );
});

describe("TablesListPage", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });
    const queryClient = createQueryClient();
    const projectPaths = getProjectPath("project-tests");

    const renderTablesListPage = async () => {
        let result: ReturnType<typeof render> | undefined;
        await act(async () => {
            result = render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <TablesList />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
        return result!;
    };

    const renderCreateTable = async () => {
        const queryClient = createQueryClient();
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <ConnectTables />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const getTableNames = () =>
        screen
            .getAllByTestId(/^entity-name-/)
            .map((el) => el.textContent?.trim() ?? "");

    const getTableDates = () =>
        screen.getAllByTestId(/^entity-name-/).map((el) => {
            const dateDiv = el.parentElement?.querySelector("div:nth-child(2)");
            return new Date(dateDiv?.textContent ?? "").getTime();
        });

    const selectTableAndVerify = async (index: number = 0) => {
        const tableToClick = screen.getByTestId(`entity-name-${index}`);
        expect(tableToClick).toBeInTheDocument();
        await userEvent.click(tableToClick);

        const resolvedTableListResponse = expectedTableListResponse();
        const expectedTable = resolvedTableListResponse[index];

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/tables/${expectedTable?.id}`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/tables/${expectedTable?.id}/col_stats`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        const resolvedConnector = expectedTableDetailsResponse(
            expectedTable?.name
        );
        const connectorId = getConnectorIDFromTableSource(
            resolvedConnector?.table?.source_table
        );
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/projects/project-tests/connectors/${connectorId}`
            );
        });

        await waitFor(() => {
            const detailName = document.querySelector(
                '[data-slot="card"]:nth-child(2)'
            );
            expect(detailName).toBeInTheDocument();
            expect(detailName).toHaveTextContent(expectedTable?.name);
        });

        return { expectedTable, connectorId };
    };

    const defaultPageLoadOptions: PageLoadTestArg = {
        checkLoading: false,
        checkApiCall: true,
        checkTableList: true,
        checkEmptyList: false
    };

    const pageLoadTest = async ({
        checkLoading = true,
        checkApiCall = true,
        checkTableList = true,
        checkEmptyList = false,
        checkPageTitle = true
    }: PageLoadTestArg) => {
        if (checkPageTitle) {
            const pageTitle = screen.getByText("Tables");
            expect(pageTitle).toBeInTheDocument();
        }
        const resolvedTableListResponse = expectedTableListResponse();
        const expectedTableList =
            resolvedTableListResponse as TableListResponse[];

        if (checkLoading) {
            await waitFor(() => {
                const loading = document.getElementsByClassName(
                    "login-anim-container"
                );
                expect(loading).toBeDefined();
            });
        }

        if (checkApiCall) {
            await waitFor(() => {
                const APIS_TO_CALL = ["v1/tables/"];
                APIS_TO_CALL.forEach((API) => {
                    expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                        `${API}`,
                        undefined,
                        {
                            headers: undefined,
                            params: { project_id: "project-tests" }
                        }
                    );
                });
            });
            if (checkEmptyList) {
                await waitFor(async () => {
                    expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                        "v1/tables/",
                        undefined,
                        {
                            headers: undefined,
                            params: { project_id: "project-tests" }
                        }
                    );
                    expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                        "v0/projects/project-tests/connectors/"
                    );
                    expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                        "v1/tables/",
                        undefined,
                        {
                            headers: undefined,
                            params: { project_id: "project-tests" }
                        }
                    );
                    expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                        "v0/projects/project-tests/connectors/"
                    );

                    expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                        "v1/tables/",
                        undefined,
                        {
                            headers: undefined,
                            params: { project_id: "project-tests" }
                        }
                    );
                });
            }
        }

        if (checkTableList) {
            await waitFor(async () => {
                const sideView = document.querySelector('[data-slot="card"]');
                await waitFor(() => {
                    expect(sideView).toBeInTheDocument();
                });
                const tableListItems = screen.getAllByTestId(/^entity-name-/);
                await waitFor(() => {
                    expect(tableListItems).toHaveLength(
                        expectedTableList?.length
                    );
                });
            });

            // Make copy of mock data and sort it;
            const expectedData = [...expectedTableList];
            expectedData.sort((a, b) =>
                parseDateTime(a?.timestamps?.created_at) >
                parseDateTime(b?.timestamps?.created_at)
                    ? -1
                    : 1
            );
            expectedData.sort((a, b) =>
                parseDateTime(a?.timestamps?.updated_at) >
                parseDateTime(b?.timestamps?.updated_at)
                    ? -1
                    : 1
            );

            for (let i = 0; i < expectedData.length; i++) {
                const entityName = screen.getByTestId(`entity-name-${i}`);
                const expectedName = expectedData[i].name;
                expect(entityName).toHaveTextContent(expectedName);
            }
        }

        if (checkEmptyList) {
            expect(screen.getByText("Add New Table")).toBeInTheDocument();
        }
    };

    beforeEach(() => {
        storeToLocalStorage(storageKeys.selectedProject, "project-tests");
        vi.clearAllMocks();
        window.location.href = window.location.href + "tables/";
        mockNavigateCall();
        queryClient.clear();

        setTableListResponseType(testResponseType.EMPTY);
        setTableUsesProjectUploadConnector(false);
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    it("Should render Table List Page with navigation", async () => {
        setTableListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest(defaultPageLoadOptions);
    });

    it("Should display 'Select a table from the left panel to view its details here.' when none of the tables are selected", async () => {
        setTableListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest(defaultPageLoadOptions);
        const unselectedEntity = screen.getByTestId("entity-item-not-selected");
        expect(unselectedEntity).toBeInTheDocument();
        expect(
            screen.getByText(
                "Select a table from the left panel to view its details here."
            )
        ).toBeInTheDocument();
    });

    it("Should not render the Raw/SQL kind filter when the feature flag is off", async () => {
        setTableListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest(defaultPageLoadOptions);

        expect(
            screen.queryByTestId("testid-table-kind-filter")
        ).not.toBeInTheDocument();
    });

    it("Should filter tables list based on search input", async () => {
        setTableListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest(defaultPageLoadOptions);

        const searchInput = screen.getByPlaceholderText("Search");
        expect(searchInput).toBeInTheDocument();

        await userEvent.type(searchInput, "transactions_hnm_demo");
        await waitFor(() => {
            const filteredTables = screen.getAllByTestId(/^entity-name-/);
            expect(filteredTables).toHaveLength(1);
            expect(
                screen.getByText("transactions_hnm_demo")
            ).toBeInTheDocument();
        });

        await userEvent.clear(searchInput);
        await waitFor(async () => {
            const allTables = screen.getAllByTestId(/^entity-name-/);
            const resolvedTableListResponse = expectedTableListResponse();
            expect(allTables).toHaveLength(resolvedTableListResponse?.length);
        });
    });

    it("Should reset the table list to its original state when the search filter is cleared.", async () => {
        setTableListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest(defaultPageLoadOptions);

        const initialTables = getTableNames();

        const searchInput = screen.getByPlaceholderText("Search");
        expect(searchInput).toBeInTheDocument();
        await userEvent.type(searchInput, "stock");
        await waitFor(() => {
            const filteredTables = screen.getAllByTestId(/^entity-name-/);
            expect(filteredTables).toHaveLength(1);
            expect(screen.getByText("stock")).toBeInTheDocument();
        });

        await userEvent.clear(searchInput);
        await waitFor(() => {
            const resetTables = getTableNames();
            expect(resetTables).toEqual(initialTables);
        });
    });

    it("Should display 'Any tables will be listed here.' for invalid search term", async () => {
        setTableListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest(defaultPageLoadOptions);

        const searchInput = screen.getByPlaceholderText("Search");
        expect(searchInput).toBeInTheDocument();
        await userEvent.type(searchInput, "#$#%#$#");
        await waitFor(() => {
            const sideView = document.querySelector('[data-slot="card"]');
            expect(sideView).toBeInTheDocument();
            expect(sideView).toHaveTextContent(
                "Any tables will be listed here."
            );
        });
    });

    it("Should sort the table list in different orders", async () => {
        setTableListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest(defaultPageLoadOptions);

        const sortDropdown = screen.getByTestId("testid-list-sort-select");
        expect(sortDropdown).toBeInTheDocument();
        expect(sortDropdown).toHaveTextContent(/Sort by: Most recent/i);
        await userEvent.click(sortDropdown);

        // Alphabetical (A–Z)
        const azOption = await screen.findByText(/Alphabetical \(A–Z\)/i);
        await userEvent.click(azOption);
        await waitFor(() => {
            const names = getTableNames();
            const expected = [...names].sort((a, b) => a.localeCompare(b));
            expect(names).toEqual(expected);
        });

        // Alphabetical (Z–A)
        await userEvent.click(sortDropdown);
        const zaOption = await screen.findByText(/Alphabetical \(Z–A\)/i);
        await userEvent.click(zaOption);
        await waitFor(() => {
            const names = getTableNames();
            const expected = [...names].sort((a, b) => b.localeCompare(a));
            expect(names).toEqual(expected);
        });

        // Least Recent
        await userEvent.click(sortDropdown);
        const leastRecentOption = await screen.findByText(/Least recent/i);
        await userEvent.click(leastRecentOption);
        await waitFor(() => {
            const timestamps = getTableDates();
            const expected = [...timestamps].sort((a, b) => a - b);
            expect(timestamps).toEqual(expected);
        });

        // Most Recent
        await userEvent.click(sortDropdown);
        const mostRecentOption = await screen.findByText(/Most recent/i);
        await userEvent.click(mostRecentOption);
        await waitFor(() => {
            const timestamps = getTableDates();
            const expected = [...timestamps].sort((a, b) => b - a);
            expect(timestamps).toEqual(expected);
        });
    });

    it("Should filter and sort the tables list based on combination search and sort", async () => {
        setTableListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest(defaultPageLoadOptions);

        const searchInput = screen.getByPlaceholderText("Search");
        expect(searchInput).toBeInTheDocument();

        await userEvent.type(searchInput, "transaction");
        await waitFor(() => {
            const filteredTables = screen.getAllByTestId(/^entity-name-/);
            expect(filteredTables).toHaveLength(4);
        });

        const sortDropdown = screen.getByTestId("testid-list-sort-select");
        expect(sortDropdown).toBeInTheDocument();

        // A–Z
        await userEvent.click(sortDropdown);
        const azOption = await screen.findByText(/Alphabetical \(A–Z\)/i);
        await userEvent.click(azOption);
        await waitFor(() => {
            const names = getTableNames();
            const expected = [...names].sort((a, b) => a.localeCompare(b));
            expect(names).toEqual(expected);
        });

        // Z–A
        await userEvent.click(sortDropdown);
        const zaOption = await screen.findByText(/Alphabetical \(Z–A\)/i);
        await userEvent.click(zaOption);
        await waitFor(() => {
            const names = getTableNames();
            const expected = [...names].sort((a, b) => b.localeCompare(a));
            expect(names).toEqual(expected);
        });

        // Least Recent
        await userEvent.click(sortDropdown);
        const leastRecentOption = await screen.findByText(/Least recent/i);
        await userEvent.click(leastRecentOption);
        await waitFor(() => {
            const dates = getTableDates();
            const expected = [...dates].sort((a, b) => a - b);
            expect(dates).toEqual(expected);
        });

        // Most Recent
        await userEvent.click(sortDropdown);
        const mostRecentOption = await screen.findByText(/Most recent/i);
        await userEvent.click(mostRecentOption);
        await waitFor(() => {
            const dates = getTableDates();
            const expected = [...dates].sort((a, b) => b - a);
            expect(dates).toEqual(expected);
        });
    });

    it("Should navigate to the new table page when clicking on the Add Table button", async () => {
        setTableListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest({ ...defaultPageLoadOptions, checkLoading: true });
        const addTableButton = screen.getByText("Add table");
        expect(addTableButton).toBeInTheDocument();

        await userEvent.click(addTableButton);

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(projectPaths.NewTable);
        });
    });

    it("Should fetch and display table details on table click", async () => {
        setTableListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest(defaultPageLoadOptions);

        const { expectedTable, connectorId } = await selectTableAndVerify();

        const createdDetail = screen.queryByRole("alert-Created-tooltip");
        if (createdDetail) {
            expect(createdDetail).toBeInTheDocument();
            await userEvent.hover(createdDetail);
            expect(createdDetail).toHaveTextContent(
                expectedTable?.timestamps?.created_at
            );
        }

        const connectorDetail = screen.queryByRole(
            "alert-table-connector-tooltip"
        );
        if (connectorDetail) {
            expect(connectorDetail).toBeInTheDocument();
            await userEvent.hover(connectorDetail);
            expect(connectorDetail).toHaveTextContent(connectorId);
        }

        const columnStatsHeader = screen.getByText("Column Statistics");
        expect(columnStatsHeader).toBeInTheDocument();
        const info = screen.getByText(
            "The following stats are computed from a sample of table data."
        );
        expect(info).toBeInTheDocument();

        const colStatsTable = screen.getByRole("table");
        await waitFor(() => {
            expect(colStatsTable).toBeInTheDocument();
        });

        const colStatsTableRows = screen.getAllByRole("row")?.slice(1); // Skip header row
        await waitFor(() => {
            expect(colStatsTableRows?.length).toBeGreaterThan(0);
        });
        const resolvedColStats = expectedColStatsResponse();
        await waitFor(() => {
            expect(colStatsTableRows?.length).toBe(
                resolvedColStats?.columns?.length
            );
        });

        const firstColumnCells = colStatsTableRows?.map((row) => {
            const cells = within(row).queryAllByRole("cell");
            return cells[0] ?? null;
        });

        const expectedColumnNames = resolvedColStats?.columns?.map(
            (col) => col?.name
        );

        firstColumnCells.forEach((cell, index) => {
            const expectedText = expectedColumnNames[index] ?? "";
            expect(cell).toHaveTextContent(expectedText);
        });
    });

    it("Should not fetch or link the hidden project upload connector", async () => {
        setTableListResponseType(testResponseType.ALL);
        setTableUsesProjectUploadConnector(true);
        await renderTablesListPage();
        await pageLoadTest(defaultPageLoadOptions);

        await userEvent.click(screen.getByTestId("entity-name-0"));

        expect(await screen.findByTestId("table-connector")).toHaveTextContent(
            "Local Upload"
        );
        expect(mockAxiosInstance.get).not.toHaveBeenCalledWith(
            "v0/projects/project-tests/connectors/project_upload_connector"
        );
        expect(
            screen.queryByRole("link", { name: "Local Upload" })
        ).not.toBeInTheDocument();
        expect(
            screen.queryByText("Something went wrong!")
        ).not.toBeInTheDocument();
    });

    it("Should isolate upload connector discovery by selected project", async () => {
        const firstRender = await renderTablesListPage();
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                "v0/projects/project-tests/connectors/uploads/connector"
            );
        });

        firstRender.unmount();
        storeToLocalStorage(storageKeys.selectedProject, "project-other");
        await renderTablesListPage();

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                "v0/projects/project-other/connectors/uploads/connector"
            );
        });
    });

    it("Should show Invalid Error Message for invalid table list response", async () => {
        setTableListResponseType(testResponseType.INVALID);
        await renderTablesListPage();
        await pageLoadTest({
            checkTableList: false,
            checkEmptyList: false,
            checkPageTitle: false
        });

        expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();
        expect(
            screen.getByText(/Please contact Kumo support/)
        ).toBeInTheDocument();

        setTableListResponseType(testResponseType.ALL);
    });

    it("Should navigate to 'Add New Table' page when tables list page is empty", async () => {
        setConnectorListResponseType(testResponseType.ALL);
        setTableListResponseType(testResponseType.EMPTY);
        await renderTablesListPage();
        await renderCreateTable();
        await pageLoadTest({
            checkLoading: false,
            checkApiCall: true,
            checkTableList: false,
            checkEmptyList: true
        });
        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                projectPaths.NewTable,
                {
                    state: {
                        "tableNotExists": true
                    }
                }
            );
        });
        expect(screen.getByText("Add New Table")).toBeInTheDocument();
    });

    it("Should navigate to the empty state for an 'Internal Server Error' response", async () => {
        setTableListResponseType(testResponseType.ERROR);
        await renderTablesListPage();
        await pageLoadTest({
            checkTableList: false,
            checkEmptyList: false
        });

        const emptyCard = document.querySelector('[data-slot="card"]');
        expect(emptyCard).not.toBeNull();

        setTableListResponseType(testResponseType.ALL);
    });

    it("Should render Table List Page with navigation - Databricks deployment", async () => {
        setTableListResponseType(testResponseType.ALL);
        setDeploymentTypeInStore(deploymentType.Databricks);
        await renderTablesListPage();
        await pageLoadTest({ ...defaultPageLoadOptions, checkLoading: true });
    });

    it("Should navigate to the Add New Table page when the table list page is empty - Databricks deployment", async () => {
        setTableListResponseType(testResponseType.EMPTY);
        setConnectorListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await renderCreateTable();
        await pageLoadTest({
            ...defaultPageLoadOptions,
            checkLoading: true,
            checkTableList: false,
            checkEmptyList: true
        });

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                projectPaths.NewTable,
                {
                    state: {
                        "tableNotExists": true
                    }
                }
            );
        });
    });

    it("Should navigate to edit table page on clicking the edit button", async () => {
        setTableListResponseType(testResponseType.ALL);
        setConnectorListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest({ ...defaultPageLoadOptions, checkLoading: true });

        await selectTableAndVerify();
        const clickToggleIcon = screen.getByTestId("more-options-dropdown");
        expect(clickToggleIcon).toBeInTheDocument();
        await userEvent.click(clickToggleIcon);

        const editButton = screen.getByText("Edit");
        await userEvent.click(editButton);

        const resolvedEditTableArgs = expectedTableDetailsResponse();
        const TableInfo: CreateTableArgs = {
            cols: resolvedEditTableArgs?.table?.cols,
            source_table: resolvedEditTableArgs?.table?.source_table,
            pkey: resolvedEditTableArgs?.table?.pkey ?? "",
            time_col: resolvedEditTableArgs?.table?.time_col ?? "",
            end_time_col: resolvedEditTableArgs?.table?.end_time_col ?? ""
        };
        const ConnectorConfig: TableConnector = {
            id: resolvedEditTableArgs?.connector?.id ?? "",
            name: resolvedEditTableArgs?.connector?.name ?? ""
        };

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                projectPaths.EditTable,
                {
                    state: {
                        createtableargs: TableInfo,
                        flowtype: FlowType.EDIT,
                        s3ConnctorConfig: ConnectorConfig,
                        tableName: resolvedEditTableArgs?.name_alias,
                        islocal: false
                    }
                }
            );
        });
    });

    it("Should display the 'Delete Table!' popup on clicking the delete button", async () => {
        setTableListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest({ ...defaultPageLoadOptions, checkLoading: true });

        const { expectedTable } = await selectTableAndVerify();
        const tableToClick = screen.getByTestId("entity-name-0");

        const clickToggleIcon = screen.getByTestId("more-options-dropdown");
        expect(clickToggleIcon).toBeInTheDocument();
        await userEvent.click(clickToggleIcon);

        const deleteButton = screen.getByText("Delete");
        await userEvent.click(deleteButton);

        const modal = screen.getByRole("dialog");
        const modalheader = screen.getByText("Delete Table");
        expect(modalheader).toBeInTheDocument();

        const cancelButton = screen.getByRole("button", { name: "Cancel" });
        expect(cancelButton).toBeInTheDocument();
        expect(modal.textContent).toContain(
            `Are you sure you want to delete table "${expectedTable?.name}"? This action cannot be undone.`
        );

        const continueButton = screen.getByRole("button", { name: "Delete" });
        expect(continueButton).toBeInTheDocument();
        await userEvent.click(continueButton);

        await waitFor(() => {
            expect(mockAxiosInstance.delete).toHaveBeenCalledWith(
                `v1/tables/${expectedTable?.name}`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        expect(tableToClick).not.toBeInTheDocument();
    });

    it("Should navigate to the Table creation page with state props on clicking duplicate button", async () => {
        setTableListResponseType(testResponseType.ALL);
        setConnectorListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest({ ...defaultPageLoadOptions, checkLoading: true });

        await selectTableAndVerify();

        const clickToggleIcon = screen.getByTestId("more-options-dropdown");
        expect(clickToggleIcon).toBeInTheDocument();
        await userEvent.click(clickToggleIcon);
        expect(screen.getByText("Duplicate")).toBeInTheDocument();

        const duplicateButton = screen.getByText("Duplicate");
        expect(duplicateButton).toBeInTheDocument();
        await userEvent.click(duplicateButton);

        const resolvedCopyTableArgs = expectedTableDetailsResponse();
        const TableInfo: CreateTableArgs = {
            cols: resolvedCopyTableArgs?.table?.cols,
            source_table: resolvedCopyTableArgs?.table?.source_table,
            pkey: resolvedCopyTableArgs?.table?.pkey ?? "",
            time_col: resolvedCopyTableArgs?.table?.time_col ?? "",
            end_time_col: resolvedCopyTableArgs?.table?.end_time_col ?? ""
        };

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                projectPaths.NewTable,
                {
                    state: {
                        copytableargs: TableInfo,
                        flowtype: FlowType.DUPLICATE,
                        tableName: resolvedCopyTableArgs?.name_alias + "_copy"
                    }
                }
            );
        });
    });

    it("Should re-ingest the table on clicking the re-ingest button", async () => {
        setTableListResponseType(testResponseType.ALL);
        setConnectorListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest({ ...defaultPageLoadOptions, checkLoading: true });

        const { expectedTable } = await selectTableAndVerify();

        const clickToggleIcon = screen.getByTestId("more-options-dropdown");
        expect(clickToggleIcon).toBeInTheDocument();
        await userEvent.click(clickToggleIcon);

        const summaryRefreshButton = screen.getByText("Re-ingest");
        await userEvent.click(summaryRefreshButton);

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v1/tables/${expectedTable?.id}/refresh`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/tables/${expectedTable?.id}`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/tables/${expectedTable?.id}/col_stats`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        const tableHeader = screen.getByTestId("table-header");
        const table = tableHeader.closest("table")!;

        const tableBody = table.querySelector('[data-slot="table-body"]')!;
        const rows = tableBody.querySelectorAll('[data-slot="table-row"]');
        expect(rows.length).toBe(8);
    });

    it("Should display icons for pkey, time_col, and end_time_col when all fields are populated", async () => {
        setTableListResponseType(testResponseType.ALL);
        setConnectorListResponseType(testResponseType.ALL);
        await renderTablesListPage();
        await pageLoadTest({ ...defaultPageLoadOptions, checkLoading: true });

        // Click on "transactions_no_time" table which has all three fields populated
        const tableToClick = screen.getByText("transactions_no_time");
        expect(tableToClick).toBeInTheDocument();
        await userEvent.click(tableToClick);

        // Wait for API calls
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                "v1/tables/NA.NA-d5828d994e8679119b05ec8f07de920a",
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                "v1/tables/NA.NA-d5828d994e8679119b05ec8f07de920a/col_stats",
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        // Verify primary key icon is displayed next to the "id" column
        const primaryKeyIcon = screen.getByTestId("primary-key-icon");
        expect(primaryKeyIcon).toBeInTheDocument();
        expect(primaryKeyIcon.closest("td")).toHaveTextContent("id");

        // Verify time column icon is displayed next to the "created_at" column
        const timeColumnIcon = screen.getByTestId("time-column-icon");
        expect(timeColumnIcon).toBeInTheDocument();
        expect(timeColumnIcon.closest("td")).toHaveTextContent("created_at");

        // Verify end time icon is displayed next to the "updated_at" column
        const endTimeIcon = screen.getByTestId("end-time-icon");
        expect(endTimeIcon).toBeInTheDocument();
        expect(endTimeIcon.closest("td")).toHaveTextContent("updated_at");
    });
});
