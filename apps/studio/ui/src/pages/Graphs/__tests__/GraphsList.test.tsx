// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import {
    findAndClickButton,
    findAndClickButtonByRole,
    findAndClickButtonByTestId,
    verifyButtonExists,
    verifyTextExists
} from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { JSGraph } from "types/graphs";
import { FlowType } from "types/tables";

import { getProjectPath } from "utils/Constants";
import { getKeys } from "utils/Helpers/objectHelpers";
import {
    formatToLongLocalDateString,
    formatToShortLocalDateString
} from "utils/timeConverter";

import GraphsList from "../GraphsList/GraphsList";
import NewGraph from "../NewGraph/NewGraph";

import { expectedGraphDetailResponse } from "./MockFunction/GraphDetail.test.mock";
import {
    expectedGraphsListResponse,
    setGraphsListInvalidField,
    setGraphsListResponseType
} from "./MockFunction/GraphsList.test.mock";

const graphDetails = expectedGraphDetailResponse();
const projectPaths = getProjectPath("project-tests");

const expectProjectGraphGet = (graphName: string) => {
    expect(mockAxiosInstance.get).toHaveBeenCalledWith(
        `v1/graphs/${graphName}`,
        undefined,
        {
            headers: undefined,
            params: { project_id: "project-tests" }
        }
    );
};

const graphInfo: JSGraph = {
    id: graphDetails.id,
    name: graphDetails.name_alias,
    tables: graphDetails.graph?.tables,
    groups: graphDetails.graph?.col_groups,
    totalLinks: graphDetails.num_links as number,
    totalTables: graphDetails.num_tables as number,
    timestamps: graphDetails.timestamps,
    created: formatToShortLocalDateString(graphDetails.timestamps?.created_at),
    dateCreated: formatToLongLocalDateString(
        graphDetails.timestamps?.created_at
    ),
    updated: formatToLongLocalDateString(graphDetails.timestamps?.updated_at),
    dateUpdated: formatToLongLocalDateString(
        graphDetails.timestamps?.updated_at
    ),
    table_name_to_fpid_mapping: graphDetails.table_name_to_fpid_mapping
};

describe("GraphsListPage", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });

    const renderGraphsListPage = async () => {
        const queryClient = createQueryClient(true);
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <GraphsList />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const renderCreateGraph = async () => {
        const queryClient = createQueryClient();
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <NewGraph />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCalls = async ({ connectors = true, graphs = true }) => {
        if (graphs) {
            await waitFor(() => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    "v1/graphs/",
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                );
            });
        }
        if (connectors) {
            await waitFor(() => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    "v0/projects/project-tests/connectors/"
                );
            });
        }
    };

    const getGraphNames = () =>
        screen
            .getAllByTestId(/^entity-name-/)
            .map((el) => el.textContent?.trim() ?? "");

    const getGraphDates = () =>
        screen.getAllByTestId(/^entity-name-/).map((el) => {
            const dateDiv = el.parentElement?.querySelector("div:nth-child(2)");
            return new Date(dateDiv?.textContent ?? "").getTime();
        });

    const selectGraphAndVerifyOverview = async () => {
        const graphToClick = await screen.findByTestId(/^entity-name-0/);
        expect(graphToClick).toBeInTheDocument();

        const resolvedGraphsListResponse = expectedGraphsListResponse();
        const expectedGraph = resolvedGraphsListResponse[0];

        await userEvent.click(graphToClick);

        await waitFor(() => {
            expectProjectGraphGet(expectedGraph?.name ?? "");
        });

        await checkApiCalls({
            graphs: false,
            connectors: false
        });

        await waitFor(() => {
            const detailName = document.querySelector(
                '[data-slot="card"]:nth-child(2)'
            );
            expect(detailName).toBeInTheDocument();
            expect(detailName).toHaveTextContent(expectedGraph?.name);
        });

        return expectedGraph;
    };

    const testSortOption = async (
        sortDropdown: HTMLElement,
        optionText: string,
        sortType: "alphabetical" | "date",
        ascending: boolean
    ) => {
        await userEvent.click(sortDropdown);
        const option = await screen.findByText(new RegExp(optionText, "i"));
        await userEvent.click(option);

        await waitFor(() => {
            if (sortType === "alphabetical") {
                const names = getGraphNames();
                const expected = [...names].sort((a, b) =>
                    ascending ? a.localeCompare(b) : b.localeCompare(a)
                );
                expect(names).toEqual(expected);
            } else {
                const dates = getGraphDates();
                const expected = [...dates].sort((a, b) =>
                    ascending ? a - b : b - a
                );
                expect(dates).toEqual(expected);
            }
        });
    };

    beforeEach(() => {
        mockNavigateCall();
        vi.clearAllMocks();

        setGraphsListResponseType(testResponseType.DEFAULT);
    });

    it("Should navigate to the new graph page when clicking on the 'Create graph' button", async () => {
        await renderGraphsListPage();
        await renderCreateGraph();

        await checkApiCalls({ graphs: true, connectors: false });

        const createGraphButton = await screen.findByText("Create graph");
        await userEvent.click(createGraphButton);

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                projectPaths.NewGraph,
                {
                    state: {
                        flow_type: FlowType.CREATE,
                        name: ""
                    }
                }
            );
        });

        verifyTextExists("Create New Graph");
    });

    it("Should show graph overview on clicking graph list item.", async () => {
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        await verifyTextExists(
            "Select a graph from the left panel to view its details here."
        );

        const graphToClick = screen.getByTestId(/^entity-name-0/);
        expect(graphToClick).toBeInTheDocument();

        const resolvedGraphsListResponse = expectedGraphsListResponse();
        const expectedGraph = resolvedGraphsListResponse[0];
        await userEvent.click(graphToClick);

        await waitFor(() => {
            expectProjectGraphGet(expectedGraph?.name ?? "");
        });

        await checkApiCalls({
            graphs: false,
            connectors: false
        });

        await waitFor(() => {
            const detailName = document.querySelector(
                '[data-slot="card"]:nth-child(2)'
            );
            expect(detailName).toBeInTheDocument();
            expect(detailName).toHaveTextContent(expectedGraph?.name);
        });

        const tables = getKeys(graphDetails?.graph?.tables);

        for (let i = 0; i < tables.length; i++) {
            await waitFor(() => {
                const node = screen.getByTestId("rf__node-" + tables[i]);
                expect(node).toBeInTheDocument();
                expect(node).toHaveTextContent(tables[i]);
            });
        }
    });

    it("Should filter and reset graph list based on search input", async () => {
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        const searchInput = await screen.findByPlaceholderText("Search");
        expect(searchInput).toBeInTheDocument();

        const initialGraphs = (
            await screen.findAllByTestId(/^entity-name-/)
        ).map((el) => el?.textContent?.trim());

        await userEvent.type(searchInput, "ltv_graph");
        await waitFor(() => {
            const filteredTables = screen.getAllByTestId(/^entity-name-/);
            expect(filteredTables).toHaveLength(1);
        });
        verifyTextExists("ltv_graph");

        await userEvent.clear(searchInput);
        await waitFor(() => {
            const resetTables = screen
                .getAllByTestId(/^entity-name-/)
                .map((el) => el?.textContent?.trim());
            expect(resetTables).toEqual(initialGraphs);
        });
    });

    it("Should filter and sort the graphs list based on combination search and sort", async () => {
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        const searchInput = await screen.findByPlaceholderText("Search");
        expect(searchInput).toBeInTheDocument();

        await userEvent.type(searchInput, "graph");
        await waitFor(() => {
            const filteredGraphs = screen.getAllByTestId(/^entity-name-/);
            expect(filteredGraphs).toHaveLength(4);
        });

        const sortDropdown = screen.getByTestId("testid-list-sort-select");
        expect(sortDropdown).toBeInTheDocument();

        await testSortOption(
            sortDropdown,
            "Alphabetical \\(A–Z\\)",
            "alphabetical",
            true
        );
        await testSortOption(
            sortDropdown,
            "Alphabetical \\(Z–A\\)",
            "alphabetical",
            false
        );
        await testSortOption(sortDropdown, "Least recent", "date", true);
        await testSortOption(sortDropdown, "Most recent", "date", false);
    });

    it("Should sort the graph list in different orders", async () => {
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        const sortDropdown = await screen.findByTestId(
            "testid-list-sort-select"
        );
        expect(sortDropdown).toBeInTheDocument();
        expect(sortDropdown).toHaveTextContent(/Sort by: Most recent/i);

        await testSortOption(
            sortDropdown,
            "Alphabetical \\(A–Z\\)",
            "alphabetical",
            true
        );
        await testSortOption(
            sortDropdown,
            "Alphabetical \\(Z–A\\)",
            "alphabetical",
            false
        );
        await testSortOption(sortDropdown, "Least recent", "date", true);
        await testSortOption(sortDropdown, "Most recent", "date", false);
    });

    it("Should display 'Any graphs will be listed here.' for invalid search term", async () => {
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        const searchInput = await screen.findByPlaceholderText("Search");
        expect(searchInput).toBeInTheDocument();
        await userEvent.type(searchInput, "#$#%#$#");
        await waitFor(() => {
            const sideView = document.querySelector('[data-slot="card"]');
            expect(sideView).toBeInTheDocument();
            expect(sideView).toHaveTextContent(
                "Any graphs will be listed here."
            );
        });
    });

    it("Should navigate to 'Create New Graph' page when graphs list is empty", async () => {
        setGraphsListResponseType(testResponseType.EMPTY);
        await renderGraphsListPage();
        await renderCreateGraph();

        await checkApiCalls({ graphs: true, connectors: true });

        expect(await screen.findByText("Create New Graph")).toBeInTheDocument();
        verifyTextExists("Link tables to create graph schema");
    });

    it("Should show Invalid Error Message for invalid graph list response", async () => {
        setGraphsListResponseType(testResponseType.INVALID);
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        expect(
            await screen.findByText(/Something went wrong/)
        ).toBeInTheDocument();
        verifyTextExists(/Please contact Kumo support/);
    });

    it("Should show empty graph list for invalid id response", async () => {
        setGraphsListResponseType(testResponseType.INVALID);
        setGraphsListInvalidField("id");
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        const filteredGraphs = await screen.findAllByTestId(/^entity-name-/);
        expect(filteredGraphs).toHaveLength(1);
    });

    it("Should show '-' timestamps in graphs list for invalid timestamps response", async () => {
        setGraphsListResponseType(testResponseType.INVALID);
        setGraphsListInvalidField("timestamps");
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        expect(await screen.findByText("-")).toBeInTheDocument();
    });

    it("Should navigate to the empty state for an 'Internal Server Error' response", async () => {
        setGraphsListResponseType(testResponseType.ERROR);
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        const graphs = await screen.findByText("Graphs");
        expect(graphs).toBeInTheDocument();

        const filteredGraphs = screen.queryByTestId(/^entity-name-/);
        expect(filteredGraphs).toBeNull();
    });

    it("Should navigate to the edit graph page when clicking the edit button", async () => {
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        const expectedGraph = await selectGraphAndVerifyOverview();

        await findAndClickButtonByTestId("more-options-dropdown");
        await findAndClickButton("Edit");

        await renderCreateGraph();

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                projectPaths.EditGraph,
                {
                    state: {
                        flow_type: FlowType.EDIT,
                        name: expectedGraph?.name,
                        jsGraph: graphInfo
                    }
                }
            );
        });
    });

    it("Should display the 'Delete Graph!' popup when clicking the delete button", async () => {
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        const expectedGraph = await selectGraphAndVerifyOverview();

        await findAndClickButtonByTestId("more-options-dropdown");
        await findAndClickButton("Delete");

        const modal = screen.getByRole("dialog");
        verifyTextExists("Delete Graph");

        verifyButtonExists("Cancel");

        expect(modal.textContent).toContain(
            `Are you sure you want to delete graph "${expectedGraph?.name}"? This action cannot be undone.`
        );

        await findAndClickButtonByRole("Delete");

        await waitFor(() => {
            expect(mockAxiosInstance.delete).toHaveBeenCalledWith(
                `v1/graphs/${expectedGraph?.name}`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });
    });

    it("Should navigate to the duplicate graph page when clicking the 'Duplicate' button", async () => {
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        const expectedGraph = await selectGraphAndVerifyOverview();

        await findAndClickButtonByTestId("more-options-dropdown");
        await findAndClickButton("Duplicate");

        await renderCreateGraph();

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                projectPaths.NewGraph,
                {
                    state: {
                        flow_type: FlowType.DUPLICATE,
                        name: expectedGraph?.name,
                        jsGraph: graphInfo
                    }
                }
            );
        });
    });

    it("Should refresh when clicking the 'Re-ingest' button", async () => {
        await renderGraphsListPage();

        await checkApiCalls({ graphs: true, connectors: false });

        const expectedGraph = await selectGraphAndVerifyOverview();

        await findAndClickButtonByTestId("more-options-dropdown");
        await findAndClickButton("Re-ingest");

        await waitFor(() => {
            expectProjectGraphGet(expectedGraph?.name ?? "");
        });

        await checkApiCalls({
            graphs: false,
            connectors: false
        });

        const tables = getKeys(graphDetails?.graph?.tables);

        //Verify that table nodes are present in the graph view.
        for (let i = 0; i < tables.length; i++) {
            await waitFor(() => {
                const node = screen.getByTestId("rf__node-" + tables[i]);
                expect(node).toBeInTheDocument();
                expect(node).toHaveTextContent(tables[i]);
            });
        }
    });
});
