// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter, Route, Routes } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    fireEvent,
    render,
    screen,
    waitFor,
    within
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { KumoapiTableTableDefinition as TableDetails } from "gen/v1/types";
import { mockTableIds } from "tests/fixtures/MockGraphInformationPanel";
import { mockAxiosInstance } from "tests/mockAxios";
import {
    mockLocationStateValue,
    mockNavigateCall,
    mockUseNavigate
} from "tests/mockRouter";
import {
    findAndClickButtonByRole,
    verifyButtonExists,
    verifyTextExists
} from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { expectedTableDetailsResponse } from "pages/Tables/TablesList/__tests__/MockFunction/TableDetail.test.mock";
import { expectedTableListResponse } from "pages/Tables/TablesList/__tests__/MockFunction/TableList.test.mock";

import { Graph, GraphDefinition } from "types/graphs";
import { FlowType } from "types/tables";

import { getProjectPath } from "utils/Constants";

import GraphsList from "../GraphsList/GraphsList";
import NewGraph from "../NewGraph/NewGraph";

import { setGraphNameResponseType } from "./MockFunction/GraphName.mock";
import { setGraphSuggestLinksResponseType } from "./MockFunction/GraphSuggestLinks.mock";
import {
    expectedNewGraphResponse,
    setGraphValidateResponseType
} from "./MockFunction/GraphValidate.test.mock";
import { executeGraphCreationFlow } from "./graphCreationHelpers";

const expectedTableList = expectedTableListResponse();

const expectProjectGet = (url: string) => {
    expect(mockAxiosInstance.get).toHaveBeenCalledWith(url, undefined, {
        headers: undefined,
        params: { project_id: "project-tests" }
    });
};

describe("NewGraph ", () => {
    const queryClient = createQueryClient();

    const renderNewGraphPage = async () => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={["/project/project-test/graphs/new-graph"]}>
                    <QueryClientProvider client={queryClient}>
                        <Routes>
                            <Route
                                path={"/project/project-test/graphs/new-graph"}
                                element={<NewGraph />}
                            />
                            <Route
                                path={"/project/project-test/graphs/" + ":id?"}
                                element={<GraphsList />}
                            />
                        </Routes>
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async () => {
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
    };

    const pageLoadTest = async () => {
        await checkApiCall();

        await verifyTextExists("Create New Graph");
        verifyButtonExists("Cancel");
        verifyButtonExists("Next");
        await verifyTextExists("Link tables to create graph schema");
    };

    const checkNameValidationApi = async (name: string) => {
        await waitFor(
            () => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    `v1/graphs/utils/validate_name`,
                    undefined,
                    {
                        "headers": undefined,
                        "params": {
                            "name": `${name}`,
                            "project_id": "project-tests"
                        }
                    }
                );
            },
            { timeout: 2000 }
        );
    };

    const enterGraphNameAndProceed = async (
        graphName: string = "test_graph"
    ) => {
        const nameInput = screen.getByPlaceholderText("name");
        expect(nameInput).toBeEnabled();
        fireEvent.change(nameInput, { target: { value: graphName } });

        const nextButton = screen.getByRole("button", { name: "Next" });
        await userEvent.click(nextButton);

        return { nameInput, nextButton };
    };

    const setupGraphCreationFlow = async () => {
        mockLocationStateValue({
            flow_type: FlowType.CREATE
        });
        await renderNewGraphPage();
        await pageLoadTest();
    };

    const selectTableByCheckbox = async (
        checkboxIndex: number,
        expectedTableId: string
    ) => {
        await waitFor(() => screen.getAllByRole("checkbox"));
        const checkBoxButtons = screen.getAllByRole("checkbox");
        const checkbox = checkBoxButtons[checkboxIndex];
        await userEvent.click(checkbox);

        await waitFor(() => {
            expectProjectGet(`v1/tables/` + expectedTableId);
        });
    };

    const verifyNodeInGraph = (tableName: string) => {
        const node = screen.getByTestId("rf__node-" + tableName);
        expect(node).toBeInTheDocument();
        expect(node).toHaveTextContent(tableName);
        return node;
    };

    const verifyColumnsInTable = async (
        container: HTMLElement,
        columns: (string | undefined)[] | undefined
    ) => {
        await waitFor(() => {
            if (columns?.length) {
                columns.forEach((columnName) => {
                    if (columnName) {
                        const col = within(container).getByRole("cell", {
                            name: columnName
                        });
                        expect(col).toBeInTheDocument();
                    }
                });
            }
        });
    };

    const changeGraphNameInput = async (name: string) => {
        const input = screen.getByPlaceholderText("name");
        fireEvent.change(input, { target: { value: name } });
        await checkNameValidationApi(name);
        return input;
    };

    const createGraphApiCallTest = async (
        args: GraphDefinition,
        options: {
            validateWithFailure?: boolean;
            validateWithInvalid?: boolean;
            createWithFailure?: boolean;
            createWithInvalid?: boolean;
            validName?: string;
        }
    ) => {
        const clickSubmitButton = async () => {
            const saveGraph = screen.getByText("Save");
            await userEvent.click(saveGraph);
        };

        await executeGraphCreationFlow(args, clickSubmitButton, {
            validateWithFailure: options.validateWithFailure,
            validateWithInvalid: options.validateWithInvalid,
            createWithFailure: options.createWithFailure,
            createWithInvalid: options.createWithInvalid,
            validName: options.validName || "test_graph",
            forceRename: false
        });
    };

    /**
     * Advances the create-graph wizard to the link-creation stage with a
     * valid name and tables selected, so the Save button is enabled.
     */
    const reachSaveStage = async () => {
        await setupGraphCreationFlow();

        await changeGraphNameInput("new_graph");

        // Select three tables (transaction_ltv1, stock, transactions_no_time)
        const allRowCheckboxes = screen.getAllByRole("checkbox", {
            name: "Select row"
        });
        await userEvent.click(allRowCheckboxes[0]);
        await userEvent.click(allRowCheckboxes[1]);
        await userEvent.click(allRowCheckboxes[4]);

        await findAndClickButtonByRole("Next");
    };

    const countValidateCalls = () =>
        mockAxiosInstance.post.mock.calls.filter(
            (call) =>
                typeof call[0] === "string" &&
                call[0].startsWith("v1/graphs/validate")
        ).length;

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
        mockNavigateCall();

        setGraphNameResponseType(testResponseType.ALL);
    });

    afterEach(() => {
        setGraphValidateResponseType();
    });

    it("disables the Next button until a graph name is entered", async () => {
        await renderNewGraphPage();

        const nextButton = screen.getByRole("button", { name: "Next" });
        expect(nextButton).toBeDisabled();

        await changeGraphNameInput("new_graph");
        await waitFor(() => expect(nextButton).toBeEnabled());
    });

    it("Should navigate back to Graph list page on `Cancel` click", async () => {
        await setupGraphCreationFlow();

        await findAndClickButtonByRole("Cancel");

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                getProjectPath("project-tests").Graphs,
                {
                    replace: true
                }
            );
        });
    });

    it("Should handle Name validation for valid and invalid message", async () => {
        await renderNewGraphPage();
        await pageLoadTest();

        setGraphNameResponseType(testResponseType.DEFAULT);
        const nameInput = screen.getByPlaceholderText("name");
        const validName = "test_graph";
        fireEvent.change(nameInput, { target: { value: validName } });
        await checkNameValidationApi(validName);

        setGraphNameResponseType(testResponseType.INVALID);
        const invalidName = "invalid graph name";
        await waitFor(() => {
            userEvent.clear(nameInput);
            fireEvent.change(nameInput, { target: { value: invalidName } });
        });

        await checkNameValidationApi(validName);
        await waitFor(
            () => {
                expect(
                    screen.getByText(
                        `Graph identifier may only contain alphanumeric characters and underscores, and must begin with a letter. Got: "${invalidName}"`
                    )
                ).toBeInTheDocument();
            },
            { timeout: 2000 }
        );

        setGraphNameResponseType(testResponseType.EXISTS);
        const alreadyExists = "ltv_graph";
        await userEvent.clear(nameInput);
        fireEvent.change(nameInput, { target: { value: alreadyExists } });

        await checkNameValidationApi(alreadyExists);
        await waitFor(() => {
            expect(
                screen.getByText(
                    `Graph with name ${alreadyExists} already exists. Please choose a different name.`
                )
            ).toBeInTheDocument();
        });
    });

    it("Should show tables for selection in Table view of New graph page", async () => {
        await renderNewGraphPage();
        await pageLoadTest();

        const expectedTableNames = expectedTableList?.map((t) => t.name) ?? [];

        await waitFor(() => screen.getAllByRole("cell"));
        const firstPageCells = screen.getAllByRole("cell");
        const firstPageTableNames = firstPageCells
            .map((cell) => cell.textContent?.trim())
            .filter((name) => name && expectedTableNames.includes(name));

        const nextPageButton = screen.queryByText("2");
        userEvent.click(nextPageButton!);

        await waitFor(() => screen.getAllByRole("cell"));
        const secondPageCells = screen.getAllByRole("cell");
        const secondPageTableNames = secondPageCells
            .map((cell) => cell.textContent?.trim())
            .filter((name) => name && expectedTableNames.includes(name));

        const allExtractedTableNames = [
            ...firstPageTableNames,
            ...secondPageTableNames
        ];
        await waitFor(() => {
            allExtractedTableNames.forEach((tableName) => {
                expect(expectedTableNames).toContain(tableName);
            });
        });
    });

    it("Should show tables in graph view for selected tables in the New graph page", async () => {
        await setupGraphCreationFlow();

        const listResp = expectedTableList.sort(
            (a, b) =>
                new Date(b.timestamps.created_at).getTime() -
                new Date(a.timestamps.created_at).getTime()
        );

        const tableNames = listResp?.map((l) => l?.name);

        expect(screen.getByText("No graph")).toBeInTheDocument();

        await selectTableByCheckbox(1, listResp[0]?.id);
        verifyNodeInGraph(tableNames[0]);

        await selectTableByCheckbox(2, listResp[1]?.id);

        await waitFor(() => {
            verifyNodeInGraph(tableNames[1]);
        });
    });

    it("Should show table preview while clicking on Preview column of a table", async () => {
        await setupGraphCreationFlow();

        const tableNames = expectedTableList?.map((l) => l?.name);
        const preview = await screen.findByTestId("preview-0");
        expect(preview).toBeInTheDocument();
        await userEvent.click(preview);

        await waitFor(() => {
            expectProjectGet("v1/tables/" + expectedTableList[0]?.id);
        });

        const tableDetail = expectedTableDetailsResponse();
        const columns = tableDetail?.table?.cols?.map((c) => c?.name);

        const modal = screen.getByRole("dialog");
        const title = within(modal).getByText(tableNames[0]);
        expect(title).toBeInTheDocument();
        const table = within(modal).getByRole("table");
        expect(table).toBeInTheDocument();

        await verifyColumnsInTable(modal, columns);
    });

    it("Should show dropdown for table selection when clicking on Next, and should show table when clicking on Back button", async () => {
        await setupGraphCreationFlow();

        await enterGraphNameAndProceed();

        const select = screen.getByTestId("testid-combobox");
        within(select).queryByPlaceholderText(/Select tables/i);

        await findAndClickButtonByRole("Back");

        expect(screen.queryByTestId("testid-combobox")).toBeNull();
        expect(screen.getByRole("table")).toBeInTheDocument();
    });

    it("Should show tables in graph view for tables selected using dropdown", async () => {
        await setupGraphCreationFlow();

        await enterGraphNameAndProceed();

        const select = screen.getByTestId("testid-combobox");
        within(select).queryByPlaceholderText(/Select tables/i);
        await userEvent.click(select);

        const allOptions = await screen.findAllByRole("option");
        const tableOptions = allOptions.slice(1); // Skip 'Add Table' option
        const selectedTable = tableOptions[0]?.textContent;
        await userEvent.click(tableOptions[0]);
        await waitFor(() => {
            expectProjectGet(`v1/tables/` + expectedTableList[11]?.id);
        });

        verifyNodeInGraph(selectedTable as string);
    });

    it("Should show table view when clicked on 'eye' icon of table node in the graph", async () => {
        await setupGraphCreationFlow();

        const tableNames = expectedTableList?.map((l) => l?.name);
        await selectTableByCheckbox(0, expectedTableList[0]?.id);

        const viewTable = await screen.findByTestId(
            tableNames[0] + "_show_icon"
        );
        fireEvent.click(viewTable);
        await waitFor(
            () => {
                expectProjectGet(`v1/tables/` + expectedTableList[0]?.id);
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    `v1/tables/${expectedTableList[0]?.id}/col_stats`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                );
            },
            { timeout: 3000 }
        );

        const table = screen.getAllByRole("table");
        const colStatsTable = table?.find(
            (t) => t?.id === "column-stats-table"
        );
        expect(colStatsTable).toBeInTheDocument();
        const tableDetail = expectedTableDetailsResponse();
        const columns = tableDetail?.table?.cols?.map((c) => c?.name);

        if (colStatsTable) {
            await verifyColumnsInTable(colStatsTable, columns);
        }

        const collapseButton = screen.getByText("Back");
        expect(collapseButton).toBeInTheDocument();
        await userEvent.click(collapseButton);
        expect(
            screen.getByTestId(tableNames[0] + "_show_icon")
        ).toBeInTheDocument();
    });

    it("Should show edit table view when clicked on edit icon of table node in the graph", async () => {
        await setupGraphCreationFlow();

        const tableNames = expectedTableList?.map((l) => l?.name);
        await selectTableByCheckbox(0, expectedTableList[0]?.id);

        const editTable = screen.getByTestId(tableNames[0] + "_edit_icon");
        fireEvent.click(editTable);

        await waitFor(() => {
            const modal = screen.getByRole("dialog");

            expect(
                within(modal).getByText(
                    "The edits made here will be saved and used as a new temp table without altering the original table."
                )
            ).toBeInTheDocument();
        });

        const saveButton = await screen.findByRole("button", {
            name: "Save"
        });
        expect(saveButton).toBeInTheDocument();
    });

    it("Should call suggest_links, display accordion with table links, and save the graph.", async () => {
        await setupGraphCreationFlow();

        await enterGraphNameAndProceed();

        const select = screen.getByTestId("testid-combobox");
        within(select).queryByPlaceholderText(/Select tables/i);
        await userEvent.click(select);

        const tableOptions = (await screen.findAllByRole("option")).slice(1); // Skip 'Add Table' option
        let transOption = null;
        for (let i = 0; i < tableOptions.length; i++) {
            if (tableOptions[i]?.textContent === "transaction_ltv1") {
                transOption = tableOptions[i];
                break;
            }
        }

        setGraphSuggestLinksResponseType(testResponseType.ALL);

        // Click on 'transaction_ltv1' from the dropdown
        await userEvent.click(transOption as Element);

        const transId = expectedTableList.find(
            (l) => l.name === "transaction_ltv1"
        );
        await waitFor(() => {
            expectProjectGet(`v1/tables/` + transId?.id);
        });

        const select1 = screen.getByRole("combobox");
        within(select1).queryByPlaceholderText(/Select tables/i);
        await userEvent.click(select1);

        // Get tableOptions for stock table
        const tableOptions1 = (await screen.findAllByRole("option")).slice(1); // Skip 'Add Table' option
        let stockOption = null;
        for (let i = 0; i < tableOptions.length; i++) {
            if (tableOptions1[i]?.textContent === "stock") {
                stockOption = tableOptions1[i];
                break;
            }
        }
        await userEvent.click(stockOption as Element); // Click on 'stock' from the dropdown

        const stockId = expectedTableList.find((l) => l.name === "stock");
        await waitFor(() => {
            expectProjectGet(`v1/tables/` + stockId?.id);
        });

        const tableDetail = expectedTableDetailsResponse();
        const stockDetail = expectedTableDetailsResponse(stockId?.name);
        const config1: Graph = {
            col_groups: [],
            tables: {
                [transId?.name as string]:
                    tableDetail?.table as unknown as TableDetails,
                [stockId?.name as string]:
                    stockDetail?.table as unknown as TableDetails
            }
        };

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v1/graphs/suggest_links`,
                config1,
                {
                    headers: { "Content-Type": "application/json" },
                    params: { project_id: "project-tests" }
                }
            );
        });

        const fwdLinkButton = screen.queryAllByTestId(/^fwdlinks-/);
        expect(fwdLinkButton).toHaveLength(1);
        fireEvent.click(fwdLinkButton[0]);

        screen.getByText("Add more links");
        screen.getByText("Table");
        screen.getByText("Column");

        const bkLinkButton = screen.queryAllByTestId(/^bklinks-/);
        expect(bkLinkButton).toHaveLength(1);

        const newGraph = expectedNewGraphResponse();
        const args = {
            "col_groups": newGraph?.col_groups ?? [],
            "tables": newGraph?.tables ?? []
        };

        await createGraphApiCallTest(args as GraphDefinition, {
            validateWithFailure: true
        });
        await createGraphApiCallTest(args as GraphDefinition, {
            validateWithInvalid: true
        });
        await createGraphApiCallTest(args as GraphDefinition, {
            createWithFailure: true
        });
        await createGraphApiCallTest(args as GraphDefinition, {
            createWithInvalid: true
        });
        await createGraphApiCallTest(args as GraphDefinition, {});
    });

    it("Should display `Please select table(s).` message when the Save button is clicked without selection of any table", async () => {
        setGraphNameResponseType(testResponseType.DEFAULT);
        await renderNewGraphPage();

        await changeGraphNameInput("new_graph");

        await findAndClickButtonByRole("Next");
        await findAndClickButtonByRole("Save");

        const errorElement = screen.getByText("Please select table(s).");
        expect(errorElement).toHaveAttribute("id", "errlabel-table");
    });

    it("Should display alert when table with primary key is not selected", async () => {
        await setupGraphCreationFlow();

        //Graph name
        await changeGraphNameInput("new_graph");

        //Select table with create date
        const allRowCheckboxes = screen.getAllByRole("checkbox", {
            name: "Select row"
        });
        const firstRowCheckbox = allRowCheckboxes[0];
        await userEvent.click(firstRowCheckbox);

        const tableId = mockTableIds.transaction_ltv1;
        await waitFor(() => {
            expectProjectGet(`v1/tables/` + tableId);
        });

        await findAndClickButtonByRole("Next");

        await waitFor(() => {
            const alert = screen.getByRole("alert");
            expect(alert).toHaveTextContent("Error");
            expect(alert).toHaveTextContent(
                /At least one of the tables need to have a primary key to create a graph. Please edit a table to add a Primary key/i
            );
        });

        const saveButton = screen.getByRole("button", { name: "Save" });
        expect(saveButton).toBeDisabled();
    });

    it("Should display alert when tables are unreachable in graph", async () => {
        await reachSaveStage();

        setGraphValidateResponseType(testResponseType.UNREACHABLE_TABLES);

        await findAndClickButtonByRole("Save");

        const alert = await screen.findByRole("alert");
        await waitFor(() => {
            expect(alert).toHaveTextContent("Graph Validation failed!");
            expect(alert).toHaveTextContent(
                "Please make sure that all tables are fully linked into a single graph"
            );
        });
    });

    it("Should disable the Save button while the validate+create round-trip is in flight", async () => {
        await reachSaveStage();

        // The validate request never resolves, keeping the submit in flight.
        setGraphValidateResponseType(testResponseType.RUNNING);

        const saveButton = screen.getByRole("button", { name: "Save" });
        expect(saveButton).toBeEnabled();
        await userEvent.click(saveButton);

        // While the round-trip is in flight the loading overlay is up; the
        // Save button must be disabled too, since the overlay only blocks
        // pointer events and the still-focused button would re-submit on a
        // keyboard Enter.
        await waitFor(() => {
            expect(screen.getByTestId("animated-loading")).toBeInTheDocument();
        });
        expect(saveButton).toBeDisabled();
    });

    it("Should fire exactly one validate round-trip for a rapid burst of Save activations", async () => {
        await reachSaveStage();

        // The validate request never resolves, keeping the submit in flight.
        setGraphValidateResponseType(testResponseType.RUNNING);

        const saveButton = screen.getByRole("button", { name: "Save" });
        expect(saveButton).toBeEnabled();

        // Simulate queued keyboard re-activations: two extra click events
        // delivered while the first activation is still dispatching, before
        // React commits the disabled/loading re-render.
        let extraActivations = 2;
        const reactivate = () => {
            if (extraActivations > 0) {
                extraActivations -= 1;
                saveButton.dispatchEvent(
                    new MouseEvent("click", { bubbles: true, cancelable: true })
                );
            }
        };
        saveButton.addEventListener("click", reactivate);
        fireEvent.click(saveButton);
        saveButton.removeEventListener("click", reactivate);

        // Let the mutation pipeline flush before counting requests.
        await act(async () => {
            await new Promise((resolve) => setTimeout(resolve, 50));
        });

        expect(countValidateCalls()).toBe(1);
    });

    it("Should disable the Next button while the graph name is server-invalid", async () => {
        await setupGraphCreationFlow();

        // A server-valid name enables Next.
        await changeGraphNameInput("new_graph");
        const nextButton = screen.getByRole("button", { name: "Next" });
        await waitFor(() => expect(nextButton).toBeEnabled());

        // The server flags the next name as already used: the field shows
        // the inline error, and the Next gate must reflect the same
        // validity instead of letting the user advance the wizard.
        setGraphNameResponseType(testResponseType.EXISTS);
        const duplicateName = "ltv_graph";
        await changeGraphNameInput(duplicateName);
        await waitFor(() => {
            expect(
                screen.getByText(
                    `Graph with name ${duplicateName} already exists. Please choose a different name.`
                )
            ).toBeInTheDocument();
        });
        expect(nextButton).toBeDisabled();

        // Editing the name re-validates it; once the server accepts the
        // new name the gate must release.
        setGraphNameResponseType(testResponseType.DEFAULT);
        await changeGraphNameInput("fresh_name");
        await waitFor(() => expect(nextButton).toBeEnabled());
    });
});
