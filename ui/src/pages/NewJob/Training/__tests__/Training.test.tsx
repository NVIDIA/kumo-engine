// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createMemoryRouter, RouterProvider } from "react-router";

import { focusManager, QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    fireEvent,
    render,
    screen,
    waitFor,
    within
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import {
    mockLocationStateValue,
    mockNavigateCall,
    mockUseNavigate
} from "tests/mockRouter";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse, createQueryClient } from "tests/testUtil";
import { vi } from "vitest";

import { expectedGraphDetailResponse } from "pages/Graphs/__tests__/MockFunction/GraphDetail.test.mock";
import { expectedGraphsListResponse } from "pages/Graphs/__tests__/MockFunction/GraphsList.test.mock";

import { copyObject, getKeys } from "utils/Helpers/objectHelpers";

import NewTrainingJob from "../Training";

import { setTestTrainResponseType } from "./MockFunctions/TrainPayload.mock";
import { setTestValidatePayloadResponseType } from "./MockFunctions/ValidatePayload.mock";
import { expectedGraphPostPayload } from "./MockResponses/GraphPayload";
import mockModelPlanResponse from "./MockResponses/ModelPlanResponse.json";
import { trainPayloadDeatails } from "./MockResponses/TrainPayload";
import { validatePayloadDeatails } from "./MockResponses/ValidatePayload";

const expectedGraphsList = expectedGraphsListResponse();
const expectedGraphDetail = expectedGraphDetailResponse();
const copiedGraph = {
    id: expectedGraphDetail.id,
    name: expectedGraphDetail.name_alias,
    tables: copyObject(expectedGraphDetail.graph?.tables ?? {}),
    groups: copyObject(expectedGraphDetail.graph?.col_groups ?? [])
};

describe("TrainingPage", () => {
    const queryClient = createQueryClient();
    const noPointerCheckUser = userEvent.setup({
        delay: null,
        pointerEventsCheck: 0
    });

    // The page registers a react-router navigation blocker (useBlocker), so
    // it must render inside a data router, like the real route tree in
    // routes/Routes.tsx does. A "/previous" history entry sits behind the
    // wizard so tests can exercise browser Back.
    let router: ReturnType<typeof createMemoryRouter>;

    const renderTrainingPage = async () => {
        let view!: ReturnType<typeof render>;
        router = createMemoryRouter(
            [
                { path: "/", element: <NewTrainingJob /> },
                {
                    path: "/previous",
                    element: <div data-testid="previous-route" />
                }
            ],
            { initialEntries: ["/previous", "/"], initialIndex: 1 }
        );
        await act(async () => {
            view = render(
                <QueryClientProvider client={queryClient}>
                    <RouterProvider router={router} />
                </QueryClientProvider>
            );
        });
        return view;
    };

    const pageLoadTest = async () => {
        screen.getByText("New Model");
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                "v1/graphs/",
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });
    };

    const graphSelection = async () => {
        const selectGraphButton = await screen.findByText("Select graph");
        fireEvent.click(selectGraphButton);

        const firstGraphOption = await screen.findByText(
            expectedGraphsList[0].name
        );
        await userEvent.click(firstGraphOption, {
            delay: null,
            pointerEventsCheck: 0
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/graphs/${expectedGraphDetail.id}`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        await verifyGraphNodes();
    };

    const verifyGraphNodes = async () => {
        const tables = getKeys(expectedGraphDetail?.graph?.tables);
        for (const table of tables) {
            screen.getByTestId(`rf__node-${table}`);
        }
    };

    const selectTable = async (searchInput: HTMLElement, tableName: string) => {
        await userEvent.clear(searchInput);
        fireEvent.change(searchInput, { target: { value: tableName } });
        const option = await screen.findByRole("option", { name: tableName });
        await userEvent.click(option);
    };

    const enterPQLQuery = async (query: string) => {
        const editorContainer = await screen.findByTestId("pql-editor");
        const codeMirrorTextBox = within(editorContainer).getByRole("textbox");
        // Use fireEvent instead of userEvent to avoid "Invalid child in posBefore"
        // error with CodeMirror in happy-dom + React 19
        codeMirrorTextBox.focus();
        const pasteEvent = new ClipboardEvent("paste", {
            clipboardData: new DataTransfer()
        });
        (pasteEvent.clipboardData as DataTransfer).setData("text/plain", query);
        fireEvent(codeMirrorTextBox, pasteEvent);
    };

    const clickCreateNewGraph = async () => {
        const selectGraphButton = screen.getByRole("button", {
            name: "Select graph"
        });
        await userEvent.click(selectGraphButton, {
            delay: null,
            pointerEventsCheck: 0
        });
        const createNewGraph = await screen.findByText("Create new graph");
        await userEvent.click(createNewGraph, {
            delay: null,
            pointerEventsCheck: 0
        });
    };

    const waitForValidationStatus = async (
        expectedText: string,
        timeout = 2000
    ) => {
        const validationStatus = screen.getByTestId("validation-status");
        await waitFor(
            () => {
                expect(validationStatus).toBeInTheDocument();
                expect(validationStatus).toHaveTextContent(expectedText);
            },
            { timeout }
        );
    };

    const createTrainApiCallTest = async ({
        createWithFailure = false,
        createWithInvalid = false
    }) => {
        if (createWithFailure) {
            setTestTrainResponseType(testResponseType.ERROR);
        } else if (createWithInvalid) {
            setTestTrainResponseType(testResponseType.INVALID);
        } else {
            setTestTrainResponseType(testResponseType.ALL);
        }

        const startTrainingButton = screen.getByTestId("start-training");

        await waitFor(() => {
            expect(startTrainingButton).toBeEnabled();
        });

        fireEvent.click(startTrainingButton);
        if (!createWithFailure && !createWithInvalid) {
            await screen.findByText("Add tag");
            screen.getByText(
                "The following jobs will be created when you start training."
            );
            expect(screen.getAllByText("Training Job").length).toBeGreaterThan(
                0
            );
            expect(
                screen.getAllByText("Training Table Generation").length
            ).toBeGreaterThan(0);
            screen.getByText(
                "Add key-value tags to organize and filter this training job. Tags will be applied to both jobs."
            );

            const addTagButton = screen.getByText("Add tag");
            await noPointerCheckUser.click(addTagButton);
            const keyInput = await screen.findByPlaceholderText("key");
            fireEvent.change(keyInput, { target: { value: "env" } });
            const valueInput = screen.getByPlaceholderText("value");
            fireEvent.change(valueInput, { target: { value: "prod" } });
            await userEvent.keyboard("{Enter}");
        }
        const secondStartTrainingButton = document.getElementById("start-job");
        await waitFor(
            async () => {
                expect(secondStartTrainingButton).toBeEnabled();
            },
            { timeout: 2000 }
        );
        await noPointerCheckUser.click(secondStartTrainingButton!);

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                "v0/training_jobs/train",
                expect.objectContaining(trainPayloadDeatails),
                {
                    headers: {
                        "Content-Type": "application/json"
                    }
                }
            );
        });

        if (createWithFailure || createWithInvalid) {
            await screen.findByText("Request failed!");
            const closeButton = screen.getByTestId("cancel-job");
            await noPointerCheckUser.click(closeButton);
        }
    };

    beforeEach(() => {
        mockLocationStateValue(null);
        queryClient.clear();
        vi.clearAllMocks();
        mockNavigateCall();
        // Stop background refetch timers from firing across tests, which
        // otherwise accumulate and inflate later tests' act() drains.
        focusManager.setFocused(false);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
    });

    const validateQueryAPI = async (payload: object) => {
        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v1/predictive_queries/validate`,
                expect.objectContaining(payload),
                {
                    headers: {
                        "Content-Type": "application/json"
                    },
                    params: { project_id: "project-tests" }
                }
            );
        });
    };

    it("Should navigate back to Models List page when Cancel button is clicked", async () => {
        await renderTrainingPage();
        await pageLoadTest();

        const closeButton = screen.getByTestId("cancel-job");
        await noPointerCheckUser.click(closeButton);

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                "/project/project-tests/models/?page=1&job_type=TRAINING_JOB",
                { replace: true }
            );
        });
    });

    // kumo-ai/kumo#30141: the unsaved-changes guard must cover every exit
    // path (browser Back / route changes / page unload), not only the Cancel
    // button.
    it("Should block route changes while the form is dirty until the discard dialog is confirmed", async () => {
        await renderTrainingPage();
        await pageLoadTest();

        const query = "PREDICT customers.age\nFOR EACH customers.customer_id";
        await enterPQLQuery(query);

        // Browser Back on the dirty form: blocked, and the same "Discard
        // this model?" confirmation the Cancel button uses shows up.
        await act(async () => {
            await router.navigate(-1);
        });
        expect(router.state.location.pathname).toBe("/");
        await screen.findByText("Discard this model?");

        // Keep editing: the dialog closes and the wizard stays mounted.
        await noPointerCheckUser.click(document.getElementById("close")!);
        await waitFor(() => {
            expect(
                screen.queryByText("Discard this model?")
            ).not.toBeInTheDocument();
        });
        expect(router.state.location.pathname).toBe("/");
        screen.getByTestId("new-model-page-heading");

        // Back again, this time confirming the discard: the blocked
        // navigation proceeds to the previous history entry.
        await act(async () => {
            await router.navigate(-1);
        });
        await screen.findByText("Discard this model?");
        await noPointerCheckUser.click(document.getElementById("discard")!);
        await waitFor(() => {
            expect(router.state.location.pathname).toBe("/previous");
        });
        await screen.findByTestId("previous-route");
    });

    it("Should navigate freely without any discard prompt while the form is clean", async () => {
        await renderTrainingPage();
        await pageLoadTest();

        await act(async () => {
            await router.navigate(-1);
        });

        expect(
            screen.queryByText("Discard this model?")
        ).not.toBeInTheDocument();
        expect(router.state.location.pathname).toBe("/previous");
        await screen.findByTestId("previous-route");
    });

    it("Should register the beforeunload prompt exactly while the form is dirty and remove it on unmount", async () => {
        const addSpy = vi.spyOn(window, "addEventListener");
        const removeSpy = vi.spyOn(window, "removeEventListener");
        const beforeUnloadAdds = () =>
            addSpy.mock.calls.filter(([type]) => type === "beforeunload");
        const beforeUnloadRemovals = () =>
            removeSpy.mock.calls.filter(([type]) => type === "beforeunload");

        try {
            const view = await renderTrainingPage();
            await pageLoadTest();

            // Clean form: no handler registered, unload is not intercepted.
            expect(beforeUnloadAdds()).toHaveLength(0);
            const cleanUnload = new Event("beforeunload", {
                cancelable: true
            });
            window.dispatchEvent(cleanUnload);
            expect(cleanUnload.defaultPrevented).toBe(false);

            // Dirty form: the handler is registered and cancels the unload,
            // which is what makes the browser show its confirmation.
            const query =
                "PREDICT customers.age\nFOR EACH customers.customer_id";
            await enterPQLQuery(query);
            await waitFor(() => {
                expect(beforeUnloadAdds()).toHaveLength(1);
            });
            const dirtyUnload = new Event("beforeunload", {
                cancelable: true
            });
            window.dispatchEvent(dirtyUnload);
            expect(dirtyUnload.defaultPrevented).toBe(true);

            // Unmount: the handler is removed again.
            view.unmount();
            expect(beforeUnloadRemovals()).toHaveLength(1);
            const goneUnload = new Event("beforeunload", { cancelable: true });
            window.dispatchEvent(goneUnload);
            expect(goneUnload.defaultPrevented).toBe(false);
        } finally {
            addSpy.mockRestore();
            removeSpy.mockRestore();
        }
    });

    it("Should select a graph, enter a PQL query, validate it, check model plan and start training", async () => {
        await renderTrainingPage();
        await pageLoadTest();

        await graphSelection();

        const Textbox = screen.getByRole("textbox");
        expect(Textbox).toHaveTextContent(
            "Start typing your predictive query here..."
        );
        // Use fireEvent.click to avoid CodeMirror issues with happy-dom + React 19
        fireEvent.click(Textbox);

        const query = "PREDICT customers.age\nFOR EACH customers.customer_id";
        await enterPQLQuery(query);

        const graph = copyObject(expectedGraphDetail.graph);
        const tmpGraph = {
            col_groups:
                graph?.col_groups?.map((c) => ({ ["columns"]: c.columns })) ??
                [],
            tables: graph?.tables ?? {}
        };
        // Payloads for validate query
        const payload = {
            pquery_resource: {
                name: expect.stringMatching(/^test_pq_\d+$/),
                query_string:
                    validatePayloadDeatails.pquery_resource.query_string,
                graph: tmpGraph
            },
            model_plan: ""
        };

        // Check api validate call
        await validateQueryAPI(payload);

        // Check model_plan call triggered after successful validation
        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                "v1/predictive_queries/model_plan",
                expect.objectContaining({
                    run_mode: "fast",
                    graph_id: "",
                    query_string:
                        "PREDICT customers.age\nFOR EACH customers.customer_id"
                }),
                expect.objectContaining({
                    params: {
                        dialect: "PQLANG_V2",
                        project_id: "project-tests"
                    }
                })
            );
        });

        const modelSetting = await screen.findByTestId("tab-model-settings"); // Switch to Model Settings tab
        await noPointerCheckUser.click(modelSetting);

        expect(
            screen.queryByText("Baseline Comparison")
        ).not.toBeInTheDocument();
        expect(screen.queryByRole("switch")).not.toBeInTheDocument();

        //check run modes
        await screen.findByTestId("testid-run-mode-select");

        screen.getByText(
            /The Run Mode field is used to determine the default values for your model's Model Plan based on common usage patterns/i
        );

        const runModeButton = screen.getByTestId("testid-run-mode-select");

        await noPointerCheckUser.click(runModeButton);

        const options = await screen.findAllByRole("option");
        const tags = options
            .map((r) => r.getAttribute("data-value"))
            .join(", ");
        const runModeOptions = ["FAST", "BEST", "NORMAL"];
        for (const runMode of runModeOptions) {
            expect(tags).contain(runMode);
        }

        const fastButton = options.find(
            (r) => r.getAttribute("data-value") === "FAST"
        );
        expect(fastButton).toBeInTheDocument();
        await noPointerCheckUser.click(fastButton as HTMLElement);

        // Check api validate call
        await validateQueryAPI(payload);

        // Check model_plan call triggered after successful validation
        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                "v1/predictive_queries/model_plan",
                expect.objectContaining({
                    run_mode: "fast",
                    graph_id: "",
                    query_string:
                        "PREDICT customers.age\nFOR EACH customers.customer_id"
                }),
                expect.objectContaining({
                    params: {
                        dialect: "PQLANG_V2",
                        project_id: "project-tests"
                    }
                })
            );
        });

        // Switch to Model Planner tab
        const modelPlanner = screen.getByTestId("tab-model-planner");
        await userEvent.click(modelPlanner);

        // Model Planner form should show section headers once YAML is loaded
        screen.getByText("Training Table Generation");
        await validateQueryAPI(payload);
        await createTrainApiCallTest({
            createWithFailure: false,
            createWithInvalid: false
        });

        await waitForValidationStatus("Looks good!", 5000);

        await createTrainApiCallTest({
            createWithFailure: true,
            createWithInvalid: true
        });
    });

    it("Should confirm absence of graph structure and open graph creation section upon clicking 'Create new graph'", async () => {
        await renderTrainingPage();
        await pageLoadTest();

        screen.getByText("Select a graph");

        await waitFor(() => {
            for (const table of getKeys(expectedGraphDetail?.graph?.tables)) {
                const node = screen.queryByTestId(`rf__node-${table}`);
                expect(node).not.toBeInTheDocument();
            }
        });

        await clickCreateNewGraph();

        const saveToReuse = screen.getByText("Save graph to reuse later");
        expect(saveToReuse).toBeInTheDocument();
        const checkbox = document.querySelector('[data-slot="checkbox"]');
        if (checkbox) {
            await noPointerCheckUser.click(checkbox);
            screen.getByText("Graph name is needed if saving for later use");
        } else {
            throw new Error("Reuse checkbox not found");
        }

        screen.getByText(/Remove graph/i);

        screen.getByText("Tables & links");
        screen.getByText(
            "You can change the tables and update links between them, the changes will be saved as a new graph without altering the original graph."
        );

        const selectTablesButton = await screen.findByRole("button", {
            name: "Select tables"
        });
        expect(selectTablesButton).toBeInTheDocument();
        await userEvent.click(selectTablesButton);

        const tablesToSelect = [
            {
                tableName: "stock",
                id: "NA.NA-b656de36982dfe8ea68ab7ab1ebac6a1"
            },
            {
                tableName: "transaction_ltv1",
                id: "NA.NA-18e2b1a6e499fff45a6f249f3024662f"
            }
        ];

        for (const { tableName, id } of tablesToSelect) {
            const allOptions = await screen.findAllByText(tableName);
            const option = allOptions[0];
            await userEvent.click(option);
            // Wait for all Axios calls in sequence
            await waitFor(async () => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    `v1/tables/${id}`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                );
            });
        }

        await waitFor(async () => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v1/graphs/suggest_links`,
                { tables: expectedGraphPostPayload.tables, col_groups: [] },
                {
                    headers: { "Content-Type": "application/json" },
                    params: { project_id: "project-tests" }
                }
            );
        });

        await waitFor(
            async () => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    `v1/graphs/validate`,
                    expectedGraphPostPayload,
                    {
                        headers: { "Content-Type": "application/json" },
                        params: { project_id: "project-tests" }
                    }
                );
            },
            { timeout: 2000 }
        );

        await waitFor(async () => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v1/graphs/`,
                expectedGraphPostPayload,
                {
                    headers: { "Content-Type": "application/json" },
                    params: {
                        name_alias: "",
                        force_rename: false,
                        project_id: "project-tests"
                    }
                }
            );
        });
    });

    it("Should display an error alert when selecting tables without a primary key", async () => {
        await renderTrainingPage();
        await pageLoadTest();

        screen.getByText("Select a graph");

        await clickCreateNewGraph();

        const selectTablesButton = await screen.findByRole("button", {
            name: "Select tables"
        });

        await userEvent.click(selectTablesButton);

        await screen.findByText("Add new table");

        const indices = [13, 1, 6];

        for (const index of indices) {
            const option = screen.getByTestId(`combobox-option-${index}`);
            await userEvent.click(option);
        }

        const titleEl = document.querySelector('[data-slot="alert-title"]');

        await waitFor(() => {
            expect(titleEl?.textContent).toBe("Error");
        });

        const descriptionEl = document.querySelector(
            '[data-slot="alert-description"]'
        );

        const normalizedText = descriptionEl?.textContent
            ?.replace(/\s+/g, " ")
            .trim();

        expect(normalizedText).toContain(
            "At least one of the tables need to have a primary key to create a graph. Please edit a table to add a Primary key"
        );
    });

    it("Should display the graph structure and connected tables & links upon selecting a graph from the 'Select Graph' dropdown list", async () => {
        await renderTrainingPage();
        await pageLoadTest();

        await graphSelection();

        // Extract table names and pkeys from the mocked graph detail response
        const targetTableNames = ["stock", "customers"];
        const filteredTables = expectedGraphDetail.graph
            ? Object.entries(expectedGraphDetail.graph.tables).filter(
                  ([name]) => targetTableNames.includes(name)
              )
            : [];
        const tableNames = filteredTables.map(([name]) => name);

        const tablePkeys = expectedGraphDetail.graph
            ? Object.values(expectedGraphDetail.graph.tables)
                  .map((table) => table.pkey)
                  .filter(Boolean) // removes undefined/null
            : [];

        // Check that all table names are rendered
        const tableForwardLinkElements = screen.queryAllByTestId(/^fwdlinks-/);
        expect(tableForwardLinkElements.length).toBeGreaterThan(0);

        const tableLinkTextsJoin = tableForwardLinkElements
            .map((el) => el.textContent)
            .join(" ");

        // Assert each expected table name is present
        tableNames.forEach((tableName) => {
            expect(tableLinkTextsJoin).toContain(tableName);
        });

        // Assert each expected table pkey name is present
        tablePkeys.forEach((pkey) => {
            expect(tableLinkTextsJoin).toContain(pkey);
        });

        const expectedText = `${tableNames[0]} ${tablePkeys[0]} 1 Link`;
        expect(tableForwardLinkElements[0]).toHaveTextContent(expectedText);
        await userEvent.click(tableForwardLinkElements[0]);

        const buttonText = `${expectedGraphDetail.num_tables} Tables`;
        const addTableButton = await screen.findByText(buttonText);
        expect(addTableButton).toBeInTheDocument();

        await userEvent.click(addTableButton);
        await waitFor(() => {
            screen.getByText("Add new table");

            const tableBackLinkElements = screen.getAllByTestId("bklinks-0");
            expect(tableBackLinkElements).toHaveLength(1);
        });
    });

    it("Should show a 'Graph already exists' alert when the 'Start training' button is clicked and the graph already exists", async () => {
        setTestTrainResponseType(testResponseType.ALL);
        await renderTrainingPage();
        await pageLoadTest();

        const selectGraphButton = screen.getByRole("button", {
            name: "Select graph"
        });
        await userEvent.click(selectGraphButton);

        const createGraphOption = await screen.findByTestId(
            "select-graph-option-0"
        );
        await userEvent.click(createGraphOption);

        const saveGraphToReuseCheckbox = document.querySelector(
            '[data-slot="checkbox"]'
        )!;
        await userEvent.click(saveGraphToReuseCheckbox);

        const selectTables = screen.getByRole("button", {
            name: "Select tables"
        });
        await userEvent.click(selectTables);

        const searchInput = screen.getByPlaceholderText("Search...");
        await selectTable(searchInput, "articles");
        await selectTable(searchInput, "customers");
        await selectTable(searchInput, "transactions");

        await waitFor(
            async () => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    `v1/graphs/${expectedGraphDetail.id}`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                );
            },
            { timeout: 2000 }
        );

        await userEvent.click(screen.getByTestId("testid-combobox")); //close select tables dropdown

        const graphNameInput = screen.getByPlaceholderText(
            "Please enter graph name"
        );
        const nameInput = "test_graph";
        fireEvent.change(graphNameInput, { target: { value: nameInput } });

        const Textbox = await screen.findAllByRole("textbox");
        // Use fireEvent.click to avoid CodeMirror issues with happy-dom + React 19
        fireEvent.click(Textbox[0]);
        expect(Textbox[0]).toBeInTheDocument();

        const query = "PREDICT customers.age\nFOR EACH customers.customer_id";
        await enterPQLQuery(query);

        await waitForValidationStatus("Looks good!", 5000);

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v1/predictive_queries/validate`,
                expect.anything(),
                {
                    headers: { "Content-Type": "application/json" },
                    params: { project_id: "project-tests" }
                }
            );
        });

        const startButton = screen.getByRole("button", {
            name: "Start training"
        });
        await waitFor(
            () => {
                expect(startButton).toBeEnabled();
            },
            { timeout: 15000 }
        );
        await userEvent.click(startButton, { pointerEventsCheck: 0 });

        // Wait for the StartJobPopover to open
        await waitFor(() => {
            screen.getByText(
                "The following jobs will be created when you start training."
            );
        });

        // Click the start button inside the popover to trigger startTraining
        const startJobButton = document.getElementById("start-job");
        await waitFor(
            () => {
                expect(startJobButton).toBeEnabled();
            },
            { timeout: 2000 }
        );
        await userEvent.click(startJobButton!);

        await waitFor(() => {
            const existingGraphName = "ltv_graph";
            const alert = screen.getByTestId("alert-dialog-content");
            expect(alert).toHaveTextContent("Graph already exists!");
            expect(alert).toHaveTextContent(
                `Graph already exists with name ${existingGraphName}, Please uncheck 'Save graph to reuse later' option and try again!`
            );
        });
    });

    it("Should display 'Empty Query' when No Predictive Query is entered", async () => {
        await renderTrainingPage();
        await pageLoadTest();

        await graphSelection();

        const Textbox = screen.getByRole("textbox");
        expect(Textbox).toHaveTextContent(
            "Start typing your predictive query here..."
        );

        await waitForValidationStatus("Empty Query");
    });

    it("Should display 'Issues found' when Invalid Predictive Query is entered", async () => {
        await renderTrainingPage();
        await pageLoadTest();

        await graphSelection();

        const Textbox = screen.getByRole("textbox");
        // Use fireEvent.click to avoid CodeMirror issues with happy-dom + React 19
        fireEvent.click(Textbox);

        const query = "PREDICT customers.age\nFOR EACH customers.customer_ids";
        setTestValidatePayloadResponseType(testResponseType.INVALID);
        await enterPQLQuery(query);
        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalled();
        });
        await waitForValidationStatus("Issues found");

        const validationStatus = screen.getByTestId("validation-status");
        await userEvent.hover(validationStatus);

        const validationCard = screen.getByTestId("validation-card");
        expect(validationCard).toHaveTextContent(/Invalid Predictive Query/i);
        expect(validationCard).toHaveTextContent(
            /Encountered the following issues during parsing: Errors:/i
        );
        expect(validationCard).toHaveTextContent(
            /1. Invalid Syntax. The target \(PREDICT\) clause in this query is empty or invalid./i
        );

        setTestValidatePayloadResponseType(testResponseType.DEFAULT);
    });

    it("Should render the initial graph nodes and deselect the table, then confirm the presence of the nodes. ", async () => {
        await renderTrainingPage();
        await pageLoadTest();

        await graphSelection();

        // Extract table names from the mocked graph detail response
        const graphTableNames = expectedGraphDetail.graph
            ? Object.keys(expectedGraphDetail.graph.tables)
            : [];

        const buttonText = `${expectedGraphDetail.num_tables} Tables`;
        const addTableButton = screen.getByText(buttonText);
        expect(addTableButton).toBeInTheDocument();
        await userEvent.click(addTableButton);

        const addTableListBox = await screen.findByRole("listbox");
        expect(addTableListBox).toBeInTheDocument();

        // Wait for first option to ensure dropdown is rendered
        await within(addTableListBox).findByRole("option", {
            name: graphTableNames[0]
        });

        for (const tableName of graphTableNames) {
            const option = within(addTableListBox).getByRole("option", {
                name: tableName
            });
            expect(option).toBeInTheDocument();
        }

        // Desect one table from the list
        const tableToDeselect = graphTableNames[0];
        const tableOption = within(addTableListBox).getByRole("option", {
            name: tableToDeselect
        });

        await userEvent.click(tableOption);

        // Wait until only 2 nodes are rendered
        await waitFor(() => {
            const renderedNodes = screen.queryAllByTestId(/^rf__node-/);
            expect(renderedNodes).toHaveLength(2);
        });

        // Confirm the deselected node is not present
        await waitFor(() => {
            const removedNode = screen.queryByTestId(
                `rf__node-${tableToDeselect}`
            );
            expect(removedNode).not.toBeInTheDocument();
        });
    });

    it("Should validate tag input and support add/remove interactions in StartJobPopover", async () => {
        setTestValidatePayloadResponseType(testResponseType.DEFAULT);
        await renderTrainingPage();
        await pageLoadTest();
        await graphSelection();

        fireEvent.click(screen.getByRole("textbox"));
        await enterPQLQuery(
            "PREDICT customers.age\nFOR EACH customers.customer_id"
        );

        await waitFor(
            () => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    "v1/predictive_queries/model_plan",
                    expect.anything(),
                    expect.objectContaining({
                        params: {
                            dialect: "PQLANG_V2",
                            project_id: "project-tests"
                        }
                    })
                );
                screen.getByTestId("infer-task-type");
                expect(
                    screen.getByRole("button", { name: "Start training" })
                ).toBeEnabled();
            },
            { timeout: 20000 }
        );
        await userEvent.click(
            screen.getByRole("button", { name: "Start training" })
        );

        // Invalid key characters
        await userEvent.click(await screen.findByText("Add tag"));
        await userEvent.type(
            await screen.findByPlaceholderText("key"),
            "invalid@key"
        );
        screen.getByText(
            "Only alphanumeric characters, underscores (_), or hyphens (-) are allowed in key"
        );

        // Invalid value characters
        await userEvent.clear(screen.getByPlaceholderText("key"));
        await userEvent.type(screen.getByPlaceholderText("key"), "env");
        await userEvent.type(
            screen.getByPlaceholderText("value"),
            "invalid@value"
        );
        screen.getByText(
            "Only alphanumeric characters, underscores (_), or hyphens (-) are allowed in value"
        );

        // Missing key
        await userEvent.clear(screen.getByPlaceholderText("key"));
        await userEvent.clear(screen.getByPlaceholderText("value"));
        await userEvent.type(screen.getByPlaceholderText("value"), "prod");
        await userEvent.keyboard("{Enter}");
        screen.getByText("Key is required");

        // Missing value
        await userEvent.clear(screen.getByPlaceholderText("key"));
        await userEvent.clear(screen.getByPlaceholderText("value"));
        await userEvent.type(screen.getByPlaceholderText("key"), "env");
        await userEvent.keyboard("{Enter}");
        screen.getByText("Value is required");

        // Add a valid tag
        await userEvent.clear(screen.getByPlaceholderText("key"));
        await userEvent.type(screen.getByPlaceholderText("key"), "env");
        await userEvent.type(screen.getByPlaceholderText("value"), "prod");
        await userEvent.keyboard("{Enter}");
        screen.getByText("env=");
        screen.getByText("prod");
        expect(screen.queryAllByTestId("tag-remove-icon")).toHaveLength(1);

        // Duplicate key error
        await userEvent.click(screen.getByText("Add tag"));
        await userEvent.type(await screen.findByPlaceholderText("key"), "env");
        await userEvent.type(screen.getByPlaceholderText("value"), "staging");
        await userEvent.keyboard("{Enter}");
        screen.getByText('Tag key "env" already exists');

        // Remove tag
        await userEvent.click(screen.getByTestId("tag-remove-icon"));
        expect(screen.queryByText("env=")).not.toBeInTheDocument();
        expect(screen.queryAllByTestId("tag-remove-icon")).toHaveLength(0);

        // Add multiple tags and remove all
        const addMultipleTag = async (key: string, value: string) => {
            await userEvent.click(await screen.findByText("Add tag"));
            await userEvent.type(
                await screen.findByPlaceholderText("key"),
                key
            );
            await userEvent.type(screen.getByPlaceholderText("value"), value);
            await userEvent.keyboard("{Enter}");
        };
        await addMultipleTag("env", "prod");
        await addMultipleTag("team", "eng");
        await addMultipleTag("region", "us");
        expect(screen.queryAllByTestId("tag-remove-icon")).toHaveLength(3);
        for (const btn of screen.queryAllByTestId("tag-remove-icon")) {
            await userEvent.click(btn);
        }
        expect(screen.queryAllByTestId("tag-remove-icon")).toHaveLength(0);
    });

    it("shows and edits copied tags without dropping them when adding tags", async () => {
        const copiedModelPlanYaml =
            mockModelPlanResponse.suggested_model_plan_yaml.replace(
                "num_experiments: 4",
                "num_experiments: 2"
            );
        mockLocationStateValue({
            taskType: "regression",
            copytrainingtags: { bp: "bp", team: "ml" },
            copytrainingargs: {
                graph_id: expectedGraphDetail.id,
                graph_name: "Stale graph label",
                graph: {
                    ...copiedGraph,
                    tables: { customers: copiedGraph.tables.customers },
                    groups: []
                },
                query_yaml:
                    "PREDICT customers.age\nFOR EACH customers.customer_id",
                model_planner_yaml: copiedModelPlanYaml
            }
        });
        await renderTrainingPage();

        await screen.findByText(expectedGraphDetail.name_alias as string);
        await waitForValidationStatus("Looks good!");
        expect(screen.getByTestId("infer-task-type")).toHaveTextContent(
            "Task type: regression"
        );
        await userEvent.click(screen.getByTestId("tab-model-planner"));
        await userEvent.click(screen.getByRole("tab", { name: "YAML" }));
        expect(document.getElementById("YAML")).toHaveTextContent(
            "num_experiments: 2"
        );
        expect(mockAxiosInstance.post).not.toHaveBeenCalledWith(
            "v1/predictive_queries/model_plan",
            expect.anything(),
            expect.anything()
        );

        await userEvent.click(screen.getByTestId("start-training"));
        expect(screen.getByText("bp=")).toBeVisible();
        expect(screen.getByText("team=")).toBeVisible();

        const teamTag = screen
            .getByText("team=")
            .closest('[data-slot="tag"]') as HTMLElement;
        await userEvent.click(within(teamTag!).getByTestId("tag-remove-icon"));
        await userEvent.click(screen.getByText("Add tag"));
        await userEvent.type(screen.getByPlaceholderText("key"), "team");
        await userEvent.type(screen.getByPlaceholderText("value"), "platform");
        await userEvent.keyboard("{Enter}");

        await userEvent.click(screen.getByText("Add tag"));
        await userEvent.type(screen.getByPlaceholderText("key"), "env");
        await userEvent.type(screen.getByPlaceholderText("value"), "prod");
        await userEvent.keyboard("{Enter}");

        await noPointerCheckUser.click(document.getElementById("start-job")!);
        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                "v0/training_jobs/train",
                expect.objectContaining({
                    job_tags: { bp: "bp", team: "platform", env: "prod" }
                }),
                expect.anything()
            );
        });
    });

    it("Should show pending task-type state, keep the result through re-validation, and clear it on edit", async () => {
        await renderTrainingPage();
        await pageLoadTest();
        await graphSelection();

        const originalPost = mockAxiosInstance.post.getMockImplementation()!;
        const modelPlan = Promise.withResolvers<unknown>();
        mockAxiosInstance.post
            .mockImplementationOnce(originalPost)
            .mockImplementationOnce(
                () => modelPlan.promise as ReturnType<typeof originalPost>
            );

        fireEvent.click(screen.getByRole("textbox"));
        await enterPQLQuery(
            "PREDICT customers.age\nFOR EACH customers.customer_id"
        );

        await waitForValidationStatus("Looks good!");
        expect(screen.queryByTestId("infer-task-type")).not.toBeInTheDocument();
        expect(screen.getByTestId("task-type-pending")).toBeInTheDocument();

        const revalidation = Promise.withResolvers<unknown>();
        mockAxiosInstance.post.mockImplementationOnce(
            () => revalidation.promise as ReturnType<typeof originalPost>
        );
        await act(async () => {
            modelPlan.resolve(createMockResponse(mockModelPlanResponse));
        });

        await waitForValidationStatus("Checking query");
        expect(screen.getByTestId("infer-task-type")).toHaveTextContent(
            "Task type: regression"
        );
        expect(
            screen.queryByTestId("task-type-pending")
        ).not.toBeInTheDocument();

        const editedValidation = Promise.withResolvers<unknown>();
        mockAxiosInstance.post.mockImplementationOnce(
            () => editedValidation.promise as ReturnType<typeof originalPost>
        );
        await enterPQLQuery("\nWHERE customers.age > 0");
        await waitFor(() =>
            expect(
                screen.queryByTestId("infer-task-type")
            ).not.toBeInTheDocument()
        );
        expect(screen.getByTestId("task-type-pending")).toBeInTheDocument();

        const requestCount = mockAxiosInstance.post.mock.calls.length;
        await act(async () => {
            revalidation.resolve(createMockResponse({ items: [] }));
        });
        expect(mockAxiosInstance.post).toHaveBeenCalledTimes(requestCount);
        expect(screen.queryByTestId("infer-task-type")).not.toBeInTheDocument();
        expect(screen.getByTestId("task-type-pending")).toBeInTheDocument();
    });
});
