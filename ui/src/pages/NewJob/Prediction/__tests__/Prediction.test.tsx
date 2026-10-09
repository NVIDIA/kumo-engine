// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { focusManager, QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    fireEvent,
    render,
    screen,
    waitFor,
    within
} from "@testing-library/react";
import { userEvent as createUserEvent } from "@testing-library/user-event";
import { setDeploymentTypeInStore } from "stores/__tests__/debugInfoStore";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import { testResponseType } from "tests/testResponseType";
import { deploymentType } from "tests/testType";
import { createQueryClient } from "tests/testUtil";

import { expectedNewPredictionJobTableHeader } from "components/Jobs/Training/__tests__/MockFunctions/TrainingJobWorkflow.mock";
import { expectedGraphSnapshotResponse } from "../../../../components/Jobs/Graph/__tests__/MockFunctions/GraphSnapshot.mock";
import mockWorkflowResponse from "../../../../components/Jobs/Training/__tests__/MockResponses/TrainingWorkflow.test.json";

import { setConnectorListResponseType } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorList.mock";
import { setGraphsListResponseType } from "pages/Graphs/__tests__/MockFunction/GraphsList.test.mock";
import NewPredictionJob from "pages/NewJob/Prediction/Prediction";

import { CreatePredictionJobArgs } from "types/predictionJob";

import { getKeys } from "utils/Helpers/objectHelpers";

import {
    expectedJobsListResponse,
    setJobsTestResponseType
} from "../../../Jobs/__tests__/MockFunctions/JobList.mock";

import {
    expectedNewBpJobResponse,
    setBatchPredictionJobsResponseType
} from "./MockFunction/BatchPredictionJobs.mock";
import { setTestInsertFilterOverridesResponseType } from "./MockFunction/InsertFilterOverrides.mock";
import { new_pred_get_many } from "./MockResponse/GetManyPayload";

interface PageLoadTestArg {
    page: number;
    status?: Array<string>;
    search_keyword?: string;
    checkSkeleton?: boolean;
    checkApiCall?: boolean;
    checkTable?: boolean;
    checkEmptyTable?: boolean;
}

// Delay-free userEvent: dropping the inter-event setTimeout (delay: null) cuts
// wall-clock on these interaction-heavy E2E tests without changing behaviour.
// No fake timers in this file, so a module-scope instance is safe and reused.
const userEvent = createUserEvent.setup({ delay: null });

describe("NewPredictionJobPage", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });

    const defaultLimit = Math.max(
        1,
        Math.floor((window.innerHeight - 285) / 44)
    );
    const outputDestinationOptions = [
        { value: "S3", label: "S3" },
        { value: "Snowflake", label: "Snowflake" },
        { value: "Databricks", label: "Databricks" },
        { value: "Bigquery", label: "BigQuery" },
        { value: "LocalDownloadOnly", label: "Local Download Only" }
    ];

    const queryClient = createQueryClient();

    const renderNewPredictionJobPage = async () => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={[
                        "/project/project-tests/newjob/prediction/"
                    ]}>
                    <QueryClientProvider client={queryClient}>
                        <NewPredictionJob />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const pageLoadTest = async ({
        page = 0,
        status = ["DONE"],
        search_keyword = "",
        checkSkeleton = true,
        checkApiCall = true,
        checkTable = true,
        checkEmptyTable = false
    }: PageLoadTestArg) => {
        const pageTitle = screen.getByText("New Prediction");

        expect(
            screen.getByText(
                "Select a training job to generate predictions on the latest data"
            )
        ).toBeInTheDocument();
        const expectedJobsList = expectedJobsListResponse(
            page,
            defaultLimit,
            status,
            search_keyword,
            ["TRAINING_JOB"]
        ) as string[];
        expect(pageTitle).toBeInTheDocument();

        if (checkSkeleton) {
            const skeleton = screen.queryByTestId("table-skeleton");
            const skeletonTableRows = screen.getAllByRole("row");
            expect(skeletonTableRows.length).toBe(6);
            await waitFor(() => {
                expect(skeleton).not.toBeInTheDocument();
            });
        }

        if (checkApiCall) {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                "v0/jobs/",
                {
                    "job_type": ["TRAINING_JOB"],
                    "limit": defaultLimit,
                    "page": page,
                    "project_id": "project-tests",
                    "search_keyword": "",
                    "sort_by": undefined,
                    "sort_desc": undefined,
                    "status": ["DONE"],
                    "tags": {},
                    "time_range_end": undefined,
                    "time_range_start": undefined
                },
                {
                    headers: {
                        "Content-Type": "application/json"
                    }
                }
            );
        }

        if (checkTable && expectedJobsList.length > 0) {
            const jobsListTable = screen.getByRole("table");
            const jobsListTableRows = screen.getAllByRole("row");

            expect(jobsListTable).toBeInTheDocument();
            expect(jobsListTableRows).toHaveLength(expectedJobsList.length + 1);

            expect(jobsListTableRows[0]).toHaveTextContent(
                expectedNewPredictionJobTableHeader
            );

            for (let i = 1; i <= expectedJobsList.length; i++) {
                const expectedJobId = expectedJobsList[i - 1].slice(0, 5);
                expect(jobsListTableRows[i]).toHaveTextContent(expectedJobId);
            }
        }

        if (checkEmptyTable) {
            await waitFor(() => {
                expect(screen.getByText("Welcome! 🎉")).toBeInTheDocument();
            });
        }
    };

    const selectTrainingJob = async () => {
        const tableHeader = screen.getByTestId("table-header");
        expect(tableHeader).toHaveTextContent("Job ID");

        const jobID = "9f8b2";
        const allRows = screen.getAllByRole("row");
        const firstRow = allRows[1];
        const cells = within(firstRow).getAllByRole("cell");
        const idCell = cells[0];

        await userEvent.click(idCell);
        await waitFor(() => {
            expect(screen.getByText(jobID)).toBeInTheDocument();
        });
    };

    const waitForValidationStatus = async (
        expectedText: string,
        timeout = 3000
    ) => {
        await waitFor(
            async () => {
                expect(
                    await screen.findByTestId("validation-status")
                ).toHaveTextContent(expectedText);
            },
            { timeout }
        );
    };

    const enableEmbeddings = async () => {
        const embeddingsCheckbox = screen.getByRole("checkbox", {
            name: /EMBEDDINGS/i
        });
        expect(embeddingsCheckbox).not.toBeChecked();
        await userEvent.click(embeddingsCheckbox);
        expect(embeddingsCheckbox).toBeChecked();
    };

    const clickStartPredictingButton = async () => {
        const startPredictingButton = await waitFor(() => {
            const button = screen.getByRole("button", {
                name: "Start predicting"
            });
            expect(button).toBeEnabled();
            return button;
        });
        await userEvent.click(startPredictingButton);
    };

    const clickSecondStartButton = async () => {
        const secondStartPredictingButton = await waitFor(() => {
            const button = document.getElementById("start-job");
            expect(button).toBeInTheDocument();
            return button;
        });
        expect(secondStartPredictingButton).toBeEnabled();
        await userEvent.click(secondStartPredictingButton!);
    };

    const addTagToJob = async (key: string, value: string) => {
        const addTagButton = screen.getByText("Add tag");
        await userEvent.click(addTagButton);
        const keyInput = await screen.findByPlaceholderText("key");
        fireEvent.change(keyInput, { target: { value: key } });
        const valueInput = screen.getByPlaceholderText("value");
        fireEvent.change(valueInput, { target: { value: value } });
        await userEvent.keyboard("{Enter}");
    };

    const closeErrorModal = async () => {
        const closeButtons = screen.getAllByRole("button", {
            name: "Cancel"
        });
        const closeButton = closeButtons[0];
        await userEvent.click(closeButton);
    };

    const createNewBpJobApiCallTest = async (
        args: CreatePredictionJobArgs,
        {
            createWithFailure = false,
            createWithInvalid = false,
            bpJobId = "bp-job-349fee63191149bf85731f7bb7ffcf45"
        }
    ) => {
        if (createWithFailure) {
            setBatchPredictionJobsResponseType(testResponseType.ERROR);
        } else if (createWithInvalid) {
            setBatchPredictionJobsResponseType(testResponseType.INVALID);
        } else {
            setBatchPredictionJobsResponseType(testResponseType.ALL);
        }

        await clickStartPredictingButton();

        if (!createWithFailure && !createWithInvalid) {
            expect(await screen.findByText("Add tag")).toBeInTheDocument();
            expect(
                screen.getByText(
                    "The following jobs will be created when you start predicting."
                )
            ).toBeInTheDocument();
            expect(screen.getByText("Prediction Job")).toBeInTheDocument();
            expect(
                screen.getByText("Prediction Table Generation")
            ).toBeInTheDocument();
            expect(
                screen.getByText(
                    "Add key-value tags to organize and filter this prediction job. Tags will be applied to both jobs."
                )
            ).toBeInTheDocument();

            await addTagToJob("env", "prod");
        }

        await clickSecondStartButton();

        await waitFor(() => {
            const url = `v1/batch_prediction_jobs/`;
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(url, args, {
                headers: {
                    "Content-Type": "application/json"
                }
            });
        });

        if (createWithFailure || createWithInvalid) {
            await closeErrorModal();
            return;
        }

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                `/project/project-tests/predictions/prediction/${bpJobId}?job_type=BATCH_PREDICTION_JOB`
            );
        });
    };

    const selectOutputDestination = async (destination: string) => {
        const dropdown = screen.getByTestId("testid-output_destination");
        await userEvent.click(dropdown);
        const dropdownList = await screen.findByRole("listbox");
        expect(dropdownList).toBeInTheDocument();

        // The list is virtualized — await the options so the assertion does not
        // race the virtualizer painting its rows.
        for (const option of outputDestinationOptions) {
            expect(
                await within(dropdownList).findByText(option.label)
            ).toBeInTheDocument();
        }

        const option = within(dropdownList).getByText(destination);
        await userEvent.click(option);
    };

    const selectConnector = async (connectorId: string) => {
        const dropdown = await screen.findByTestId("testid-connectors");
        await userEvent.click(dropdown);
        const listbox = await screen.findByRole("listbox");
        await userEvent.click(await within(listbox).findByText(connectorId));
    };

    const verifyConnectorSetup = async (connectorType: string) => {
        expect(mockAxiosInstance.get).toHaveBeenCalledWith(
            "v0/projects/project-tests/connectors/"
        );
        const connectorsDropdown = screen.getByTestId("testid-connectors");
        expect(connectorsDropdown).toHaveTextContent("Select Connector");
        const connectorId = {
            Snowflake: "hnm",
            Databricks: "test_databricks",
            BigQuery: "test_bigquery"
        }[connectorType];
        expect(connectorId).toBeDefined();
        await userEvent.click(connectorsDropdown);
        const listbox = await screen.findByRole("listbox");
        await userEvent.click(
            await within(listbox).findByText(connectorId as string)
        );
    };

    const verifyPredictionsCheckbox = () => {
        const predictionsCheckbox = screen.getByRole("checkbox", {
            name: /PREDICTIONS/i
        });
        expect(predictionsCheckbox).toBeEnabled();
        expect(predictionsCheckbox).toBeChecked();
    };

    const verifyTableName = (jobID: string, suffix: string) => {
        const tableNameInput = screen.getByPlaceholderText("Table Name");
        const upperCaseJobID = jobID.toUpperCase();
        expect(tableNameInput).toHaveValue(`${upperCaseJobID}_${suffix}`);
    };

    beforeEach(() => {
        vi.clearAllMocks();
        setJobsTestResponseType(testResponseType.DEFAULT);
        setBatchPredictionJobsResponseType(testResponseType.DEFAULT);
        setTestInsertFilterOverridesResponseType(testResponseType.DEFAULT);
        setDeploymentTypeInStore(deploymentType.SAAS);
        mockNavigateCall();
        queryClient.clear();
        focusManager.setFocused(false);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
    });

    it("Should render job IDs and show hover card on the Job List page", async () => {
        const initialJobId = "9f8b2";

        await renderNewPredictionJobPage();
        await pageLoadTest({ page: 0 });

        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
            `v1/predictive_queries/get_many`,
            new_pred_get_many,
            {
                headers: {
                    "Content-Type": "application/json"
                },
                params: { project_id: "project-tests" }
            }
        );

        const tableCells = await screen.findAllByRole("cell");
        const idCell = tableCells.find((cell) =>
            cell.textContent?.includes(initialJobId)
        );
        expect(idCell).toBeInTheDocument();

        const idSpan = within(idCell!).getByText(initialJobId);
        await userEvent.hover(idSpan);

        await waitFor(() => {
            expect(screen.getByText("Query")).toBeInTheDocument();
        });
    });

    it.each([
        { scenario: "from job list page", selectJob: false },
        { scenario: "from prediction creation section", selectJob: true }
    ])(
        "Should navigate back to Predictions list page when Cancel button is clicked $scenario",
        async ({ selectJob }) => {
            await renderNewPredictionJobPage();
            await pageLoadTest({ page: 0 });

            if (selectJob) {
                await selectTrainingJob();
            }

            const cancelButton = await screen.findByRole("button", {
                name: "Cancel"
            });
            expect(cancelButton).toBeInTheDocument();
            userEvent.click(cancelButton);

            if (selectJob) {
                // When a training job is selected, cancel shows confirmation dialog
                const discardButton = await screen.findByRole("button", {
                    name: "Discard"
                });
                userEvent.click(discardButton);
            }

            await waitFor(() => {
                expect(mockUseNavigate).toHaveBeenCalledWith(
                    "/project/project-tests/predictions/?page=1&job_type=BATCH_PREDICTION_JOB",
                    { replace: true }
                );
            });
        }
    );

    it("Should display welcome empty state when there are no trained model jobs", async () => {
        setJobsTestResponseType(testResponseType.EMPTY);
        setConnectorListResponseType(testResponseType.ALL);
        setGraphsListResponseType(testResponseType.ALL);
        await renderNewPredictionJobPage();
        await pageLoadTest({
            page: 0,
            checkTable: false,
            checkEmptyTable: true,
            checkSkeleton: false
        });
    });

    it("Should navigate to the empty state for an 'Internal Server Error' response", async () => {
        setJobsTestResponseType(testResponseType.ERROR);
        await renderNewPredictionJobPage();
        await pageLoadTest({
            page: 0,
            checkTable: false,
            checkEmptyTable: false,
            checkSkeleton: false
        });
        const header = await screen.findByText("New Prediction");
        expect(header).toBeInTheDocument();

        await waitFor(() => {
            expect(document.querySelector('[data-slot="card"]')).not.toBeNull();
        });
    });

    it.each([
        {
            scenario: "invalid",
            searchTerm: "*r493475%",
            postSearchKeyword: "",
            assertResult: async () => {
                expect(
                    await screen.findByText("No jobs found")
                ).toBeInTheDocument();
            }
        },
        {
            scenario: "valid",
            searchTerm: "31e1b",
            postSearchKeyword: "31e1b",
            assertResult: async () => {
                const firstResultRow = await screen.findAllByRole("row");
                const idCell = within(firstResultRow[1]).getAllByRole(
                    "cell"
                )[0];
                expect(idCell.textContent).toContain("31e1b");
            }
        }
    ])(
        "Should handle $scenario search term correctly",
        async ({ searchTerm, postSearchKeyword, assertResult }) => {
            await renderNewPredictionJobPage();
            await pageLoadTest({ page: 0 });

            const searchInput = await screen.findByPlaceholderText("Search");
            await userEvent.type(searchInput, searchTerm);

            await pageLoadTest({
                page: 0,
                search_keyword: postSearchKeyword,
                checkSkeleton: false
            });

            await waitFor(assertResult, { timeout: 2000 });
        }
    );

    it("Should navigate back to New Prediction page with training job list when 'Back' is clicked", async () => {
        await renderNewPredictionJobPage();
        await pageLoadTest({ page: 0 });

        await selectTrainingJob();

        expect(screen.getByText("Prediction Settings")).toBeInTheDocument();
        expect(screen.getByText("Output table")).toBeInTheDocument();
        expect(screen.getByText("Graph")).toBeInTheDocument();

        const backButton = await screen.findByRole("button", { name: "Back" });
        await userEvent.click(backButton);

        await waitFor(() => {
            const jobsListTableRows = screen.getAllByRole("row");
            expect(jobsListTableRows[0]).toHaveTextContent(
                expectedNewPredictionJobTableHeader
            );
        });
    });

    it("Should display display the Training Job side panel when the Training Job ID is clicked", async () => {
        await renderNewPredictionJobPage();
        await pageLoadTest({ page: 0 });

        await selectTrainingJob();

        const id = "trainingjob-9f8b259b4f764c8f932ad5912a926ec2";
        const expectedWorkflow = { ...mockWorkflowResponse };
        expectedWorkflow.training_job.id = id;
        const displayId = id.replace("trainingjob-", "").slice(0, 5);

        const viewTrainingJobButton = await screen.findByText(
            String(displayId)
        );
        await userEvent.click(viewTrainingJobButton);

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/training_jobs/${id}/warnings`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/training_jobs/${id}/workflow`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        const headerText = "Training";
        await waitFor(async () => {
            await screen.findAllByText(headerText);
        });

        expect(screen.getByText("Model Performance")).toBeInTheDocument();
        expect(screen.getByText("Graph Snapshot")).toBeInTheDocument();
        expect(
            screen.getByText("Training Table Generation")
        ).toBeInTheDocument();
    });

    it("Should display graph structure and output schema on clicking corresponding fields", async () => {
        await renderNewPredictionJobPage();
        await pageLoadTest({ page: 0 });

        await selectTrainingJob();

        // Validation status should not show errors initially
        const trainingJobId = "trainingjob-9f8b259b4f764c8f932ad5912a926ec2";
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/predictive_queries/${trainingJobId}/get_allowed_bp_query_overrides`
            );
        });
        await waitFor(
            () => {
                expect(
                    screen.queryByTestId("validation-status")
                ).not.toBeInTheDocument();
            },
            { timeout: 2000 }
        );

        //Graph Schema
        expect(screen.getByText("Schema")).toBeInTheDocument();
        const tabList = await screen.findByRole("tablist");

        const resolvedGraphSnapshotId = expectedGraphSnapshotResponse();
        const graphSnapshotId =
            resolvedGraphSnapshotId?.snapshot_resource?.id ?? "";
        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/snapshots/graph`,
                {
                    "graph_snapshot_id": `${graphSnapshotId}`,
                    "graph_definiton": undefined,
                    "graph_version": undefined
                },
                {
                    "headers": {
                        "Content-Type": "application/json"
                    },
                    "params": { "project_id": "project-tests" }
                }
            );
        });

        const graphDetails = expectedGraphSnapshotResponse();
        const tables = getKeys(graphDetails?.definition?.tables);
        for (let i = 0; i < tables.length; i++) {
            await waitFor(() => {
                const node = screen.getByTestId("rf__node-" + tables[i]);
                expect(node).toBeInTheDocument();
                expect(node).toHaveTextContent(tables[i]);
            });
        }

        //Output table schema
        const outputSchemaButton = within(tabList).getByRole("tab", {
            name: /output table/i
        });
        expect(outputSchemaButton).toBeInTheDocument();
        await userEvent.click(outputSchemaButton);

        const args = {
            "query_type": "temporal",
            "task_type": "binary_classification"
        };
        await waitFor(() => {
            const url = `v1/batch_prediction_jobs/output_schema`;
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(url, args, {
                headers: {
                    "Content-Type": "application/json"
                }
            });
        });

        const outputSchemaTableHeader = screen.getByTestId("table-header");
        expect(outputSchemaTableHeader).toBeInTheDocument();
        expect(outputSchemaTableHeader).toHaveTextContent("Column name");
        expect(outputSchemaTableHeader).toHaveTextContent("Data type");

        const allOutputSchemaTableRows = screen.getAllByRole("row");
        expect(allOutputSchemaTableRows.length).toBeGreaterThan(3);
    });

    it.each([
        {
            field: "Threshold for Binary Classification",
            invalidValue: "5",
            defaultValue: "0.5",
            errorMessage:
                "Please enter Threshold for binary classification which must be between 0 and 1."
        },
        {
            field: "Number of parallel workers",
            invalidValue: "0",
            defaultValue: "1",
            errorMessage:
                "Please enter Number of parallel workers which must be greater than 0"
        }
    ])(
        "Should display error message for invalid entry of $field",
        async ({ field, invalidValue, defaultValue, errorMessage }) => {
            await renderNewPredictionJobPage();
            await pageLoadTest({ page: 0 });

            await selectTrainingJob();

            const input = await screen.findByPlaceholderText(field);
            await waitFor(() => {
                expect(input.getAttribute("value")).toBe(defaultValue);
            });
            await userEvent.click(input);
            await userEvent.clear(input);
            await userEvent.type(input, invalidValue);
            await waitFor(() => {
                expect(input.getAttribute("value")).toBe(invalidValue);
            });

            const startPredictingButton = await screen.findByRole("button", {
                name: "Start predicting"
            });
            expect(startPredictingButton).not.toBeEnabled();

            await waitFor(
                async () => {
                    expect(
                        await screen.findByText(errorMessage)
                    ).toBeInTheDocument();
                },
                { timeout: 4000 }
            );
        }
    );

    it("Should update the Query Box when the Prediction Entity Filter is entered", async () => {
        await renderNewPredictionJobPage();
        await pageLoadTest({ page: 0 });

        await selectTrainingJob();

        //Enable explanation
        const enableExplanationButton = await screen.findByRole("switch");
        expect(enableExplanationButton).toBeInTheDocument();
        expect(enableExplanationButton).not.toBeChecked();

        //Entity Filter initial call
        const trainingJobId = "trainingjob-9f8b259b4f764c8f932ad5912a926ec2";
        const args1 = {
            "entity_filter": "",
            "target_filter": ""
        };
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/predictive_queries/${trainingJobId}/get_allowed_bp_query_overrides`
            );
        });
        await waitFor(
            () => {
                const url = `v1/predictive_queries/${trainingJobId}/insert_filter_overrides`;
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    url,
                    args1,
                    {
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }
                );
            },
            { timeout: 3000 }
        );

        //Predictive Query
        const queryCard = await screen.findAllByRole("textbox");

        const cmLines = queryCard[0].querySelectorAll(".cm-line");
        const queryText = Array.from(cmLines)
            .map((line) => line.textContent || "")
            .join("\n");

        const intialQueryText =
            "PREDICT COUNT(FINANCE_TRANS.*, 0, 30, days) = 0 \nFOR EACH FINANCE_ACCOUNT.ACCOUNT_ID";
        expect(intialQueryText).toBe(queryText);

        //Prediction Entity Filter
        const entityFilterTextarea = screen.getByTestId("prediction-filter");
        const entityFilter = "FINANCE_ACCOUNT.FREQUENCY = 'POPLATEK MESICNE'";
        await userEvent.click(entityFilterTextarea);
        await userEvent.type(entityFilterTextarea, entityFilter);

        const args2 = {
            "entity_filter": entityFilter,
            "target_filter": ""
        };
        await waitFor(
            () => {
                const url = `v1/predictive_queries/${trainingJobId}/insert_filter_overrides`;
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    url,
                    args2,
                    {
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }
                );
            },
            { timeout: 3000 }
        );

        const queryWithEntityFilter =
            "PREDICT COUNT(FINANCE_TRANS.*, 0, 30, days) = 0 \nFOR EACH FINANCE_ACCOUNT.ACCOUNT_ID \nWHERE FINANCE_ACCOUNT.FREQUENCY = 'POPLATEK MESICNE'";

        await waitFor(() => {
            const finalQueryText = Array.from(
                screen.getAllByRole("textbox")[0].querySelectorAll(".cm-line")
            )
                .map((line) => line.textContent || "")
                .join("\n");
            expect(finalQueryText).toBe(queryWithEntityFilter);
        });
    });

    it("Should display 'Issues found' when Invalid Prediction Entity Filter is entered", async () => {
        setTestInsertFilterOverridesResponseType(testResponseType.INVALID);
        await renderNewPredictionJobPage();
        await pageLoadTest({ page: 0 });

        const tableHeader = await screen.findByTestId("table-header");
        expect(tableHeader).toHaveTextContent("Job ID");

        const allRows = screen.getAllByRole("row");
        const firstRow = allRows[1];
        const cells = within(firstRow).getAllByRole("cell");
        const idCell = cells[0];

        await userEvent.click(idCell);

        await waitFor(
            async () => {
                const validationStatus =
                    await screen.findByTestId("validation-status");
                expect(validationStatus).toHaveTextContent("Issues found");
                await userEvent.hover(validationStatus);
            },
            { timeout: 4000 }
        );

        const validationCard = await screen.findByTestId("validation-card");
        expect(validationCard).toBeInTheDocument();
        expect(validationCard).toHaveTextContent(
            "Prediction Filter is invalid."
        );

        expect(validationCard).toHaveTextContent(
            "Encountered the following issues during parsing: Errors:"
        );

        expect(validationCard).toHaveTextContent(
            "1. Invalid Syntax. The entity filter (FOR EACH ... WHERE) clause in this query is empty or invalid. This clause allows you to filter the entities that the model will generate training examples for."
        );

        expect(validationCard).toHaveTextContent(
            "2. Invalid Syntax. Line 1, col 90; Input Mismatch, got 'diomdm', but expected one of '(', FQN_COLUMN, WILDCARD_COLUMN, NOT, AGGR."
        );
    });

    it("Should configure S3 output with predictions and embeddings files and handle API error", async () => {
        setConnectorListResponseType(testResponseType.ALL);
        useUserStore.getState().setUserDetails({
            isLoggedIn: true,
            first_name: "test",
            last_name: "user"
        });

        const jobID = "9f8b2";
        await renderNewPredictionJobPage();
        await pageLoadTest({ page: 0 });

        await selectTrainingJob();
        await selectConnector("ltv_test");

        //Output Destination - S3 is default
        verifyPredictionsCheckbox();

        //PREDICTIONS -File Name
        const predictionFileNameInput =
            await screen.findByPlaceholderText("File Name");
        expect(predictionFileNameInput).toHaveValue(`${jobID}_predictions`);
        await userEvent.click(predictionFileNameInput);
        await userEvent.clear(predictionFileNameInput);
        const newPredictionFileName = "new_predictions_file";
        fireEvent.change(predictionFileNameInput, {
            target: { value: newPredictionFileName }
        });
        expect(predictionFileNameInput).toHaveValue(newPredictionFileName);

        //PREDICTIONS -Output Directory
        const predictionOutputDirInput = screen.getByPlaceholderText(
            "s3://<bucket>/<key prefix>"
        );
        const dir =
            "s3://kumo-public-datasets/customerltv_integ_test/momentic_predictions/predictions/pred/";
        fireEvent.change(predictionOutputDirInput, { target: { value: dir } });
        expect(predictionOutputDirInput).toHaveValue(dir);

        //PREDICTIONS -File Type
        const predictionFileTypeDropdown =
            screen.getByTestId("testid-ext-select");
        await userEvent.click(predictionFileTypeDropdown);
        const listbox = await screen.findByRole("listbox");
        const predictionFileTypeOption =
            await within(listbox).findByText(".csv");
        await userEvent.click(predictionFileTypeOption);
        await waitFor(() => {
            expect(screen.getByText(".csv")).toBeInTheDocument();
        });
        expect(
            screen.getByRole("button", { name: "Start predicting" })
        ).toBeDisabled();

        //EMBEDDINGS checkbox
        await enableEmbeddings();

        //EMBEDDINGS -File Name
        const fileNameInput = screen.getAllByPlaceholderText("File Name");
        const embeddingsFileNameInput = fileNameInput[1];
        expect(embeddingsFileNameInput).toHaveValue(`${jobID}_embeddings`);
        await userEvent.clear(embeddingsFileNameInput);
        const newFileName = "new_embeddings_file";
        fireEvent.change(embeddingsFileNameInput, {
            target: { value: newFileName }
        });
        expect(embeddingsFileNameInput).toHaveValue(newFileName);

        //EMBEDDINGS -Output Directory
        const outputDirectoryInput = screen.getAllByPlaceholderText(
            "s3://<bucket>/<key prefix>"
        );
        const embeddingsOutputDirInput = outputDirectoryInput[1];
        fireEvent.change(embeddingsOutputDirInput, { target: { value: dir } });
        expect(embeddingsOutputDirInput).toHaveValue(dir);

        await waitForValidationStatus("Looks good!");

        await waitFor(
            () =>
                expect(
                    screen.getByRole("button", {
                        name: "Start predicting"
                    })
                ).toBeEnabled(),
            { timeout: 4000 }
        );

        const newBpJob = expectedNewBpJobResponse();
        const baseArgs = {
            "project_id": "project-tests",
            "job_tags": newBpJob?.job_tags,
            "model_training_job_id": newBpJob?.model_training_job_id,
            "predict_options": {
                "binary_classification_threshold": 0.5,
                "num_workers": 1
            },
            "outputs": (newBpJob?.outputs ?? []).map((output) => ({
                ...output,
                connector_id: "ltv_test"
            })),
            "graph_snapshot_id": newBpJob?.graph_snapshot_id,
            "explanations": newBpJob?.explanations,
            "predictive_query":
                "PREDICT COUNT(FINANCE_TRANS.*, 0, 30, days) = 0 FOR EACH FINANCE_ACCOUNT.ACCOUNT_ID",
            "anchor_time": newBpJob?.anchor_time,
            "filter_overrides": {
                "entity_filter": "",
                "target_filter": ""
            },
            "graph_id": newBpJob?.graph_id,
            "user": newBpJob?.user ?? []
        };

        await createNewBpJobApiCallTest(baseArgs as CreatePredictionJobArgs, {
            createWithFailure: true
        });
    });

    it("Should handle batch prediction job creation with invalid response after setting filters and thresholds", async () => {
        useUserStore.getState().setUserDetails({
            isLoggedIn: true,
            first_name: "test",
            last_name: "user"
        });

        await renderNewPredictionJobPage();
        await pageLoadTest({ page: 0 });

        await selectTrainingJob();

        const trainingJobId = "trainingjob-9f8b259b4f764c8f932ad5912a926ec2";

        //Prediction Entity Filter
        const entityFilterTextarea =
            await screen.findByTestId("prediction-filter");
        const entityFilter = "FINANCE_ACCOUNT.FREQUENCY = 'POPLATEK MESICNE'";
        await userEvent.click(entityFilterTextarea);
        fireEvent.change(entityFilterTextarea, {
            target: { value: entityFilter }
        });
        const args2 = {
            "entity_filter": entityFilter,
            "target_filter": ""
        };
        await waitFor(
            () => {
                const url = `v1/predictive_queries/${trainingJobId}/insert_filter_overrides`;
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    url,
                    args2,
                    {
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }
                );
            },
            { timeout: 3000 }
        );

        // Threshold for Binary Classification
        const threshold = screen.getByPlaceholderText(
            "Threshold for Binary Classification"
        );
        await userEvent.click(threshold);
        await userEvent.clear(threshold);
        const thresholdInput = "0.6";
        fireEvent.change(threshold, { target: { value: thresholdInput } });
        await waitFor(() => {
            expect(threshold.getAttribute("value")).toBe(thresholdInput);
        });

        //Number of parallel workers
        const parallelWorkers = screen.getByPlaceholderText(
            "Number of parallel workers"
        );
        await userEvent.click(parallelWorkers);
        await userEvent.clear(parallelWorkers);
        const parallelWorkersInput = "2";
        fireEvent.change(parallelWorkers, {
            target: { value: parallelWorkersInput }
        });
        await waitFor(() => {
            expect(parallelWorkers.getAttribute("value")).toBe(
                parallelWorkersInput
            );
        });

        //Output Destination
        expect(await screen.findByText("S3")).toBeInTheDocument();
        await selectOutputDestination("Local Download Only");

        await waitForValidationStatus("Looks good!");

        const startPredictingButton = screen.getByRole("button", {
            name: "Start predicting"
        });
        expect(startPredictingButton).toBeEnabled();

        const newBpJob = expectedNewBpJobResponse();
        const baseArgs = {
            "project_id": "project-tests",
            "job_tags": newBpJob?.job_tags,
            "model_training_job_id": newBpJob?.model_training_job_id,
            "predict_options": newBpJob?.predict_options,
            "graph_snapshot_id": newBpJob?.graph_snapshot_id,
            "explanations": newBpJob?.explanations,
            "predictive_query": newBpJob?.predictive_query,
            "anchor_time": newBpJob?.anchor_time,
            "filter_overrides": newBpJob?.filter_overrides,
            "graph_id": newBpJob?.graph_id,
            "user": newBpJob?.user ?? []
        };

        await createNewBpJobApiCallTest(baseArgs as CreatePredictionJobArgs, {
            createWithInvalid: true
        });
    });

    it("Should successfully create batch prediction job with Local Download Only output and tags", async () => {
        useUserStore.getState().setUserDetails({
            isLoggedIn: true,
            first_name: "test",
            last_name: "user"
        });

        await renderNewPredictionJobPage();
        await pageLoadTest({ page: 0 });

        await selectTrainingJob();
        await selectOutputDestination("Local Download Only");

        // Local Download Only should not show connector/table fields
        await waitFor(() => {
            expect(
                screen.getByText(
                    "With this option, we won't write the predictions back to your data plane. You will be able to download a sample of the outputs (up to 1GB). Please note that you can download a sample even if you choose to write predictions to another data source."
                )
            ).toBeInTheDocument();
        });
        expect(
            screen.queryByRole("checkbox", { name: /PREDICTIONS/i })
        ).not.toBeInTheDocument();
        expect(
            screen.queryByPlaceholderText("Table Name")
        ).not.toBeInTheDocument();

        await waitForValidationStatus("Looks good!");

        const startPredictingButton = screen.getByRole("button", {
            name: "Start predicting"
        });
        expect(startPredictingButton).toBeEnabled();

        const newBpJob = expectedNewBpJobResponse();
        const baseArgs = {
            "project_id": "project-tests",
            "job_tags": newBpJob?.job_tags,
            "model_training_job_id":
                "trainingjob-9f8b259b4f764c8f932ad5912a926ec2",
            "predict_options": {
                "binary_classification_threshold": 0.5,
                "num_workers": 1
            },
            "graph_snapshot_id": newBpJob?.graph_snapshot_id,
            "explanations": newBpJob?.explanations,
            "predictive_query":
                "PREDICT COUNT(FINANCE_TRANS.*, 0, 30, days) = 0 FOR EACH FINANCE_ACCOUNT.ACCOUNT_ID",
            "anchor_time": newBpJob?.anchor_time,
            "filter_overrides": {
                "entity_filter": "",
                "target_filter": ""
            },
            "graph_id": newBpJob?.graph_id,
            "user": newBpJob?.user ?? []
        };

        await createNewBpJobApiCallTest(
            {
                ...baseArgs,
                job_tags: { "env": "prod" }
            } as CreatePredictionJobArgs,
            {}
        );
    });

    it.each([
        { destination: "Snowflake", hasTableSchema: false },
        { destination: "Databricks", hasTableSchema: true },
        { destination: "BigQuery", hasTableSchema: false }
    ])(
        "Should display connector dropdown and pre-filled table names when output destination is set to $destination",
        async ({ destination, hasTableSchema }) => {
            const jobID = "9f8b2";
            await renderNewPredictionJobPage();
            await pageLoadTest({ page: 0 });

            await selectTrainingJob();
            await selectOutputDestination(destination);
            await verifyConnectorSetup(destination);

            verifyPredictionsCheckbox();

            if (hasTableSchema) {
                expect(
                    screen.getByPlaceholderText("Table Schema")
                ).toBeInTheDocument();
            }

            verifyTableName(jobID, "PREDICTIONS");

            await enableEmbeddings();

            if (hasTableSchema) {
                const tableSchema =
                    screen.getAllByPlaceholderText("Table Schema");
                const embeddingsTableSchema = tableSchema[1];
                expect(embeddingsTableSchema).toBeInTheDocument();
            }

            const fileNameInput = screen.getAllByPlaceholderText("Table Name");
            const embeddingsTableNameInput = fileNameInput[1];
            const upperCaseJobID = jobID.toUpperCase();
            expect(embeddingsTableNameInput).toHaveValue(
                `${upperCaseJobID}_EMBEDDINGS`
            );
        }
    );

    it("Should validate tag input and support add/remove interactions in StartJobPopover", async () => {
        await renderNewPredictionJobPage();
        await pageLoadTest({ page: 0 });
        await selectTrainingJob();
        await selectOutputDestination("Local Download Only");
        await waitForValidationStatus("Looks good!", 4000);

        const startButton = await screen.findByRole("button", {
            name: "Start predicting"
        });
        expect(startButton).toBeEnabled();
        await userEvent.click(startButton);

        // Invalid key characters
        await userEvent.click(await screen.findByText("Add tag"));
        await userEvent.type(
            await screen.findByPlaceholderText("key"),
            "invalid@key"
        );
        expect(
            screen.getByText(
                "Only alphanumeric characters, underscores (_), or hyphens (-) are allowed in key"
            )
        ).toBeInTheDocument();

        // Invalid value characters
        await userEvent.clear(screen.getByPlaceholderText("key"));
        await userEvent.type(screen.getByPlaceholderText("key"), "env");
        await userEvent.type(
            screen.getByPlaceholderText("value"),
            "invalid@value"
        );
        expect(
            screen.getByText(
                "Only alphanumeric characters, underscores (_), or hyphens (-) are allowed in value"
            )
        ).toBeInTheDocument();

        // Missing key
        await userEvent.clear(screen.getByPlaceholderText("key"));
        await userEvent.clear(screen.getByPlaceholderText("value"));
        await userEvent.type(screen.getByPlaceholderText("value"), "prod");
        await userEvent.keyboard("{Enter}");
        expect(screen.getByText("Key is required")).toBeInTheDocument();

        // Missing value
        await userEvent.clear(screen.getByPlaceholderText("key"));
        await userEvent.clear(screen.getByPlaceholderText("value"));
        await userEvent.type(screen.getByPlaceholderText("key"), "env");
        await userEvent.keyboard("{Enter}");
        expect(screen.getByText("Value is required")).toBeInTheDocument();

        // Add a valid tag and verify it appears
        await userEvent.clear(screen.getByPlaceholderText("key"));
        await userEvent.type(screen.getByPlaceholderText("key"), "env");
        await userEvent.type(screen.getByPlaceholderText("value"), "prod");
        await userEvent.keyboard("{Enter}");
        expect(screen.getByText("env=")).toBeInTheDocument();
        expect(screen.getByText("prod")).toBeInTheDocument();
        expect(screen.queryAllByTestId("tag-remove-icon")).toHaveLength(1);

        // Duplicate key error
        await userEvent.click(screen.getByText("Add tag"));
        await userEvent.type(await screen.findByPlaceholderText("key"), "env");
        await userEvent.type(screen.getByPlaceholderText("value"), "staging");
        await userEvent.keyboard("{Enter}");
        expect(
            screen.getByText('Tag key "env" already exists')
        ).toBeInTheDocument();

        // Remove the tag
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

    it("Should render warning message in yellow when prediction filter has warnings", async () => {
        setTestInsertFilterOverridesResponseType(
            testResponseType.WITH_WARNINGS
        );
        await renderNewPredictionJobPage();
        await pageLoadTest({ page: 0 });

        await selectTrainingJob();

        const validationStatus = await screen.findByTestId("validation-status");
        expect(validationStatus).toHaveTextContent("Looks good!");
        expect(validationStatus).toHaveAttribute("data-state", "warning");

        await userEvent.hover(validationStatus);

        const validationCard = await screen.findByTestId("validation-card");
        expect(validationCard).toBeInTheDocument();
        expect(validationCard).toHaveTextContent(
            "Prediction Filter has warnings."
        );
    });
});
