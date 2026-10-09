// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

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
import { ConnectorResponse, GenericJob } from "gen/v0/types";
import { ErrorDetail } from "gen/v1/types";
import { setDeploymentType } from "stores/__tests__/MockFunctions/DebugInfo.mock";
import { mockAxiosInstance } from "tests/mockAxios";
import { deploymentType } from "tests/testType";
import { createMockResponse, createQueryClient } from "tests/testUtil";

import mockJobsListResponse from "pages/Jobs/__tests__/MockResponses/JobsList.test.response.json";

import {
    PredictiveQueryInfo,
    QueryType,
    TaskType
} from "types/predictiveQueries";

import { ReactQueryKeys } from "utils/reactQueryKeys";

import PredictionSettings, {
    PredictionSettingsProps
} from "../Settings/PredictionSettings";

const jobList = { ...mockJobsListResponse };
const mockTrainingJob = {
    ...jobList.jobs[0],
    id: "trainingjob-9f8b259b4f764c8f932ad5912a926ec2",
    queryYaml:
        "PREDICT COUNT(FINANCE_TRANS.*, 0, 30, days) = 0 FOR EACH FINANCE_ACCOUNT.ACCOUNT_ID",
    graphName: "financial_graph",
    created: "2024-11-25T04:51:50.659325+00:00",
    dateCreated: "Mon Nov 25, 2024, 5:17:37 AM UTC",
    trainingStatus: "DONE",
    statusKey: "done",
    queryType: QueryType.TEMPORAL,
    taskType: TaskType.BINARY_CLASSIFICATION,
    jobSource: "UI",
    graphSnapshotId:
        "graph-d445eeb8b2aa1b79a142406239fa74e8@6f0ae33ac31edb0c321fbe182e39e460",
    trainingJobId: "trainingjob-9f8b259b4f764c8f932ad5912a926ec2"
} as unknown as GenericJob & PredictiveQueryInfo;

const setPredictionJobArgs = vi.fn();
const setError = vi.fn();
const defaultProps: PredictionSettingsProps = {
    selectedTrainingJob: mockTrainingJob,
    setPredictionJobArgs,
    setError,
    setFilterErrors: vi.fn(),
    setValidationState: vi.fn(),
    setFilterValidState: vi.fn(),
    setHighLightQuery: vi.fn(),
    error: [] as ErrorDetail[]
};

describe("PredictionSettings", () => {
    const renderPredictionSettings = async (
        props: PredictionSettingsProps,
        connectorList?: ConnectorResponse[]
    ) => {
        const queryClient = createQueryClient();
        if (connectorList) {
            const connectorListKey = [
                ReactQueryKeys.ConnectorList,
                "project-tests"
            ];
            queryClient.setQueryDefaults(connectorListKey, {
                staleTime: Infinity
            });
            queryClient.setQueryData(
                connectorListKey,
                createMockResponse(connectorList)
            );
        }
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <PredictionSettings {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const closeDropdown = () => {
        document.body.style.pointerEvents = "";
        fireEvent.click(document); // Clicking outside the popover
    };

    const outputDestinationOptions = [
        { value: "S3", label: "S3" },
        { value: "Snowflake", label: "Snowflake" },
        { value: "Databricks", label: "Databricks" },
        { value: "Bigquery", label: "BigQuery" },
        { value: "LocalDownloadOnly", label: "Local Download Only" }
    ];

    const selectOutputDestination = async (destinationLabel: string) => {
        expect(screen.getByText("S3")).toBeInTheDocument();
        const dropdown = screen.getByTestId("testid-output_destination");
        await userEvent.click(dropdown);
        const dropdownList = await screen.findByRole("listbox");
        expect(dropdownList).toBeInTheDocument();
        outputDestinationOptions.forEach((option) => {
            expect(
                within(dropdownList).getByText(option.label)
            ).toBeInTheDocument();
        });

        closeDropdown();
        const optionElement = within(dropdownList).getByText(destinationLabel);
        await userEvent.click(optionElement);
    };

    const verifyPredictionsCheckbox = (expectedTableName: string) => {
        const predictionsCheckbox = screen.getByRole("checkbox", {
            name: /PREDICTIONS/i
        });
        expect(predictionsCheckbox).toBeEnabled();
        expect(predictionsCheckbox).toBeChecked();

        const predictionTableNameInput =
            screen.getByPlaceholderText("Table Name");
        expect(predictionTableNameInput).toHaveValue(expectedTableName);
    };

    const verifyEmbeddingsCheckbox = async (expectedTableName: string) => {
        const embeddingsCheckbox = screen.getByRole("checkbox", {
            name: /EMBEDDINGS/i
        });
        expect(embeddingsCheckbox).not.toBeChecked();
        await userEvent.click(embeddingsCheckbox);
        expect(embeddingsCheckbox).toBeChecked();

        const fileNameInput = screen.getAllByPlaceholderText("Table Name");
        const embeddingsTableNameInput = fileNameInput[1];
        expect(embeddingsTableNameInput).toHaveValue(expectedTableName);
    };

    beforeEach(async () => {
        vi.clearAllMocks();
        setDeploymentType(deploymentType.SAAS);
    });

    it("Should render `PredictionSettings` with Prediction Entity Filter field when all the props are set `null`", async () => {
        await renderPredictionSettings({
            ...defaultProps,
            selectedTrainingJob: null
        });

        expect(screen.getByText("Prediction Settings")).toBeInTheDocument();
        expect(
            screen.getByText("Prediction Entity Filter")
        ).toBeInTheDocument();
    });

    it("Should update `Prediction Entity Filter` when input changes", async () => {
        await renderPredictionSettings(defaultProps);

        const textarea = screen.getByTestId(
            "prediction-filter"
        ) as HTMLTextAreaElement;
        const textareaText = "FINANCE_ACCOUNT.FREQUENCY = 'POPLATEK MESICNE'";
        await userEvent.type(textarea, textareaText);

        await waitFor(() => {
            expect(textarea.value).toBe(textareaText);
        });
    });

    it("Should display link prediction specific fields: Target Filter and Predicted Links per Entity for static query (excluding Anchor Time and Binary threshold)", async () => {
        await renderPredictionSettings({
            ...defaultProps,
            selectedTrainingJob: {
                ...mockTrainingJob,
                taskType: TaskType.LINK_PREDICTION,
                queryType: QueryType.STATIC
            } as GenericJob & PredictiveQueryInfo
        });

        expect(
            screen.getByText("Prediction Target Filter")
        ).toBeInTheDocument();
        expect(screen.getByText("PREDICTIONS")).toBeInTheDocument();
        expect(
            screen.getByText("Predicted Links per Entity")
        ).toBeInTheDocument();

        //Anchor time not present for static query
        expect(
            screen.queryByText("Anchor time for prediction")
        ).not.toBeInTheDocument();

        //Threshold not present for non binary task
        expect(
            screen.queryByText("Threshold for Binary Classification")
        ).not.toBeInTheDocument();
    });

    it("Should display binary classification specific fields: Anchor Time and Binary threshold for temporal query (excluding Target Filter, Predicted Links per Entity, and Max Categories)", async () => {
        const mockError: ErrorDetail[] = [];
        await renderPredictionSettings({
            ...defaultProps,
            error: mockError
        });

        expect(screen.getByText("Prediction Anchor Time")).toBeInTheDocument();
        expect(
            screen.queryByText("Threshold for Binary Classification")
        ).toBeInTheDocument();

        //Target Filter & predicted links per entity not present for non link prediction task
        expect(
            screen.queryByText("Prediction Target Filter")
        ).not.toBeInTheDocument();
        expect(
            screen.queryByText("Predicted Links per Entity")
        ).not.toBeInTheDocument();

        //Max categories not present of non multilabel task
        expect(screen.queryByText("Max Categories")).not.toBeInTheDocument();
    });

    it("Should render core prediction settings fields (parallel workers, explanations toggle, output destination)", async () => {
        await renderPredictionSettings(defaultProps);

        expect(
            screen.getByText("Number of parallel workers")
        ).toBeInTheDocument();
        expect(screen.getByText("Enable Explanations")).toBeInTheDocument();
        expect(screen.getByText("Output Destination")).toBeInTheDocument();

        // Test explanations toggle interaction
        const toggle = screen.getByRole("switch");
        await userEvent.click(toggle);
    });

    it("Should display connector dropdown and pre-filled table names when output destination is set to `Snowflake`", async () => {
        await renderPredictionSettings(defaultProps);
        await selectOutputDestination("Snowflake");

        expect(screen.getByText("Select Connector")).toBeInTheDocument();

        const jobID = "9f8b2";
        const upperCaseJobID = jobID.toUpperCase();

        verifyPredictionsCheckbox(`${upperCaseJobID}_PREDICTIONS`);
        await verifyEmbeddingsCheckbox(`${upperCaseJobID}_EMBEDDINGS`);
    });

    it("Should display connector dropdown,table schema and pre-filled table names when output destination is set to `Databricks`", async () => {
        await renderPredictionSettings(defaultProps);
        await selectOutputDestination("Databricks");

        expect(screen.getByText("Select Connector")).toBeInTheDocument();

        const jobID = "9f8b2";
        const upperCaseJobID = jobID.toUpperCase();

        expect(screen.getByPlaceholderText("Table Schema")).toBeInTheDocument();

        verifyPredictionsCheckbox(`${upperCaseJobID}_PREDICTIONS`);
        await verifyEmbeddingsCheckbox(`${upperCaseJobID}_EMBEDDINGS`);

        const tableSchema = screen.getAllByPlaceholderText("Table Schema");
        expect(tableSchema[1]).toBeInTheDocument();
    });

    it("Should display connector dropdown and pre-filled table names when output destination is set to `BigQuery`", async () => {
        await renderPredictionSettings(defaultProps);
        await selectOutputDestination("BigQuery");

        expect(screen.getByText("Select Connector")).toBeInTheDocument();

        const jobID = "9f8b2";
        const upperCaseJobID = jobID.toUpperCase();

        verifyPredictionsCheckbox(`${upperCaseJobID}_PREDICTIONS`);
        await verifyEmbeddingsCheckbox(`${upperCaseJobID}_EMBEDDINGS`);
    });

    it("Should not display connector or table fields when `Local Download Only` is selected", async () => {
        await renderPredictionSettings(defaultProps);
        await selectOutputDestination("Local Download Only");

        expect(
            screen.getByText(
                "With this option, we won't write the predictions back to your data plane. You will be able to download a sample of the outputs (up to 1GB). Please note that you can download a sample even if you choose to write predictions to another data source."
            )
        ).toBeInTheDocument();

        expect(
            screen.queryByRole("checkbox", { name: /PREDICTIONS/i })
        ).not.toBeInTheDocument();
    });

    it("Should display `Max Categories` for Multilabel Classification task type", async () => {
        await renderPredictionSettings({
            ...defaultProps,
            selectedTrainingJob: {
                ...mockTrainingJob,
                taskType: TaskType.MULTILABEL_CLASSIFICATION
            } as GenericJob & PredictiveQueryInfo
        });

        expect(screen.getByText("Max Categories")).toBeInTheDocument();
    });

    it("Should load connectors and filter overrides on mount", async () => {
        await renderPredictionSettings(defaultProps);

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/projects/project-tests/connectors/`
            );
        });

        const trainingJobId = "trainingjob-9f8b259b4f764c8f932ad5912a926ec2";
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/predictive_queries/${trainingJobId}/get_allowed_bp_query_overrides`
            );
        });
    });

    it("reports source-only S3 connectors as present but not writable", async () => {
        await renderPredictionSettings(defaultProps, [
            {
                id: "source-only-s3",
                project_id: "project-tests",
                identity_id: "identity-s3",
                name: "All S3",
                timestamps: {
                    created_at: "2026-07-11T00:00:00Z",
                    updated_at: "2026-07-11T00:00:00Z"
                },
                config: {
                    type: "s3",
                    target: { mode: "any_uri" }
                }
            }
        ]);

        await waitFor(
            () => {
                expect(setError).toHaveBeenCalledWith(
                    expect.arrayContaining([
                        expect.objectContaining({
                            title: "No writable/rooted S3 connectors have been added to Kumo.",
                            description: expect.stringContaining(
                                "Only source-only any-URI S3 connectors are available."
                            )
                        })
                    ])
                );
            },
            { timeout: 4000 }
        );
    });

    it("Should show validation error for invalid 'Threshold for Binary Classification' input", async () => {
        const mockError: ErrorDetail[] = [];
        await renderPredictionSettings({
            ...defaultProps,
            error: mockError
        });

        const threshold = screen.getByPlaceholderText(
            "Threshold for Binary Classification"
        );

        await waitFor(() => {
            expect(threshold).toHaveValue("0.5");
        });

        //Type value which is out of range
        await userEvent.clear(threshold);
        await userEvent.type(threshold, "1.5");

        await waitFor(
            () => {
                expect(setError).toHaveBeenCalledWith(
                    expect.arrayContaining([
                        expect.objectContaining({
                            title: "Threshold for binary classification is invalid.",
                            description: expect.stringContaining(
                                "must be between 0 and 1"
                            )
                        })
                    ])
                );
            },
            { timeout: 4000 }
        );
    });

    it("Should show validation error for invalid `Number of parallel workers` input", async () => {
        await renderPredictionSettings(defaultProps);

        const parallelWorkers = screen.getByPlaceholderText(
            "Number of parallel workers"
        );
        await waitFor(() => {
            expect(parallelWorkers.getAttribute("value")).toBe("1");
        });

        //Type value which is out of range
        await userEvent.click(parallelWorkers);
        await userEvent.clear(parallelWorkers);
        const parallelWorkersInput = "0";
        await userEvent.type(parallelWorkers, parallelWorkersInput);

        await waitFor(
            () => {
                expect(setError).toHaveBeenCalledWith(
                    expect.arrayContaining([
                        expect.objectContaining({
                            title: "Number of parallel workers is invalid.",
                            description:
                                "Please enter Number of parallel workers which must be greater than 0"
                        }),
                        expect.objectContaining({
                            title: "S3 Output directory is invalid for PREDICTIONS.",
                            description:
                                "Enter a deep relative or matching full S3 path beneath the selected rooted connector."
                        })
                    ])
                );
            },
            { timeout: 4000 }
        );
    });

    it("Should show validation error for invalid `Predicted Links per Entity` input", async () => {
        await renderPredictionSettings({
            ...defaultProps,
            selectedTrainingJob: {
                ...mockTrainingJob,
                taskType: TaskType.LINK_PREDICTION
            } as GenericJob & PredictiveQueryInfo
        });

        const linkPerEntityInput =
            screen.getByPlaceholderText("No predictions");
        const inputValue = "0";
        await userEvent.type(linkPerEntityInput, inputValue);

        await waitFor(
            () => {
                expect(setError).toHaveBeenCalledWith(
                    expect.arrayContaining([
                        expect.objectContaining({
                            title: "Predicted Links per Entity or Max Categories is invalid for PREDICTIONS.",
                            description:
                                "Please enter Predicted Links per Entity or Max Categories which must be a positive number."
                        }),
                        expect.objectContaining({
                            title: "S3 Output directory is invalid for PREDICTIONS.",
                            description:
                                "Enter a deep relative or matching full S3 path beneath the selected rooted connector."
                        })
                    ])
                );
            },
            { timeout: 4000 }
        );
    });

    it("Should valide `Max Categories` for invalid input", async () => {
        await renderPredictionSettings({
            ...defaultProps,
            selectedTrainingJob: {
                ...mockTrainingJob,
                taskType: TaskType.MULTILABEL_CLASSIFICATION
            } as GenericJob & PredictiveQueryInfo
        });

        const input = screen.getByPlaceholderText("All");
        const inputValue = "0";
        await userEvent.type(input, inputValue);

        await waitFor(
            () => {
                expect(setError).toHaveBeenCalledWith(
                    expect.arrayContaining([
                        expect.objectContaining({
                            title: "Predicted Links per Entity or Max Categories is invalid for PREDICTIONS.",
                            description:
                                "Please enter Predicted Links per Entity or Max Categories which must be a positive number."
                        }),
                        expect.objectContaining({
                            title: "S3 Output directory is invalid for PREDICTIONS.",
                            description:
                                "Enter a deep relative or matching full S3 path beneath the selected rooted connector."
                        })
                    ])
                );
            },
            { timeout: 4000 }
        );
    });

    it("Should handle threshold input with more than 2 decimal places by truncating to 4 characters", async () => {
        await renderPredictionSettings(defaultProps);

        const threshold = screen.getByPlaceholderText(
            "Threshold for Binary Classification"
        );

        await waitFor(() => {
            expect(threshold).toHaveValue("0.5");
        });

        await userEvent.clear(threshold);
        await userEvent.type(threshold, "0.123456");

        await waitFor(() => {
            expect(threshold).toHaveValue("0.12");
        });
    });
});
