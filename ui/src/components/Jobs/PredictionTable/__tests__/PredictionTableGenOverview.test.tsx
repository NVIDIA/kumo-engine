// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { focusManager, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { setDeploymentTypeInStore } from "stores/__tests__/debugInfoStore";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { deploymentType } from "tests/testType";
import { createQueryClient } from "tests/testUtil";

import { setAddTagTestResponseType } from "components/Jobs/Prediction/__tests__/MockFunctions/AddTag.mock";
import { setDeleteTagTestResponseType } from "components/Jobs/Prediction/__tests__/MockFunctions/DeleteTag.mock";
import { expectedMetadataResponse } from "components/Jobs/TrainingTable/__tests__/MockFunctions/PredictiveQueryMetaDeta.mock";

import { PredictionTableGeneration } from "types/batchPredictions";
import { TaskType } from "types/predictiveQueries";

import { jobStatus } from "utils/Constants";
import { parsePredictionTableGenResponse } from "utils/Converters/jobsConverter";

import PredictionTableGenOverview, {
    PredictionTableGenOverviewProps
} from "../PredictionTableGenOverview";

import {
    expectedGenPredictionTableJobResponse,
    setGenPredictionTableJobResponseStatus
} from "./MockFunctions/PredictionTableGeneration.mock";
import { doneResponse } from "./MockResponses/PredictionTableGeneration";

const id = "gen-predtable-job-00060000000000000000000000000000";
const expectedPredictionTable = doneResponse;
expectedPredictionTable.resource.job_id = id;
const predictionTableDetails = parsePredictionTableGenResponse(
    expectedPredictionTable
);

const defaultProps: PredictionTableGenOverviewProps = {
    title: id,
    buttons: [],
    predictionTableDetails: predictionTableDetails,
    loadingStatus: false,
    onClose: () => {},
    onClickArchive: () => {},
    onClickRecover: () => {},
    queryYaml: "",
    taskType: expectedMetadataResponse.task_type as TaskType,
    anchorTime: "",
    hidePreTitle: false,
    showRelatedJobs: false,
    graphId: ""
};

describe("PredictionTableGenOverview", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });

    const queryClient = createQueryClient();
    const renderPredictionTableGeneration = async (
        props: PredictionTableGenOverviewProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={[
                        "/project/project-test/predictions/predictiontable/"
                    ]}>
                    <QueryClientProvider client={queryClient}>
                        <PredictionTableGenOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const pageLoadTest = async () => {
        const headerText = "Prediction Table Generation";
        await waitFor(() => {
            screen.getAllByText(headerText);
        });
    };

    const verifyStatus = async (expectedStatus: string) => {
        const status = await screen.findAllByTestId("header-status-chip-id");
        expect(status[0].textContent).toBe(expectedStatus);
    };

    beforeEach(() => {
        vi.clearAllMocks();
        setDeploymentTypeInStore(deploymentType.SAAS);
        setGenPredictionTableJobResponseStatus(jobStatus.Done);
        // Suppress RUNNING/QUEUED refetchInterval timer accumulation across
        // tests.
        focusManager.setFocused(false);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
    });

    it("Should render PredictionTableGeneration page with `Empty/Invalid` data", async () => {
        await renderPredictionTableGeneration({
            ...defaultProps,
            predictionTableDetails: {} as PredictionTableGeneration,
            loadingStatus: true
        });
        await pageLoadTest();

        expect(
            document.querySelectorAll('[data-slot="skeleton"]').length
        ).toBeGreaterThan(1);
    });

    it("Should render PredictionTableGeneration page without RELATED JOBS section", async () => {
        await renderPredictionTableGeneration(defaultProps);
        await pageLoadTest();

        expect(screen.queryByText("Graph Snapshot")).not.toBeInTheDocument();
    });

    it("Should render PredictionTableGeneration page and show `skeleton` for Query", async () => {
        await renderPredictionTableGeneration({
            ...defaultProps,
            loadingStatus: true
        });

        const skeletons = document.querySelectorAll('[data-slot="skeleton"]');
        expect(skeletons.length).toBeGreaterThan(0);
    });

    it("Should render PredictionTableGeneration page with `Related Jobs`", async () => {
        await renderPredictionTableGeneration({
            ...defaultProps,
            showRelatedJobs: true
        });
        await pageLoadTest();

        await verifyStatus("Done");

        const graphSnap = screen.queryByText("Graph Snapshot");
        expect(graphSnap).toBeInTheDocument();
        const graphId =
            expectedPredictionTable?.resource?.config?.graph_snapshot_id;

        const graphIdElement = screen.queryAllByText(
            graphId?.replace("graph-", "").substring(0, 5) ?? ""
        );
        expect(graphIdElement.length).toBeGreaterThan(0);
    });

    it("Should render PredictionTableGeneration page for `Failed` status", async () => {
        setGenPredictionTableJobResponseStatus(jobStatus.Failed);
        const expectedPredictionTable =
            expectedGenPredictionTableJobResponse(id);
        const predictionTableDetails = parsePredictionTableGenResponse(
            expectedPredictionTable
        );
        await renderPredictionTableGeneration({
            ...defaultProps,
            predictionTableDetails: predictionTableDetails
        });
        await pageLoadTest();

        await verifyStatus("Failed");

        const error = screen.queryByText(
            "This job has failed. Please contact Kumo for more information."
        );
        expect(error).toBeInTheDocument();
    });

    it("Should render PredictionTableGeneration page for `Cancelled` status", async () => {
        setGenPredictionTableJobResponseStatus(jobStatus.Cancelled);
        const expectedPredictionTable =
            expectedGenPredictionTableJobResponse(id);
        const predictionTableDetails = parsePredictionTableGenResponse(
            expectedPredictionTable
        );
        await renderPredictionTableGeneration({
            ...defaultProps,
            predictionTableDetails: predictionTableDetails
        });
        await pageLoadTest();

        await verifyStatus("Canceled");
    });

    it("Should render PredictionTableGeneration page for `Queued` status", async () => {
        setGenPredictionTableJobResponseStatus(jobStatus.Queued);
        const expectedPredictionTable =
            expectedGenPredictionTableJobResponse(id);
        const predictionTableDetails = parsePredictionTableGenResponse(
            expectedPredictionTable
        );
        await renderPredictionTableGeneration({
            ...defaultProps,
            predictionTableDetails: predictionTableDetails
        });
        await pageLoadTest();

        await verifyStatus("Queued");

        const error = screen.queryByText("This job is queued");
        expect(error).toBeInTheDocument();
    });

    it("Should render PredictionTableGeneration page for `Running` status", async () => {
        setGenPredictionTableJobResponseStatus(jobStatus.Running);
        const expectedPredictionTable =
            expectedGenPredictionTableJobResponse(id);
        const predictionTableDetails = parsePredictionTableGenResponse(
            expectedPredictionTable
        );
        await renderPredictionTableGeneration({
            ...defaultProps,
            predictionTableDetails: predictionTableDetails
        });
        await pageLoadTest();

        await verifyStatus("Running");

        const error = screen.queryByText("Generating prediction table...");
        expect(error).toBeInTheDocument();
    });

    it("Should render PredictionTableGeneration page with `buttons` ", async () => {
        setGenPredictionTableJobResponseStatus(jobStatus.Running);
        const expectedResponse = expectedGenPredictionTableJobResponse(id);
        const predictionTableDetails =
            parsePredictionTableGenResponse(expectedResponse);
        await renderPredictionTableGeneration({
            ...defaultProps,
            predictionTableDetails: predictionTableDetails,
            buttons: [<button data-testid="test-button">Test Button</button>]
        });

        const cancelButton = screen.getByRole("button", {
            name: "Test Button"
        });
        expect(cancelButton).toBeInTheDocument();
    });

    describe("Tags", () => {
        it("Should add tag", async () => {
            await renderPredictionTableGeneration(defaultProps);

            const addTag = screen.getByText("Add tag");
            await userEvent.click(addTag);

            const keyInput = screen.getByPlaceholderText("key");
            const key = "tag_key";
            await userEvent.type(keyInput, key);

            const valueInput = screen.getByPlaceholderText("value");
            const value = "tag_value";
            await userEvent.type(valueInput, value);

            await userEvent.keyboard("{Enter}");

            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    "v0/jobs/add_tag",
                    {
                        job_id: "gen-predtable-job-00060000000000000000000000000000",
                        key: key,
                        project_id: "project-tests",
                        value: value
                    },
                    {
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }
                );
            });

            expect(await screen.findByText(value)).toBeInTheDocument();
            expect(screen.getByText(/TAG_KEY=/i)).toBeInTheDocument();
        });

        it("Should delete tag", async () => {
            await renderPredictionTableGeneration(defaultProps);

            const closeButton = screen.getByTestId("tag-remove-icon");
            await userEvent.click(closeButton);

            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    "v0/jobs/delete_tag",
                    {
                        job_id: "gen-predtable-job-00060000000000000000000000000000",
                        key: "test_new_pred_page",
                        project_id: "project-tests",
                        value: null
                    },
                    {
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }
                );
            });

            expect(
                screen.queryByText("test_new_pred_page")
            ).not.toBeInTheDocument();
        });

        it.each([
            {
                operation: "add",
                responseType: testResponseType.ERROR,
                text: "tag_value",
                shouldExist: false
            },
            {
                operation: "delete",
                responseType: testResponseType.ERROR,
                text: "test_new_pred_page",
                shouldExist: true
            }
        ])(
            "Should not $operation tag when API returns error",
            async ({ operation, responseType, text, shouldExist }) => {
                if (operation === "add") {
                    setAddTagTestResponseType(responseType);
                } else {
                    setDeleteTagTestResponseType(responseType);
                }

                await renderPredictionTableGeneration(defaultProps);

                if (operation === "add") {
                    const addTag = screen.getByText("Add tag");
                    await userEvent.click(addTag);

                    const keyInput = screen.getByPlaceholderText("key");
                    await userEvent.type(keyInput, "tag_key");

                    const valueInput = screen.getByPlaceholderText("value");
                    await userEvent.type(valueInput, "tag_value");

                    await userEvent.keyboard("{Enter}");

                    await waitFor(() => {
                        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                            "v0/jobs/add_tag",
                            expect.any(Object),
                            expect.any(Object)
                        );
                    });
                } else {
                    const closeButton = screen.getByTestId("tag-remove-icon");
                    await userEvent.click(closeButton);

                    await waitFor(() => {
                        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                            "v0/jobs/delete_tag",
                            expect.any(Object),
                            expect.any(Object)
                        );
                    });
                }

                await waitFor(() => {
                    const element = screen.queryByText(text);
                    if (shouldExist) {
                        expect(element).toBeInTheDocument();
                    } else {
                        expect(element).not.toBeInTheDocument();
                    }
                });
            }
        );
    });
});
