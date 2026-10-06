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

import { QueryType, TaskType } from "types/predictiveQueries";
import { TrainTableGeneration } from "types/training_job";
import { CTAInfo } from "types/warnings";

import { jobStatus } from "utils/Constants";
import { parseTrainTableGenResponse } from "utils/Converters/jobsConverter";

import mockTrainingResponse from "../../Training/__tests__/MockResponses/TrainingWorkflow.test.json";
import {
    TrainingJobOverviewDetail,
    TrainingTableGenOverviewProps
} from "../TrainingTableGenOverview";

import { expectedMetadataResponse } from "./MockFunctions/PredictiveQueryMetaDeta.mock";
import { setSplitStatsResponseType } from "./MockFunctions/SplitStats.mock";
import {
    expectedTrainingTableGenResponse,
    setTrainingGenResponseStatus,
    setTrainingGenTestResponseType
} from "./MockFunctions/TrainingTableGeneration.mock";

const id = "gen-traintable-job-1db2edcc9f30420898ce79b9161a4c56";
const expectedResponse = expectedTrainingTableGenResponse(id);
const trainTableDetails = parseTrainTableGenResponse(expectedResponse);

const defaultProps: TrainingTableGenOverviewProps = {
    title: id,
    buttons: [],
    trainTableDetails: trainTableDetails,
    ctaMsgs: {} as unknown as CTAInfo,
    loadingStatus: false,
    onClose: () => {},
    onClickArchive: () => {},
    onClickRecover: () => {},
    queryYaml: "",
    taskType: expectedMetadataResponse.task_type as TaskType,
    queryType: expectedMetadataResponse.query_type as QueryType,
    hidePreTitle: false,
    graphId: "",
    showRelatedJobs: false,
    isTrainingPage: true
};

describe("TrainingTableGenOverview", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });
    const queryClient = createQueryClient();

    const renderTrainingTableGenOverview = async (
        props: TrainingTableGenOverviewProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={[
                        "/project/project-test/models/trainingtable/"
                    ]}>
                    <QueryClientProvider client={queryClient}>
                        <TrainingJobOverviewDetail {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const waitForPageLoad = async () => {
        const headerText = "Training Table";
        await screen.findAllByText(headerText);
    };

    const verifyStatus = async (expectedStatus: string) => {
        const status = await screen.findAllByTestId("header-status-chip-id");
        expect(status[0].textContent).toBe(expectedStatus);
    };

    beforeEach(() => {
        vi.clearAllMocks();
        setDeploymentTypeInStore(deploymentType.SAAS);
        queryClient.clear();
        // Suppress RUNNING/QUEUED refetchInterval timer accumulation across
        // tests.
        focusManager.setFocused(false);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
    });

    it("Should render TrainingTableGeneration page with empty/invalid data", async () => {
        await renderTrainingTableGenOverview({
            ...defaultProps,
            trainTableDetails: {} as TrainTableGeneration,
            loadingStatus: true
        });

        const skeleton = document.querySelector('[data-slot="skeleton"]');
        expect(skeleton).toBeInTheDocument();
        expect(skeleton).not.toBeNull();
    });

    it("Should render TrainingTableGeneration page with actual data", async () => {
        await renderTrainingTableGenOverview(defaultProps);

        const status = await screen.findAllByTestId("header-status-chip-id");
        expect(status[0].textContent).toBe("Done");

        const statsTabs = ["Training Data Statistics", "Split stats"];
        statsTabs.forEach((text) => {
            const element = screen.queryAllByText(text);
            expect(element.length).toBeGreaterThanOrEqual(1);
        });

        const relatedJobs = screen.queryByText("RELATED JOBS");
        expect(relatedJobs).not.toBeInTheDocument();
    });

    it("Should render TrainingTableGeneration page and show `skeleton` for Query", async () => {
        await renderTrainingTableGenOverview({
            ...defaultProps,
            loadingStatus: true
        });

        const skeleton = document.querySelector('[data-slot="skeleton"]');
        expect(skeleton).toBeInTheDocument();
        expect(skeleton).not.toBeNull();
    });

    it("Should render TrainingTableGeneration page with `Related Jobs`", async () => {
        await renderTrainingTableGenOverview({
            ...defaultProps,
            showRelatedJobs: true
        });

        await waitForPageLoad();
        await verifyStatus("Done");

        const graphSnap = await screen.findByText("Graph Snapshot");
        expect(graphSnap).toBeInTheDocument();
        const graphId =
            mockTrainingResponse.generation_job.config.graph_snapshot_id;

        const graphIdElement = screen.queryAllByText(
            graphId?.replace("graph-", "").substring(0, 5)
        );
        expect(graphIdElement.length).toBeGreaterThan(0);
    });

    it("Should render TrainingTableGeneration page with `Stats` and `Splits`", async () => {
        setSplitStatsResponseType(testResponseType.DEFAULT);
        await renderTrainingTableGenOverview({
            ...defaultProps,
            showRelatedJobs: true
        });

        expect(mockAxiosInstance.get).toHaveBeenCalledWith(
            `v0/training_jobs/${expectedResponse.resource.tags.parent_job}/get_train_table_split_stats`,
            undefined,
            {
                headers: undefined,
                params: { project_id: "project-tests" }
            }
        );
        expect(mockAxiosInstance.get).toHaveBeenCalledWith(
            `v0/gentraintable_jobs/${id}/generation_stats`,
            undefined,
            {
                headers: undefined,
                params: { project_id: "project-tests" }
            }
        );

        const splits = ["Training", "Validation", "Holdout"];
        for (const split of splits) {
            const elements = await screen.findAllByText(split);
            expect(elements.length).toBeGreaterThanOrEqual(1);
        }

        const statsTabs = [
            "Training Data Statistics",
            "Timeframe chart",
            "Timeframe stats",
            "Split stats"
        ];

        statsTabs.forEach((text) => {
            const element = screen.queryAllByText(text);
            expect(element.length).toBeGreaterThanOrEqual(1);
        });
    });

    it.each([
        {
            status: jobStatus.Failed,
            displayStatus: "Failed",
            message:
                "This job has failed. Please contact Kumo for more information."
        },
        {
            status: jobStatus.Running,
            displayStatus: "Running",
            message: "Generating training table..."
        },
        {
            status: jobStatus.Queued,
            displayStatus: "Queued",
            message: "This job is queued"
        }
    ])(
        "Should render TrainingTableGeneration page for `$displayStatus` status",
        async ({ status, displayStatus, message }) => {
            setTrainingGenResponseStatus(status);
            const expectedResponse = expectedTrainingTableGenResponse(id);
            const trainTableDetails =
                parseTrainTableGenResponse(expectedResponse);
            await renderTrainingTableGenOverview({
                ...defaultProps,
                trainTableDetails: trainTableDetails,
                showRelatedJobs: true
            });

            await waitForPageLoad();
            await verifyStatus(displayStatus);

            const alert = await screen.findByText(message);
            expect(alert).toBeInTheDocument();
        }
    );

    it("Should render TrainingTableGeneration page with `buttons` ", async () => {
        setTrainingGenResponseStatus(jobStatus.Running);
        setTrainingGenTestResponseType(testResponseType.DEFAULT);
        const expectedResponse = expectedTrainingTableGenResponse(id);
        const trainTableDetails = parseTrainTableGenResponse(expectedResponse);
        await renderTrainingTableGenOverview({
            ...defaultProps,
            trainTableDetails: trainTableDetails,
            buttons: [<button data-testid="test-button">Test Button</button>],
            showRelatedJobs: true
        });

        const cancelButton = await screen.findByRole("button", {
            name: "Test Button"
        });
        expect(cancelButton).toBeInTheDocument();
    });

    describe("Tags", () => {
        it("Should add tag", async () => {
            await renderTrainingTableGenOverview(defaultProps);

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
                        job_id: "gen-traintable-job-1db2edcc9f30420898ce79b9161a4c56",
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
            await renderTrainingTableGenOverview(defaultProps);

            const closeButton = screen.getByTestId("tag-remove-icon");
            await userEvent.click(closeButton);

            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    "v0/jobs/delete_tag",
                    {
                        job_id: "gen-traintable-job-1db2edcc9f30420898ce79b9161a4c56",
                        key: "new_train_tag",
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

            expect(screen.queryByText("new_train_tag")).not.toBeInTheDocument();
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
                text: "new_train_tag",
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

                await renderTrainingTableGenOverview(defaultProps);

                if (operation === "add") {
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
