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
import { createQueryClient, normalizeString } from "tests/testUtil";

import {
    minimalTrainingWorkflow,
    setCustomTrainingWorkflowResponse,
    setTrainingWorkflowTestResponseType
} from "components/Jobs/Training/__tests__/MockFunctions/TrainingJobWorkflow.mock";

import { jobStatus } from "utils/Constants";
import { getIdHash } from "utils/Helpers/stringHelpers";

import {
    default as mockTrainingResponse,
    default as mockTrainingWorkflowResponse
} from "../../Training/__tests__/MockResponses/TrainingWorkflow.test.json";
import TableGeneration, {
    TrainingTableGenerationProps
} from "../TrainingTableGeneration";

import { setCancelTrainingTableResponseType } from "./MockFunctions/CancelTrainingTable.mock";
import { setGenStatsResponseType } from "./MockFunctions/GenerationStats.mock";
import { setSplitStatsResponseType } from "./MockFunctions/SplitStats.mock";
import {
    setTrainingGenResponseStatus,
    setTrainingGenTestResponseType
} from "./MockFunctions/TrainingTableGeneration.mock";
import { doneResponse } from "./MockResponses/TrainingTableStatusTypes";

const id = "gen-traintable-job-1db2edcc9f30420898ce79b9161a4c56";
const mockOnArchive = vi.fn();
const mockOnRecover = vi.fn();

const defaultProps: TrainingTableGenerationProps = {
    hidePreTitle: false,
    id: id,
    onClose: vi.fn(),
    onTableStatusUpdate: vi.fn(),
    showRelatedJobs: false,
    modelPlan: null,
    trainingPlan: null,
    isTrainingPage: false,
    queryInfo: undefined,
    onArchive: mockOnArchive,
    onRecover: mockOnRecover
};

describe("TableGeneration", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });
    const queryClient = createQueryClient(false);

    const renderTrainingjobPage = async (
        props: TrainingTableGenerationProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={[
                        "/project/project-test/models/trainingtable/"
                    ]}>
                    <QueryClientProvider client={queryClient}>
                        <TableGeneration {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCalls = (
        id: string,
        trainingJobId: string,
        withWorkflowApi: boolean = true
    ) => {
        const APIS_TO_CALL = [`v0/gentraintable_jobs/${id}`];
        if (withWorkflowApi) {
            APIS_TO_CALL.push(`v0/training_jobs/${trainingJobId}/workflow`);
        }
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
    };

    const waitForPageLoad = async (id: string) => {
        const headerText = "Training Table";
        await screen.findAllByText(headerText);

        const displayId = id.replace("gen-traintable-job-", "").slice(0, 5);
        const jobIds = await screen.findAllByText(displayId);
        expect(jobIds.length).toBeGreaterThan(0);
    };

    const pageLoadTest = async (
        id: string,
        trainingJobId: string,
        withWorkflowApi: boolean = true
    ) => {
        await waitFor(() => checkApiCalls(id, trainingJobId, withWorkflowApi));
        await waitForPageLoad(id);
    };

    const verifyQueryText = async () => {
        const queryCard = await screen.findByRole("textbox");
        const cmLines = queryCard.querySelectorAll(".cm-line");
        const queryText = Array.from(cmLines)
            .map((line) => line.textContent || "")
            .join("\n");

        await waitFor(() => {
            expect(normalizeString(queryText)).toBe(
                normalizeString(
                    mockTrainingWorkflowResponse.pquery_definition.query_string
                )
            );
        });
    };

    const verifyStatus = async (expectedStatus: string) => {
        const status = await screen.findAllByTestId("header-status-chip-id");
        expect(status[0].textContent).toBe(expectedStatus);
    };

    const setupRunningJobAndOpenCancelDialog = async () => {
        const expectedResponse = doneResponse;
        expectedResponse.resource.job_id = id;
        // Cancel-flow test: the related-jobs section still mounts (its workflow
        // GET is asserted by checkApiCalls — unaffected by the response body),
        // but a minimal workflow skips the heavy eval/explanation cascade.
        setCustomTrainingWorkflowResponse(minimalTrainingWorkflow(id, "DONE"));
        setTrainingGenResponseStatus(jobStatus.Running);
        await renderTrainingjobPage({
            ...defaultProps,
            showRelatedJobs: true
        });
        await pageLoadTest(id, expectedResponse.resource.tags.parent_job);

        await verifyStatus("Running");

        const alert = await screen.findByText("Generating training table...");
        expect(alert).toBeInTheDocument();

        const cancelButton = await screen.findByRole("button", {
            name: "Cancel this job"
        });
        expect(cancelButton).toBeInTheDocument();

        await userEvent.click(cancelButton);

        const dialog = await screen.findByTestId("dialog-content");
        expect(dialog).toHaveTextContent("Cancel Training Job?");
        expect(
            screen.getByText(
                /Stopping now will discard all progress on this job. This cannot be undone./i
            )
        ).toBeInTheDocument();

        return {
            dialog,
            continueButton: screen.getByRole("button", { name: "Cancel job" })
        };
    };

    beforeEach(() => {
        vi.clearAllMocks();
        setDeploymentTypeInStore(deploymentType.SAAS);
        queryClient.clear();
        // Suppress RUNNING/QUEUED refetchInterval timer accumulation across
        // tests.
        focusManager.setFocused(false);
        setTrainingGenResponseStatus(jobStatus.Done);
        setTrainingGenTestResponseType(testResponseType.ALL);
        setTrainingWorkflowTestResponseType(testResponseType.ALL);
        setCancelTrainingTableResponseType(testResponseType.ALL);
    });

    afterEach(() => {
        setCustomTrainingWorkflowResponse(null);
        focusManager.setFocused(undefined);
    });

    it("Should render TrainingTableGeneration page", async () => {
        const expectedResponse = doneResponse;
        expectedResponse.resource.job_id = id;
        await renderTrainingjobPage(defaultProps);
        await pageLoadTest(id, expectedResponse.resource.tags.parent_job);

        await verifyQueryText();
        await verifyStatus("Done");

        const statsTabs = ["Training Data Statistics", "Split stats"];
        statsTabs.forEach((text) => {
            const element = screen.getAllByText(text);
            expect(element.length).toBeGreaterThanOrEqual(1);
        });

        const relatedJobs = screen.queryByText("GRAPH SNAPSHOT");
        expect(relatedJobs).not.toBeInTheDocument();
    });

    // Regression test for kumo-ai/kumo#30119: the page fired the warnings
    // request with its own gen-traintable id, which the training-job-only
    // warnings route rejects with a guaranteed 404 on every page load.
    it("Should fetch warnings for the parent training job, never the gen-traintable id", async () => {
        const expectedResponse = doneResponse;
        expectedResponse.resource.job_id = id;
        await renderTrainingjobPage(defaultProps);
        await pageLoadTest(id, expectedResponse.resource.tags.parent_job);

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/training_jobs/${expectedResponse.resource.tags.parent_job}/warnings`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        expect(mockAxiosInstance.post).not.toHaveBeenCalledWith(
            `v0/training_jobs/${id}/warnings`,
            undefined,
            expect.anything()
        );
    });

    it("Should call `onArchive` when the Archive button is clicked", async () => {
        const expectedResponse = doneResponse;
        expectedResponse.resource.job_id = id;
        await renderTrainingjobPage(defaultProps);
        await pageLoadTest(id, expectedResponse.resource.tags.parent_job);

        const dropdownButton = await screen.findByTestId(
            "more-options-dropdown"
        );
        await userEvent.click(dropdownButton);

        const archiveButton = await screen.findByText(/Archive/i);
        expect(archiveButton).toBeInTheDocument();

        await userEvent.click(archiveButton);

        const dialog = await screen.findByTestId("dialog-content");
        expect(dialog).toBeInTheDocument();

        expect(dialog).toHaveTextContent("Archive Job");
        expect(dialog).toHaveTextContent(
            `Are you sure you want to archive Job "${getIdHash(id)}"? You'll still be able to restore it later if needed.`
        );

        const continueButton = screen.getByRole("button", { name: "Archive" });
        await userEvent.click(continueButton);

        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
            `v0/jobs/${id}/archive`,
            undefined,
            {
                headers: undefined,
                params: {
                    archived: true,
                    project_id: "project-tests"
                }
            }
        );

        expect(mockOnArchive).toHaveBeenCalled();
    });

    it("Should call `onRecover` when the Restore Archive button is clicked", async () => {
        setTrainingGenTestResponseType(testResponseType.ARCHIVED);
        const expectedResponse = doneResponse;
        expectedResponse.resource.job_id = id;
        await renderTrainingjobPage(defaultProps);
        await pageLoadTest(id, expectedResponse.resource.tags.parent_job);

        const restoreButton = await screen.findByTestId("restore-button");
        expect(restoreButton).toBeInTheDocument();

        await userEvent.click(restoreButton);

        const dialog = await screen.findByTestId("dialog-content");
        expect(dialog).toBeInTheDocument();

        expect(dialog).toHaveTextContent("Restore Job");
        expect(dialog).toHaveTextContent(
            `Do you want to restore Job "${getIdHash(id)}"?`
        );

        const continueButton = screen.getByRole("button", { name: "Restore" });
        await userEvent.click(continueButton);

        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
            `v0/jobs/${id}/archive`,
            undefined,
            {
                headers: undefined,
                params: {
                    archived: false,
                    project_id: "project-tests"
                }
            }
        );

        expect(mockOnRecover).toHaveBeenCalled();
        setTrainingGenTestResponseType(testResponseType.ALL);
    });

    it("Should render TrainingTableGeneration page with `Related Jobs`", async () => {
        const expectedResponse = doneResponse;
        expectedResponse.resource.job_id = id;
        await renderTrainingjobPage({
            ...defaultProps,
            showRelatedJobs: true
        });
        await pageLoadTest(id, expectedResponse.resource.tags.parent_job);

        await verifyQueryText();
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
        const expectedResponse = doneResponse;
        expectedResponse.resource.job_id = id;
        setGenStatsResponseType(testResponseType.DEFAULT);
        setSplitStatsResponseType(testResponseType.DEFAULT);
        await renderTrainingjobPage({
            ...defaultProps,
            showRelatedJobs: true
        });
        await pageLoadTest(id, expectedResponse.resource.tags.parent_job);

        await verifyStatus("Done");

        const APIS_TO_CALL = [
            `v0/training_jobs/${expectedResponse.resource.tags.parent_job}/get_train_table_split_stats`,
            `v0/gentraintable_jobs/${id}/generation_stats`
        ];
        expect(mockAxiosInstance.get).toHaveBeenCalledWith(
            APIS_TO_CALL[0],
            undefined,
            {
                headers: undefined,
                params: { project_id: "project-tests" }
            }
        );
        expect(mockAxiosInstance.get).toHaveBeenCalledWith(
            APIS_TO_CALL[1],
            undefined,
            {
                headers: undefined,
                params: { project_id: "project-tests" }
            }
        );

        const splits = ["Training", "Validation", "Holdout"];
        splits.forEach((text) => {
            const element = screen.queryAllByText(text);
            expect(element.length).toBeGreaterThanOrEqual(1);
        });
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

    it("Should render TrainingTableGeneration page for `Failed` status", async () => {
        const expectedResponse = doneResponse;
        expectedResponse.resource.job_id = id;
        setTrainingGenResponseStatus(jobStatus.Failed);
        await renderTrainingjobPage({
            ...defaultProps,
            showRelatedJobs: true
        });
        await pageLoadTest(id, expectedResponse.resource.tags.parent_job);

        await verifyStatus("Failed");

        const error = await screen.findByText(
            "This job has failed. Please contact Kumo for more information."
        );
        expect(error).toBeInTheDocument();
    });

    it("Should render TrainingTableGeneration page for `Cancelled` status", async () => {
        const expectedResponse = doneResponse;
        expectedResponse.resource.job_id = id;
        setTrainingGenResponseStatus(jobStatus.Cancelled);
        await renderTrainingjobPage({
            ...defaultProps,
            showRelatedJobs: true
        });
        await pageLoadTest(id, expectedResponse.resource.tags.parent_job);

        await verifyStatus("Canceled");
    });

    it("Should render TrainingTableGeneration page for `Running` status", async () => {
        const expectedResponse = doneResponse;
        expectedResponse.resource.job_id = id;
        setTrainingGenResponseStatus(jobStatus.Running);
        await renderTrainingjobPage({
            ...defaultProps,
            showRelatedJobs: true
        });
        await pageLoadTest(id, expectedResponse.resource.tags.parent_job);

        await verifyStatus("Running");

        const alert = await screen.findByText("Generating training table...");
        expect(alert).toBeInTheDocument();

        const cancelButton = await screen.findByRole("button", {
            name: "Cancel this job"
        });
        expect(cancelButton).toBeInTheDocument();
    });

    it("Should render TrainingTableGeneration page for `Queued` status", async () => {
        const expectedResponse = doneResponse;
        expectedResponse.resource.job_id = id;
        setTrainingGenResponseStatus(jobStatus.Queued);
        await renderTrainingjobPage({
            ...defaultProps,
            showRelatedJobs: true
        });
        await pageLoadTest(id, expectedResponse.resource.tags.parent_job);

        await verifyStatus("Queued");

        const alert = await screen.findByText("This job is queued");
        expect(alert).toBeInTheDocument();

        const cancelButton = await screen.findByRole("button", {
            name: "Cancel this job"
        });
        expect(cancelButton).toBeInTheDocument();
    });

    it("Should open `Cancel Training!` confirmation dialog and cancel job successfully", async () => {
        const { continueButton } = await setupRunningJobAndOpenCancelDialog();

        await userEvent.click(continueButton);

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/gentraintable_jobs/${id}/cancel`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        await waitFor(() => {
            expect(
                screen.queryByTestId("dialog-content")
            ).not.toBeInTheDocument();
        });
    });

    it("Should display error message when training table job cancellation fails", async () => {
        setCancelTrainingTableResponseType(testResponseType.ERROR);
        const { continueButton } = await setupRunningJobAndOpenCancelDialog();

        await userEvent.click(continueButton);

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/gentraintable_jobs/${id}/cancel`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        await waitFor(() => {
            expect(
                screen.queryByTestId("dialog-content")
            ).not.toBeInTheDocument();
        });

        const errorMessage = await screen.findByText(
            /Unable to Cancel Training Table Job\./i
        );
        expect(errorMessage).toBeInTheDocument();
    });

    it("Should render TrainingTableGeneration page for `Invalid response`", async () => {
        const expectedResponse = doneResponse;
        expectedResponse.resource.job_id = id;
        setTrainingGenTestResponseType(testResponseType.INVALID);
        await renderTrainingjobPage({
            ...defaultProps,
            showRelatedJobs: true
        });
        await pageLoadTest(
            id,
            expectedResponse.resource.tags.parent_job,
            false
        );
    });

    it("Should render TrainingTableGeneration page and show `skeleton` for Query", async () => {
        setTrainingWorkflowTestResponseType(testResponseType.ERROR);
        await renderTrainingjobPage({
            ...defaultProps,
            showRelatedJobs: true
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/gentraintable_jobs/${id}`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        const skeleton = document.querySelector('[data-slot="skeleton"]');
        expect(skeleton).toBeInTheDocument();

        setTrainingWorkflowTestResponseType(testResponseType.ALL);
    });

    it("Should render TrainingTableGeneration page for `Internal server error` ", async () => {
        setTrainingGenTestResponseType(testResponseType.ERROR);
        await renderTrainingjobPage({
            ...defaultProps,
            showRelatedJobs: true
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/gentraintable_jobs/${id}`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        const dialog = await screen.findByRole("dialog");
        expect(dialog).toHaveTextContent("Something went wrong!");
    });

    it("Should handle user metadata from response", async () => {
        setTrainingGenTestResponseType(testResponseType.DEFAULT);

        await renderTrainingjobPage({
            ...defaultProps,
            showRelatedJobs: false
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/gentraintable_jobs/${id}`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        await verifyStatus("Done");
        expect(await screen.findByText("User")).toBeInTheDocument();
        expect(
            await screen.findByText("test-user@example.com")
        ).toBeInTheDocument();
    });

    it("Should use queryTypeFromMetadata when no queryResponse and no queryInfo", async () => {
        setTrainingGenTestResponseType(testResponseType.FILTERED_OUT);
        await renderTrainingjobPage({
            ...defaultProps,
            queryInfo: undefined,
            showRelatedJobs: false
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/gentraintable_jobs/${id}`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        await verifyStatus("Done");
        expect(await screen.findByText("Training Table")).toBeInTheDocument();

        expect(
            await screen.findByRole("heading", { name: "Overview" })
        ).toBeInTheDocument();

        expect(await screen.findByTestId("pql-editor")).toBeInTheDocument();

        expect(
            await screen.findByText("Training Data Statistics")
        ).toBeInTheDocument();
    });
});
