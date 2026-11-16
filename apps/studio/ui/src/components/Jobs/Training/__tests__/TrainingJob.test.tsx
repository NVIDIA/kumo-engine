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

import { jobStatus } from "utils/Constants";
import { getIdHash } from "utils/Helpers/stringHelpers";
import { jobRunTimeFromJobResponse, secondsToHm } from "utils/timeConverter";

import TrainingJobOverview, { TrainingJobOverviewProps } from "../TrainingJob";

import { setCancelTrainingResponseType } from "./MockFunctions/CancelTraining.mock";
import {
    minimalTrainingWorkflow,
    setCustomTrainingWorkflowResponse,
    setTrainingWorkflowResponseStatus,
    setTrainingWorkflowTestResponseType
} from "./MockFunctions/TrainingJobWorkflow.mock";
import mockWorkflowResponse from "./MockResponses/TrainingWorkflow.test.json";

const id = "trainingjob-00059000000000000000000000000000";
const mockOnArchive = vi.fn();
const mockOnRecover = vi.fn();

const defaultProps: TrainingJobOverviewProps = {
    hidePreTitle: false,
    id: id,
    onClose: vi.fn(),
    setStatus: vi.fn(),
    showRelatedJobs: false,
    onArchive: mockOnArchive,
    onRecover: mockOnRecover
};

describe("TrainingJob", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });
    const queryClient = createQueryClient(false);

    const renderTrainingjobPage = async (props: TrainingJobOverviewProps) => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={["/project/project-test/models/training/"]}>
                    <QueryClientProvider client={queryClient}>
                        <TrainingJobOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async (jobId: string) => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/training_jobs/${jobId}/workflow`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });
        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/training_jobs/${jobId}/warnings`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });
    };

    const waitForJobIdToLoad = async (jobId: string) => {
        const displayId = jobId.replace("trainingjob-", "").slice(0, 5);
        const jobIdElement = await screen.findAllByText(displayId);
        expect(jobIdElement?.length).toBeGreaterThanOrEqual(1);
    };

    beforeEach(() => {
        vi.clearAllMocks();
        setDeploymentTypeInStore(deploymentType.SAAS);
        queryClient.clear();
        // Stop background refetch/retry timers from firing fetch cascades
        // during the test. These are behaviour-neutral for assertions but,
        // for RUNNING jobs with refetchInterval, add seconds per test.
        queryClient.setDefaultOptions({
            queries: {
                retry: false,
                retryDelay: 0,
                refetchInterval: false,
                refetchOnWindowFocus: false,
                refetchOnReconnect: false
            }
        });
        focusManager.setFocused(false);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
        setCustomTrainingWorkflowResponse(null);
    });

    it("Should render TrainingJob page without `Related Jobs`", async () => {
        // Asserts only the absence of related-job sections + that the header
        // job ID loads — no eval/explanation content. Minimal workflow keeps
        // the page mounting without the eval cascade.
        setCustomTrainingWorkflowResponse(minimalTrainingWorkflow(id, "DONE"));
        await renderTrainingjobPage(defaultProps);
        await checkApiCall(id);
        await waitForJobIdToLoad(id);

        expect(screen.queryByText("GRAPH SNAPSHOT")).not.toBeInTheDocument();
        expect(
            screen.queryByText("TRAINING TABLE GENERATION")
        ).not.toBeInTheDocument();
    });

    it("Should render TrainingJob page with `Related Jobs`", async () => {
        await renderTrainingjobPage({
            ...defaultProps,
            showRelatedJobs: true
        });
        await checkApiCall(id);
        await waitForJobIdToLoad(id);

        //Graph Snapshot
        const graphSnap = await screen.findByText("Graph Snapshot");
        expect(graphSnap).toBeInTheDocument();
        const graphId =
            mockWorkflowResponse.generation_job.config.graph_snapshot_id;

        const graphIdElement = await screen.findAllByText(
            graphId?.replace("graph-", "").substring(0, 5)
        );
        expect(graphIdElement.length).toBeGreaterThan(0);

        //Training Table Generation
        const trainingTable = await screen.findByText(
            "Training Table Generation"
        );
        expect(trainingTable).toBeInTheDocument();

        const genTable = mockWorkflowResponse.generation_job.job_id;

        const genTableElement = await screen.findAllByText(
            genTable?.replace("gen-traintable-job-", "").substring(0, 5)
        );
        expect(genTableElement.length).toBeGreaterThan(0);
    });

    it("Should render TrainingJob page with `Overview` tabs", async () => {
        // Asserts tab presence/enabled state only (not eval/explanation
        // content) — tab visibility is gated on status (DONE), so a minimal
        // workflow renders all three tabs without the eval cascade.
        setCustomTrainingWorkflowResponse(minimalTrainingWorkflow(id, "DONE"));
        await renderTrainingjobPage(defaultProps);
        await checkApiCall(id);

        // Check Training, Evaluation, Explanation Tabs
        const tabs = ["Training", "Evaluation", "Explanation"];
        for (const tab of tabs) {
            const elements = screen.queryAllByRole("button", { name: tab });

            for (const element of elements) {
                expect(element).toBeInTheDocument();
                expect(element).toBeEnabled();
                await userEvent.click(element);
            }
        }
    });

    it("Should render TrainingJob page for `Done` status", async () => {
        await renderTrainingjobPage(defaultProps);
        await checkApiCall(id);

        const status = await screen.findAllByTestId("header-status-chip-id");
        expect(status[0].textContent).toBe("Done");

        const labels = ["Created", "Total runtime", "Task type", "Source"];
        for (const label of labels) {
            expect(await screen.findByText(label)).toBeInTheDocument();
        }

        // The runtime shown is the admission-to-close wall clock, computed by
        // the same helper the jobs-list Duration column uses, so the two
        // surfaces agree. It is not the trainer-reported training time.
        const expectedWorkflow = { ...mockWorkflowResponse };
        const wallClock = jobRunTimeFromJobResponse(
            expectedWorkflow?.training_job?.execution?.start_time,
            expectedWorkflow?.training_job?.execution?.last_updated_at
        );
        expect(screen.queryByText(wallClock)).toBeInTheDocument();

        const trainerTime = secondsToHm(
            expectedWorkflow?.training_job?.execution?.results
                ?.total_training_time_sec
        );
        expect(
            screen.queryByText(trainerTime as string)
        ).not.toBeInTheDocument();
    });

    it("Should call `onArchive` when the Archive button is clicked", async () => {
        // Header-action test only — use a minimal workflow so the eval/
        // explanation tab cascade does not mount (faster, same assertions).
        setCustomTrainingWorkflowResponse(minimalTrainingWorkflow(id, "DONE"));
        await renderTrainingjobPage(defaultProps);
        await checkApiCall(id);

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

        const continueButton = await screen.findByRole("button", {
            name: "Archive"
        });
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
        // archived tag drives isArchived=true (shows restore button); minimal
        // workflow keeps the test focused on the recover action.
        setCustomTrainingWorkflowResponse(
            minimalTrainingWorkflow(id, "DONE", { archived: "archived" })
        );
        await renderTrainingjobPage(defaultProps);
        await checkApiCall(id);

        const restoreButton = await screen.findByTestId("restore-button");
        expect(restoreButton).toBeInTheDocument();

        await userEvent.click(restoreButton);

        const dialog = await screen.findByTestId("dialog-content");
        expect(dialog).toBeInTheDocument();

        expect(dialog).toHaveTextContent("Restore Job");
        expect(dialog).toHaveTextContent(
            `Do you want to restore Job "${getIdHash(id)}"?`
        );

        const continueButton = await screen.findByRole("button", {
            name: "Restore"
        });
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
        setTrainingWorkflowTestResponseType(testResponseType.ALL);
    });

    describe("Job Status Tests", () => {
        const statusTestCases: Array<{
            status: jobStatus;
            displayStatus: string;
            expectedText?: string;
            hasCancelButton?: boolean;
        }> = [
            {
                status: jobStatus.Failed,
                displayStatus: "Failed",
                expectedText: "Training job failed"
            },
            {
                status: jobStatus.Cancelled,
                displayStatus: "Canceled"
            },
            {
                status: jobStatus.Running,
                displayStatus: "Running",
                expectedText: "Ingesting Data",
                hasCancelButton: true
            },
            {
                status: jobStatus.Queued,
                displayStatus: "Queued",
                expectedText: "This training job is queued",
                hasCancelButton: true
            }
        ];

        it.each(statusTestCases)(
            "Should render TrainingJob page for `$displayStatus` status",
            async ({
                status,
                displayStatus,
                expectedText,
                hasCancelButton
            }) => {
                setTrainingWorkflowResponseStatus(status);
                await renderTrainingjobPage({
                    ...defaultProps,
                    showRelatedJobs: status === jobStatus.Failed
                });
                await checkApiCall(id);
                await waitForJobIdToLoad(id);

                const statusChip = await screen.findAllByTestId(
                    "header-status-chip-id"
                );
                expect(statusChip[0].textContent).toBe(displayStatus);

                if (expectedText) {
                    const alert = await screen.findByText(expectedText);
                    expect(alert).toBeInTheDocument();
                }

                if (status === jobStatus.Failed) {
                    const description = await screen.findByText(
                        /workflow did not complete due to the following validation issues/i
                    );
                    expect(description).toBeInTheDocument();
                }

                if (hasCancelButton) {
                    const stopButton = await screen.findByRole("button", {
                        name: "Cancel this job"
                    });
                    expect(stopButton).toBeInTheDocument();
                }

                setTrainingWorkflowResponseStatus(jobStatus.Done);
            }
        );
    });

    describe("Cancel Training Job", () => {
        const setupCancelTest = async () => {
            // Cancel-flow test only — minimal RUNNING workflow renders the
            // status chip + cancel control without the eval cascade.
            setCustomTrainingWorkflowResponse(
                minimalTrainingWorkflow(id, jobStatus.Running)
            );
            await renderTrainingjobPage(defaultProps);
            await checkApiCall(id);

            const status = await screen.findAllByTestId(
                "header-status-chip-id"
            );
            expect(status[0].textContent).toBe("Running");

            const stopButton = await screen.findByRole("button", {
                name: "Cancel this job"
            });
            await userEvent.click(stopButton);

            const dialog = await screen.findByTestId("dialog-content");
            expect(dialog).toHaveTextContent("Cancel Training Job?");

            const continueButton = await screen.findByRole("button", {
                name: "Cancel job"
            });
            return { continueButton, dialog };
        };

        it("Should open confirmation dialog and call cancel API when cancelling a running training job", async () => {
            const { continueButton } = await setupCancelTest();

            await userEvent.click(continueButton);

            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    `v0/training_jobs/${id}/cancel`,
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

        it("Should display error message when training job cancellation fails", async () => {
            setCancelTrainingResponseType(testResponseType.ERROR);
            const { continueButton } = await setupCancelTest();

            await userEvent.click(continueButton);

            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    `v0/training_jobs/${id}/cancel`,
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
                /Unable to Cancel Training\./i
            );
            expect(errorMessage).toBeInTheDocument();

            setCancelTrainingResponseType(testResponseType.DEFAULT);
        });
    });

    it("Should set Training tab as the active tab by default", async () => {
        // Default active tab is fixed (Evaluation) regardless of eval data, so
        // a minimal DONE workflow exercises the same behaviour without the
        // eval cascade.
        setCustomTrainingWorkflowResponse(minimalTrainingWorkflow(id, "DONE"));

        await renderTrainingjobPage(defaultProps);

        await checkApiCall(id);

        await waitFor(() => {
            const activeTab = document.querySelector(`[data-state="active"]`);
            expect(activeTab).toBeInTheDocument();
            expect(activeTab?.textContent).toBe("Evaluation");
        });
    });

    it("Should update queryTab when Evaluation and Explanation tabs are selected", async () => {
        // Tab switching is pure UI (aria-selected) and independent of eval
        // content, so a minimal DONE workflow avoids the eval cascade.
        setCustomTrainingWorkflowResponse(minimalTrainingWorkflow(id, "DONE"));
        await renderTrainingjobPage(defaultProps);

        await checkApiCall(id);

        const evaluationTab = await screen.findByRole("tab", {
            name: "Evaluation"
        });
        const explanationTab = await screen.findByRole("tab", {
            name: "Explanation"
        });

        await userEvent.click(evaluationTab);
        expect(evaluationTab).toHaveAttribute("aria-selected", "true");

        await userEvent.click(explanationTab);
        expect(explanationTab).toHaveAttribute("aria-selected", "true");
    });

    describe("Tabs visibility based on job status", () => {
        const tabVisibilityTestCases = [
            { status: jobStatus.Failed, statusName: "Failed" },
            { status: jobStatus.Cancelled, statusName: "Cancelled" }
        ];

        it.each(tabVisibilityTestCases)(
            "Should not render the Tabs when the job status is $statusName",
            async ({ status }) => {
                // Tab visibility is gated purely on status (Failed/Cancelled
                // hide all tabs); a minimal workflow exercises the same gate
                // without the full status fixture cascade.
                setCustomTrainingWorkflowResponse(
                    minimalTrainingWorkflow(id, status)
                );

                await renderTrainingjobPage(defaultProps);
                await checkApiCall(id);

                const tabs = ["Training", "Evaluation", "Explanation"];
                tabs.forEach((tab) => {
                    const elements = screen.queryAllByRole("button", {
                        name: tab
                    });
                    expect(elements.length).toBe(0);
                });
            }
        );
    });

    describe("Error Response Tests", () => {
        const errorResponseCases: Array<{
            responseType: testResponseType;
            testDescription: string;
            expectedContent: string;
        }> = [
            {
                responseType: testResponseType.INVALID,
                testDescription: "Invalid response",
                expectedContent:
                    "Sorry, there is an error with one of the inputs"
            },
            {
                responseType: testResponseType.ERROR,
                testDescription: "Internal server error",
                expectedContent: "Something went wrong!"
            }
        ];

        it.each(errorResponseCases)(
            "Should render TrainingJob page for `$testDescription` status",
            async ({ responseType, expectedContent }) => {
                setTrainingWorkflowTestResponseType(responseType);
                await renderTrainingjobPage(defaultProps);
                await checkApiCall(id);

                await waitForJobIdToLoad(id);

                if (responseType === testResponseType.ERROR) {
                    const dialog = await screen.findByRole("dialog");
                    expect(dialog).toHaveTextContent(expectedContent);
                    const closeButton = await screen.findByRole("button", {
                        name: "Cancel"
                    });
                    expect(closeButton).toBeInTheDocument();
                } else if (responseType === testResponseType.INVALID) {
                    const alert = await screen.findByRole("alert");
                    expect(alert).toHaveTextContent(expectedContent);
                    expect(alert).toHaveTextContent("Error in Training Job");
                }

                setTrainingWorkflowTestResponseType(testResponseType.DEFAULT);
            }
        );
    });
});
