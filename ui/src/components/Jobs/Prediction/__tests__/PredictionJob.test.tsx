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

import PredictionJob, {
    PredictionJobProps
} from "components/Jobs/Prediction/PredictionJob";
import {
    setTrainingWorkflowResponseStatus,
    setTrainingWorkflowTestResponseType
} from "components/Jobs/Training/__tests__/MockFunctions/TrainingJobWorkflow.mock";

import { jobStatus } from "utils/Constants";

import {
    minimalBpJob,
    setCustomBatchPredictionResponse
} from "./MockFunctions/BatchPredictionResponse.mock";
import { setCancelCancelBatchPredictionResponseType } from "./MockFunctions/CancelBatchPrediction.mock";
import { setPredictiveQueryResoureTestResponseType } from "./MockFunctions/PredictiveQueryResource.mock";

const id = "bp-job-3f5f52a53f01473d8c5123ff7ba4fd15";
const mockOnClose = vi.fn();
const mockOnArchive = vi.fn();
const mockOnRecover = vi.fn();

const defaultProps: PredictionJobProps = {
    bpId: id,
    onClose: mockOnClose,
    onArchive: mockOnArchive,
    onRecover: mockOnRecover
};

describe("PredictionJob", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });
    const queryClient = createQueryClient(false);

    const renderPredictionJob = async (props: PredictionJobProps) => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={[
                        "/project/project-test/predictions/prediction/"
                    ]}>
                    <QueryClientProvider client={queryClient}>
                        <PredictionJob {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async (id: string) => {
        await waitFor(() => {
            const APIS_TO_CALL = [`v1/batch_prediction_jobs/${id}`];
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

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
        queryClient.setDefaultOptions({
            queries: {
                retry: false,
                retryDelay: 0,
                refetchInterval: false,
                refetchOnWindowFocus: false,
                refetchOnReconnect: false
            }
        });
        // Prevent the refetchInterval timer from executing fetches when it fires.
        focusManager.setFocused(false);
        // Return an empty training workflow response so the related training job's
        // hooks (tensorboard, encoders, eval) all resolve enabled=false, preventing
        // the cascade of GET calls that inflates test duration.
        setTrainingWorkflowTestResponseType(testResponseType.EMPTY);

        setDeploymentTypeInStore(deploymentType.SAAS);
        setCancelCancelBatchPredictionResponseType(testResponseType.ALL);
        setPredictiveQueryResoureTestResponseType(testResponseType.ALL);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
        setTrainingWorkflowResponseStatus(jobStatus.Done);
        setTrainingWorkflowTestResponseType(testResponseType.ALL);
        setCustomBatchPredictionResponse(null);
    });

    it("Should render PredictionJob component with Done status", async () => {
        setCustomBatchPredictionResponse(minimalBpJob(id, "DONE"));
        await renderPredictionJob(defaultProps);
        await checkApiCall(id);

        const displayId = id.replace("bp-job-", "").slice(0, 5);
        expect(await screen.findByText(displayId)).toBeInTheDocument();

        const status = await screen.findAllByTestId("header-status-chip-id");
        expect(status[0].textContent).toBe("Done");

        expect(
            screen.queryByRole("button", { name: /Cancel this job/i })
        ).not.toBeInTheDocument();
    });

    it.each([
        [jobStatus.Running, "Running"],
        [jobStatus.Queued, "Queued"]
    ])("Should render cancel button for %s job", async (status) => {
        setCustomBatchPredictionResponse(minimalBpJob(id, status as string));
        await renderPredictionJob(defaultProps);

        const cancelButton = await screen.findByRole("button", {
            name: /Cancel this job/i
        });
        expect(cancelButton).toBeInTheDocument();
        expect(cancelButton).toBeEnabled();
    });

    it("Should handle Cancel/Continue button click event of cancel job via dialog", async () => {
        setCustomBatchPredictionResponse(minimalBpJob(id, "RUNNING"));
        await renderPredictionJob(defaultProps);
        await checkApiCall(id);

        const cancelButton = await screen.findByRole("button", {
            name: /Cancel this job/i
        });

        await userEvent.click(cancelButton);

        let dialog = await screen.findByRole("dialog");
        expect(dialog).toBeInTheDocument();

        const cancelDialogButton = await screen.findByRole("button", {
            name: "Cancel"
        });

        await userEvent.click(cancelDialogButton);

        await waitFor(() => {
            expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
        });

        await userEvent.click(cancelButton);

        dialog = await screen.findByRole("dialog");
        expect(dialog).toHaveTextContent("Cancel Prediction Job?");
        expect(dialog).toHaveTextContent(
            /Stopping now will discard all progress on this job. This cannot be undone/
        );

        const continueButton = await screen.findByRole("button", {
            name: "Cancel job"
        });

        await userEvent.click(continueButton);

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v1/batch_prediction_jobs/${id}/cancel`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });
    });

    it("Should display error message when cancellation fails", async () => {
        setCustomBatchPredictionResponse(minimalBpJob(id, "RUNNING"));
        await renderPredictionJob(defaultProps);
        await checkApiCall(id);

        const cancelButton = await screen.findByRole("button", {
            name: /Cancel this job/i
        });
        await userEvent.click(cancelButton);

        setCancelCancelBatchPredictionResponseType(testResponseType.ERROR);

        const continueButton = await screen.findByRole("button", {
            name: "Cancel job"
        });
        await userEvent.click(continueButton);

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v1/batch_prediction_jobs/${id}/cancel`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        await waitFor(() => {
            expect(
                screen.getByText(/Unable to Cancel Prediction Job\./i)
            ).toBeInTheDocument();
        });
    });

    it("Should open SDK Code modal when SDK Code option is clicked", async () => {
        setCustomBatchPredictionResponse(minimalBpJob(id, "DONE"));
        await renderPredictionJob(defaultProps);
        await checkApiCall(id);

        const dropdownButton = await screen.findByTestId(
            "more-options-dropdown"
        );
        await userEvent.click(dropdownButton);

        const sdkCodeItem = await screen.findByText(/SDK Code/i);
        await userEvent.click(sdkCodeItem);

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/batch_prediction_jobs/${id}/sdk`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        const modal = await screen.findByTestId("sdk-content");
        expect(modal).toBeInTheDocument();
        expect(modal).toHaveTextContent(/View SDK Code/i);
    });

    it("Should close SDK Code modal when close button is clicked", async () => {
        setCustomBatchPredictionResponse(minimalBpJob(id, "DONE"));
        await renderPredictionJob(defaultProps);
        await checkApiCall(id);

        const dropdownButton = await screen.findByTestId(
            "more-options-dropdown"
        );
        await userEvent.click(dropdownButton);

        const sdkCodeItem = await screen.findByText(/SDK Code/i);
        await userEvent.click(sdkCodeItem);

        const modal = await screen.findByTestId("sdk-content");
        expect(modal).toBeInTheDocument();

        const closeIcons = screen.getAllByTestId("close-icon");

        await userEvent.click(closeIcons[closeIcons.length - 1]);

        await waitFor(() => {
            expect(screen.queryByTestId("sdk-content")).not.toBeInTheDocument();
        });
    });

    it("Should archive job when Continue button is clicked in archive dialog", async () => {
        setCustomBatchPredictionResponse(minimalBpJob(id, "DONE"));
        await renderPredictionJob(defaultProps);
        await checkApiCall(id);

        const dropdownButton = await screen.findByTestId(
            "more-options-dropdown"
        );
        await userEvent.click(dropdownButton);

        const archiveButton = await screen.findByText(/Archive/i);
        await userEvent.click(archiveButton);

        const dialog = await screen.findByTestId("dialog-content");
        expect(dialog).toBeInTheDocument();
        expect(dialog).toHaveTextContent("Archive Job");

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

        await waitFor(() => {
            expect(mockOnArchive).toHaveBeenCalled();
        });
    });

    it("Should restore archived job when Restore Archive button is clicked", async () => {
        // archived tag makes isArchived=true, showing the restore button
        setCustomBatchPredictionResponse(
            minimalBpJob(id, "DONE", { archived: "archived" })
        );
        await renderPredictionJob(defaultProps);
        await checkApiCall(id);

        const restoreButton = await screen.findByTestId("restore-button");
        await userEvent.click(restoreButton);

        const dialog = await screen.findByTestId("dialog-content");
        expect(dialog).toBeInTheDocument();
        expect(dialog).toHaveTextContent("Restore Job");

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

        await waitFor(() => {
            expect(mockOnRecover).toHaveBeenCalled();
        });
    });
});
