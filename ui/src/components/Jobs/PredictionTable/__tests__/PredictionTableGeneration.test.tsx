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

import { setPredictiveQueryResoureTestResponseType } from "components/Jobs/Prediction/__tests__/MockFunctions/PredictiveQueryResource.mock";
import { PQueryResourceResponse } from "components/Jobs/Prediction/__tests__/MockResponses/PredictiveQueryResource";
import { setMetadataResponseType } from "components/Jobs/TrainingTable/__tests__/MockFunctions/PredictiveQueryMetaDeta.mock";

import { jobStatus } from "utils/Constants";
import { getIdHash } from "utils/Helpers/stringHelpers";
import { jobRunTimeFromJobResponse } from "utils/timeConverter";

import PredictionTableGeneration, {
    PredictionTableOverviewProps
} from "../PredictionTableGeneration";

import {
    setGenPredictionTableJobResponseStatus,
    setGenPredictionTableJobTestResponseType
} from "./MockFunctions/PredictionTableGeneration.mock";
import { doneResponse } from "./MockResponses/PredictionTableGeneration";

const id = "gen-predtable-job-960dbaace6ef49b8b83d648204f50e27";
const expectedPredictionTable = doneResponse;
expectedPredictionTable.resource.job_id = id;
const mockOnArchive = vi.fn();
const mockOnRecover = vi.fn();

const defaultProps: PredictionTableOverviewProps = {
    hidePreTitle: false,
    id: id,
    onClose: vi.fn(),
    setStatus: vi.fn(),
    showRelatedJobs: false,
    onArchive: mockOnArchive,
    onRecover: mockOnRecover
};

describe("PredictionTableGeneration", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });
    const queryClient = createQueryClient(false);

    const renderPredictionTableOverview = async (
        props: PredictionTableOverviewProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={[
                        "/project/project-test/predictions/predictiontable/"
                    ]}>
                    <QueryClientProvider client={queryClient}>
                        <PredictionTableGeneration {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async (id: string) => {
        await waitFor(() => {
            const APIS_TO_CALL = [
                `v0/genpredtable_jobs/${id}`,
                `v0/genpredtable_jobs/${id}/anchor_time`
            ];
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

    const pageLoadTest = async (id: string) => {
        expect(
            await screen.findByText("Prediction Table Generation")
        ).toBeInTheDocument();

        const displayId = id.replace("gen-predtable-job-", "").slice(0, 5);
        const jobIds = await screen.findAllByText(displayId);
        expect(jobIds.length).greaterThanOrEqual(1);
    };

    const verifyStatus = async (expectedStatus: string) => {
        const status = await screen.findAllByTestId("header-status-chip-id");
        expect(status[0].textContent).toBe(expectedStatus);
    };

    const verifyQueryText = async () => {
        const expectedPqueryResource = { ...PQueryResourceResponse };

        const queryCard = await screen.findByRole("textbox");

        const cmLines = queryCard.querySelectorAll(".cm-line");
        const queryText = Array.from(cmLines)
            .map((line) => line.textContent || "")
            .join("\n");

        await waitFor(() => {
            expect(normalizeString(queryText)).toBe(
                normalizeString(expectedPqueryResource.query_string)
            );
        });
    };

    const verifyCancelDialog = async () => {
        const cancelButton = await screen.findByRole("button", {
            name: "Cancel this job"
        });
        expect(cancelButton).toBeInTheDocument();
        expect(cancelButton).toBeEnabled();

        await userEvent.click(cancelButton);
        const dialog = await screen.findByRole("dialog");
        expect(dialog).toHaveTextContent("Cancel Prediction Job?");
        expect(dialog).toHaveTextContent(
            /Stopping now will discard all progress on this job. This cannot be undone./
        );

        return dialog;
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();

        // Suppress RUNNING/QUEUED refetchInterval timer accumulation across
        // tests.
        focusManager.setFocused(false);

        setDeploymentTypeInStore(deploymentType.SAAS);
        setGenPredictionTableJobResponseStatus(jobStatus.Done);
        setGenPredictionTableJobTestResponseType(testResponseType.ALL);
        setPredictiveQueryResoureTestResponseType(testResponseType.ALL);
        setMetadataResponseType(testResponseType.ALL);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
    });

    it("Should render PredictionTableOverview page", async () => {
        await renderPredictionTableOverview(defaultProps);
        await checkApiCall(id);

        await verifyQueryText();
        expect(
            await screen.findByText(/Binary Classification/i)
        ).toBeInTheDocument();

        const texts = [
            "Created",
            "Total runtime",
            "Task type",
            "Source",
            "Prediction anchor time"
        ];
        for (const text of texts) {
            const element = await screen.findByText(text);
            expect(element).toBeInTheDocument();
        }

        expect(screen.queryByText("Graph Snapshot")).not.toBeInTheDocument();
    });

    it("Should call `onArchive` when the Archive button is clicked", async () => {
        await renderPredictionTableOverview(defaultProps);
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
        setGenPredictionTableJobTestResponseType(testResponseType.ARCHIVED);
        await renderPredictionTableOverview(defaultProps);
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
    });

    it("Should render PredictionTableOverview page with Status `Done`", async () => {
        await renderPredictionTableOverview({
            ...defaultProps,
            showRelatedJobs: true
        });
        await checkApiCall(id);
        await pageLoadTest(id);

        await verifyQueryText();

        //Graph Snapshot
        const graphSnap = await screen.findByText("Graph Snapshot");
        expect(graphSnap).toBeInTheDocument();
        const graphId =
            expectedPredictionTable?.resource?.config?.graph_snapshot_id;
        const graphIdElement = screen.queryAllByText(
            graphId?.replace("graph-", "").substring(0, 5) ?? ""
        );
        expect(graphIdElement.length).toBeGreaterThan(0);

        const graphDropdown = await screen.findByText("Graph Snapshot");
        await userEvent.click(graphDropdown);

        //Total runtime
        let totalTime = "";
        const startPrediction = new Date(
            expectedPredictionTable?.resource?.job_status_report?.start_time
        );

        const endTime =
            expectedPredictionTable?.resource?.job_status_report?.end_time;
        if (endTime) {
            const endPrediction = new Date(endTime);

            totalTime = jobRunTimeFromJobResponse(
                startPrediction,
                endPrediction
            );

            const runtimeElement = screen.getByText((text) =>
                text.includes(totalTime)
            );
            expect(runtimeElement).toBeInTheDocument();
        }
    });

    it("Should render PredictionTableOverview page with Status `Failed`", async () => {
        const expectedPredictionTable = doneResponse;
        expectedPredictionTable.resource.job_id = id;
        setGenPredictionTableJobResponseStatus(jobStatus.Failed);
        await renderPredictionTableOverview(defaultProps);
        await checkApiCall(id);
        await pageLoadTest(id);

        await verifyStatus("Failed");

        await verifyQueryText();
        const error = await screen.findByText(
            "This job has failed. Please contact Kumo for more information."
        );
        expect(error).toBeInTheDocument();
    });

    it("Should render PredictionTableOverview page with Status `Cancelled`", async () => {
        setGenPredictionTableJobResponseStatus(jobStatus.Cancelled);
        await renderPredictionTableOverview(defaultProps);
        await checkApiCall(id);
        await pageLoadTest(id);

        await verifyStatus("Canceled");

        await verifyQueryText();
    });

    it("Should render PredictionTableOverview page with Status `Queued`", async () => {
        setGenPredictionTableJobResponseStatus(jobStatus.Queued);
        await renderPredictionTableOverview(defaultProps);
        await checkApiCall(id);
        await pageLoadTest(id);

        await verifyStatus("Queued");

        await verifyQueryText();
        const description = await screen.findByText("This job is queued");
        expect(description).toBeInTheDocument();

        const cancelButton = await screen.findByRole("button", {
            name: "Cancel this job"
        });
        expect(cancelButton).toBeInTheDocument();
        expect(cancelButton).toBeEnabled();
    });

    it("Should render PredictionTableOverview page with Status `Running`", async () => {
        setGenPredictionTableJobResponseStatus(jobStatus.Running);
        await renderPredictionTableOverview(defaultProps);
        await checkApiCall(id);
        await pageLoadTest(id);

        await verifyStatus("Running");

        await verifyQueryText();
        const description = await screen.findByText(
            "Generating prediction table..."
        );
        expect(description).toBeInTheDocument();

        const cancelButton = await screen.findByRole("button", {
            name: "Cancel this job"
        });
        expect(cancelButton).toBeInTheDocument();
        expect(cancelButton).toBeEnabled();
    });

    it("Should abort generating prediction table when cancel dialog is confirmed", async () => {
        setGenPredictionTableJobResponseStatus(jobStatus.Running);
        await renderPredictionTableOverview(defaultProps);
        await checkApiCall(id);

        await verifyCancelDialog();

        const continueButtton = await screen.findByRole("button", {
            name: "Cancel job"
        });
        await userEvent.click(continueButtton);

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/genpredtable_jobs/${id}/cancel`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        await waitFor(() => {
            expect(screen.queryByTestId("dialog")).not.toBeInTheDocument();
        });
    });

    it.each([
        [testResponseType.ERROR, "Error"],
        [testResponseType.INVALID, "Invalid"],
        [testResponseType.EMPTY, "Empty"]
    ])(
        "Should  show skeleton and not show Task type on %s response",
        async (responseType) => {
            setMetadataResponseType(responseType);
            setGenPredictionTableJobTestResponseType(responseType);
            await renderPredictionTableOverview(defaultProps);
            await checkApiCall(id);

            expect(
                screen.queryByText(/Binary Classification/i)
            ).not.toBeInTheDocument();

            const skeleton = document.querySelector('[data-slot="skeleton"]');
            expect(skeleton).toBeInTheDocument();
        }
    );

    it("Should render PredictionTableOverview Overview page for Error response", async () => {
        setGenPredictionTableJobTestResponseType(testResponseType.ERROR);
        await renderPredictionTableOverview(defaultProps);

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/genpredtable_jobs/${id}`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });
        const headerText = "Prediction Table Generation";
        await waitFor(() => {
            screen.getAllByText(headerText);
        });

        const displayId = id.replace("gen-predtable-job-", "").slice(0, 5);
        const jobId = await screen.findByText(displayId);
        expect(jobId).toBeInTheDocument();

        const skeleton = document.querySelector('[data-slot="skeleton"]');
        expect(skeleton).toBeInTheDocument();
    });
});
