// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { createQueryClient } from "tests/testUtil";

import {
    minimalBpJob,
    setCustomBatchPredictionResponse
} from "components/Jobs/Prediction/__tests__/MockFunctions/BatchPredictionResponse.mock";
import {
    minimalGenPredTableJob,
    setCustomGenPredictionTableResponse
} from "components/Jobs/PredictionTable/__tests__/MockFunctions/PredictionTableGeneration.mock";
import {
    minimalTrainingWorkflow,
    setCustomTrainingWorkflowResponse
} from "components/Jobs/Training/__tests__/MockFunctions/TrainingJobWorkflow.mock";

import { JobOverviewType } from "types/jobs";

import { getIdHash } from "utils/Helpers/stringHelpers";

import JobOverview, { JobOverviewProps } from "../JobOverview";

const trainJobId = "trainingjob-00046000000000000000000000000000";
const mockOnClose = vi.fn();
const mockOnArchive = vi.fn();
const mockOnRecover = vi.fn();

const defaultProps: JobOverviewProps = {
    jobId: null,
    jobOverviewType: null,
    onClose: mockOnClose,
    onArchive: mockOnArchive,
    onRecover: mockOnRecover
};

describe("JobOverview", () => {
    const queryClient = createQueryClient();
    const renderJobOverview = async (props: JobOverviewProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <JobOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const testArchiveFlow = async (id: string) => {
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

        await waitFor(() => {
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
        });

        expect(mockOnArchive).toHaveBeenCalled();
    };

    const testCloseFlow = async (
        id: string,
        expectedApiCalls: Array<{
            method: "get" | "post";
            args: unknown[];
        }> = []
    ) => {
        expect(screen.queryByRole("dialog")).toBeInTheDocument();
        const closeIcon = await screen.findByTestId("close-icon");
        expect(closeIcon).toBeInTheDocument();

        await userEvent.click(closeIcon);

        for (const apiCall of expectedApiCalls) {
            await waitFor(() => {
                const axiosMethod = mockAxiosInstance[apiCall.method];
                expect(axiosMethod).toHaveBeenCalledWith(...apiCall.args);
            });
        }

        await waitFor(
            () => {
                expect(mockOnClose).toHaveBeenCalled();
            },
            { timeout: 2000 }
        );
    };

    const verifyDisplayId = async (id: string, prefix: string) => {
        const displayId = id.replace(prefix, "").slice(0, 5);
        await waitFor(() => {
            expect(screen.getAllByText(displayId).length).toBeGreaterThan(0);
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    afterEach(() => {
        setCustomTrainingWorkflowResponse(null);
        setCustomBatchPredictionResponse(null);
        setCustomGenPredictionTableResponse(null);
    });

    it("Should render `JobOverview` when all the props are set null", async () => {
        await renderJobOverview(defaultProps);

        const element = document.getElementById("jobs-side-window-view");
        expect(element).not.toBeNull();
    });

    it("Should render `TrainingJobOverview` when jobOverviewType is `TRAINING_JOB`", async () => {
        const id = "trainingjob-00046000000000000000000000000000";
        // Dispatcher test: only verifies the right overview renders + archive/
        // close flows. Minimal workflow keeps the training section mounting (so
        // its warnings call still fires) but skips the eval/explanation cascade.
        setCustomTrainingWorkflowResponse(minimalTrainingWorkflow(id, "DONE"));
        await renderJobOverview({
            ...defaultProps,
            jobId: id,
            jobOverviewType: JobOverviewType.TRAINING_JOB
        });

        expect(screen.getByText("Training Job")).toBeInTheDocument();
        await verifyDisplayId(id, "trainingjob-");

        await testArchiveFlow(id);

        await testCloseFlow(id, [
            {
                method: "post",
                args: [
                    `v0/training_jobs/${id}/warnings`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                ]
            }
        ]);
    });

    it("Should render `TrainingTableOverview` when jobOverviewType is `TRAIN_TABLE_JOB`", async () => {
        const id = "gen-traintable-job-00046000000000000000000000000000";
        await renderJobOverview({
            ...defaultProps,
            jobId: id,
            jobOverviewType: JobOverviewType.TRAIN_TABLE_JOB
        });

        expect(await screen.findByText("Training Table")).toBeInTheDocument();

        await verifyDisplayId(id, "gen-traintable-job-");

        await testCloseFlow(id, [
            {
                method: "get",
                args: [
                    `v0/gentraintable_jobs/${id}`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                ]
            },
            {
                method: "get",
                args: [
                    `v0/gentraintable_jobs/${id}/generation_stats`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                ]
            },
            {
                method: "get",
                args: [
                    `v0/training_jobs/${trainJobId}/workflow`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                ]
            },
            {
                method: "get",
                args: [
                    `v0/snapshots/table/table-bc150313117da0324bfe31ca4ff499e5@1664307208/col_stats`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                ]
            },
            {
                // kumo-ai/kumo#30119: warnings are fetched for the parent
                // training job, never for the gen-traintable id (which the
                // training-job warnings route rejects with a 404).
                method: "post",
                args: [
                    `v0/training_jobs/${trainJobId}/warnings`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                ]
            }
        ]);
    });

    it("Should render `PredictionJobOverview` when jobOverviewType is `BATCH_PREDICTION_JOB`", async () => {
        const id = "bp-job-0001a000000000000000000000000000";
        // The BP overview embeds the related training job; a minimal BP
        // response nulls the pred-table/graph/eval cascade while keeping the
        // training link (so its warnings call still fires on close), and the
        // minimal workflow keeps that training section from mounting eval.
        setCustomBatchPredictionResponse(
            minimalBpJob(id, "DONE", null, trainJobId)
        );
        setCustomTrainingWorkflowResponse(
            minimalTrainingWorkflow(trainJobId, "DONE")
        );
        await renderJobOverview({
            ...defaultProps,
            jobId: id,
            jobOverviewType: JobOverviewType.BATCH_PREDICTION_JOB
        });

        expect(await screen.findByText("Batch Prediction")).toBeInTheDocument();

        await verifyDisplayId(id, "bp-job-");

        await testArchiveFlow(id);

        await testCloseFlow(id, [
            {
                method: "post",
                args: [
                    `v0/training_jobs/${trainJobId}/warnings`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                ]
            }
        ]);
    });

    it("Should render `PredictionTableOverview` when jobOverviewType is `PREDICTION_TABLE_JOB`", async () => {
        const id = "gen-predtable-job-00060000000000000000000000000000";
        // Dispatcher test: a minimal gen-pred-table response nulls the nested
        // graph_snapshot_id/pquery_id so the embedded GraphSnapshotOverview and
        // PQ panels don't mount (skips their graph-viz/metadata cascade). The
        // close-flow still asserts the genpredtable_jobs/{id} GET (URL-based, so
        // unaffected by the response body).
        setCustomGenPredictionTableResponse(minimalGenPredTableJob(id, "DONE"));
        await renderJobOverview({
            ...defaultProps,
            jobId: id,
            jobOverviewType: JobOverviewType.PREDICTION_TABLE_JOB
        });

        expect(
            await screen.findByText("Prediction Table Generation")
        ).toBeInTheDocument();

        await verifyDisplayId(id, "gen-predtable-job-");

        await testArchiveFlow(id);

        await testCloseFlow(id, [
            {
                method: "get",
                args: [
                    `v0/genpredtable_jobs/${id}`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                ]
            }
        ]);
    });

    it("Should render `GraphSnapshotOverview` when jobOverviewType is `GRAPH_SNAPSHOT_JOB`", async () => {
        const id =
            "graph-0005b000000000000000000000000000@00033000000000000000000000000000";
        await renderJobOverview({
            ...defaultProps,
            jobId: id,
            jobOverviewType: JobOverviewType.GRAPH_SNAPSHOT_JOB
        });

        expect(await screen.findByText("Overview")).toBeInTheDocument();

        await verifyDisplayId(id, "graph-");

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/snapshots/graph`,
                {
                    "graph_snapshot_id": `${id}`,
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
    });

    it("Should not display Sheet when jobId is null", async () => {
        await renderJobOverview({
            ...defaultProps,
            jobOverviewType: JobOverviewType.TRAINING_JOB
        });

        expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
});
