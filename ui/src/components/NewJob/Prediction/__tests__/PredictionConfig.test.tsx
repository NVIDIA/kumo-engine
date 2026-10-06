// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GenericJob } from "gen/v0/types";
import { ErrorDetail } from "gen/v1/types";
import { createQueryClient } from "tests/testUtil";

import {
    minimalTrainingWorkflow,
    setCustomTrainingWorkflowResponse
} from "components/Jobs/Training/__tests__/MockFunctions/TrainingJobWorkflow.mock";

import mockJobsListResponse from "../../../../pages/Jobs/__tests__/MockResponses/JobsList.test.response.json";

import { PredictiveQueryInfo, TaskType } from "types/predictiveQueries";

import PredictionConfig, {
    PredictionConfigProps
} from "../Config/PredictionConfig";

const jobList = { ...mockJobsListResponse };
const mockTrainingJob = {
    ...jobList.jobs[0],
    id: "pquery-629e15c329bb119cbae6de2fc2021165",
    queryYaml:
        "PREDICT COUNT(FINANCE_TRANS.*, 0, 30, days) = 0 \nFOR EACH FINANCE_ACCOUNT.ACCOUNT_ID",
    graphName: "financial_graph",
    created: "2024-11-25T04:51:50.659325+00:00",
    dateCreated: "Mon Nov 25, 2024, 5:17:37 AM UTC",
    trainingStatus: "DONE",
    statusKey: "done",
    taskType: TaskType.BINARY_CLASSIFICATION,
    jobSource: "UI",
    graphSnapshotId:
        "graph-d445eeb8b2aa1b79a142406239fa74e8@6f0ae33ac31edb0c321fbe182e39e460",
    trainingJobId: "trainingjob-cd40820089504c91825ec0b413e97cbb"
} as unknown as GenericJob & PredictiveQueryInfo;

const defaultProps: PredictionConfigProps = {
    selectedTrainingJob: mockTrainingJob,
    setPredictionJobArgs: vi.fn(),
    setError: vi.fn(),
    setFilterErrors: vi.fn(),
    setValidationState: vi.fn(),
    setFilterValidState: vi.fn(),
    error: {} as ErrorDetail[]
};

describe("PredictionConfig", () => {
    const queryClient = createQueryClient();
    const renderPredictionConfig = async (props: PredictionConfigProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <PredictionConfig {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    afterEach(() => {
        setCustomTrainingWorkflowResponse(null);
    });

    it("Should render PredictionConfig with no training job id", async () => {
        await renderPredictionConfig({
            ...defaultProps,
            selectedTrainingJob: null
        });

        expect(screen.getByText("-")).toBeInTheDocument();
        expect(screen.getByText(/Training Job/i)).toBeInTheDocument();
    });

    it("Should open and close the training-job side panel", async () => {
        // The side panel mounts a full TrainingJobOverview; a minimal workflow
        // response keeps it rendering (header asserted) without the heavy
        // eval/explanation fetch cascade. See test-performance-playbook.md.
        setCustomTrainingWorkflowResponse(
            minimalTrainingWorkflow(
                mockTrainingJob.trainingJobId as string,
                "DONE"
            )
        );
        await renderPredictionConfig(defaultProps);

        const trainingJobId = "629e1";
        expect(screen.getByText(trainingJobId)).toBeInTheDocument();

        await userEvent.click(screen.getByText(trainingJobId));

        const dialog = await screen.findByRole("dialog");
        expect(dialog).toBeInTheDocument();

        await screen.findAllByText("Training Job");

        const closeButton = screen.getByTestId("close-icon");
        await userEvent.click(closeButton);

        await waitFor(() => {
            expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
        });
    });

    it("Should render the selected training job's queryYaml", async () => {
        await renderPredictionConfig(defaultProps);

        const queryCard = screen.getByRole("textbox");
        const cmLines = queryCard.querySelectorAll(".cm-line");
        const queryText = Array.from(cmLines)
            .map((line) => line.textContent || "")
            .join("\n");

        expect(queryText).toBe(mockTrainingJob.queryYaml);
    });

    it("Should render `Empty card` when training job's queryYaml is set null", async () => {
        await renderPredictionConfig({
            ...defaultProps,
            selectedTrainingJob: {
                ...mockTrainingJob,
                queryYaml: null
            } as unknown as GenericJob & PredictiveQueryInfo
        });

        const card = screen.getByTestId("pquery-card");
        const content = card.querySelector('[data-slot="query-card-content"]');
        expect(content?.textContent?.trim()).toBe("");
    });

    it("Should display PredictionSchema", async () => {
        await renderPredictionConfig(defaultProps);

        expect(screen.getByText("Schema")).toBeInTheDocument();

        const tableRole = screen.getByRole("tablist");
        expect(tableRole).toBeInTheDocument();
        const tabList = screen.getAllByRole("tab");
        const outputSchema = tabList[1];
        expect(outputSchema).toBeInTheDocument();

        await userEvent.click(outputSchema);
        expect(screen.getByText("Predictions table")).toBeInTheDocument();
    });
});
