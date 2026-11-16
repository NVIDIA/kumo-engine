// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TrainingWorkflowResponse } from "gen/v0/types";
import { mockAxiosInstance } from "tests/mockAxios";
import {
    verifyButtonExists,
    verifyElementExists,
    verifyTextExists
} from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { setModelPlanTestResponseType } from "components/Jobs/Training/__tests__/MockFunctions/ModelPlan.mock";
import { ModelPlan } from "components/PredictiveQueries/Overview/__tests__/MockResponses/ModelPlan";

import { JobSource } from "types/jobs";
import { PredictiveQueryInfo } from "types/predictiveQueries";

import { parseTrainingWorkflowResponse } from "utils/Converters/predictiveQueriesConverter";

import mockWorkflowResponse from "../../../Jobs/Training/__tests__/MockResponses/TrainingWorkflow.test.json";
import PQTrainingOverview, {
    PQTrainingOverviewProps
} from "../Overview/PQTrainingOverview";

const id = "trainingjob-00053000000000000000000000000000";
const expectedWorkflow = { ...mockWorkflowResponse };
expectedWorkflow.training_job.id = id;
const queryInfo = parseTrainingWorkflowResponse(
    expectedWorkflow as unknown as TrainingWorkflowResponse
);
queryInfo.modelPlannerYaml = ModelPlan;

const defaultProps: PQTrainingOverviewProps = {
    queryInfo: queryInfo as PredictiveQueryInfo,
    show: true,
    ctaMsgs: {}
};

describe("PQTrainingOverview", () => {
    const queryClient = createQueryClient();
    const renderPQTrainingOverview = async (props: PQTrainingOverviewProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <PQTrainingOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async () => {
        await waitFor(() => {
            const APIS_TO_CALL = [
                `v0/training_jobs/${id}/model_plans`,
                `v0/training_jobs/${id}/encoders`
            ];
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                APIS_TO_CALL[0],
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(APIS_TO_CALL[1]);
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
        setModelPlanTestResponseType(testResponseType.DEFAULT);
    });

    it("Should render all main sections and experiment selector", async () => {
        await renderPQTrainingOverview(defaultProps);
        await checkApiCall();

        verifyElementExists(
            screen.getByRole("heading", { name: "Experiments" })
        );
        verifyButtonExists("Experiment 1 (BEST)");
        verifyElementExists(screen.getByTestId("testid-experiment-dropdown"));
        await verifyTextExists("Training Settings");
        verifyElementExists(screen.getByRole("tab", { name: "Model plan" }));
    });

    it("Should update selected experiment when dropdown value changes", async () => {
        await renderPQTrainingOverview(defaultProps);
        await checkApiCall();

        const comboBox = await screen.findByRole("button", {
            name: "Experiment 1 (BEST)"
        });
        expect(comboBox).toHaveTextContent("Experiment 1 (BEST)");

        await userEvent.click(comboBox);

        const experiment2Option = await screen.findByText("Experiment 2");
        await userEvent.click(experiment2Option);

        expect(
            screen.getByRole("button", { name: "Experiment 2" })
        ).toHaveTextContent("Experiment 2");
    });

    it("Should render experiment performance chart with correct axis labels and values", async () => {
        await renderPQTrainingOverview(defaultProps);
        await checkApiCall();

        expect(
            document.getElementById("performance-chart")
        ).toBeInTheDocument();
        expect(screen.getAllByText("AUROC").length).toBeGreaterThan(0);

        const xAxis = screen.getAllByText("Experiments");
        expect(xAxis.length).toBeGreaterThan(0);

        const chartContainer = document.getElementById("performance-chart");
        ["0", "0.2", "0.4", "0.6", "0.8", "0.999"].forEach((value) => {
            expect(
                within(chartContainer!).getByText(value)
            ).toBeInTheDocument();
        });

        ["1", "2"].forEach((value) => {
            expect(screen.getAllByText(value).length).toBeGreaterThan(0);
        });
    });

    it("Should render model plan with correct details and tabs", async () => {
        await renderPQTrainingOverview(defaultProps);
        await checkApiCall();

        const editors = screen.getAllByRole("textbox");
        const allContent = editors.map((e) => e.textContent).join(" ");
        expect(allContent).toMatch(/training_table_generation:/i);
        expect(allContent).toMatch(/forecast_length: 1/i);

        const modelHeading = await screen.findByRole("heading", {
            name: "Model Plan"
        });
        expect(modelHeading).toBeInTheDocument();
        expect(screen.getByText("Column processing")).toBeInTheDocument();

        const modelPlanContainer = await waitFor(() =>
            document.getElementById("model_plan")
        );
        expect(modelPlanContainer).toBeInTheDocument();
        const planEditor = within(modelPlanContainer!).getByRole("textbox");
        expect(planEditor.textContent).toMatch(/training_table_generation:/i);
        expect(planEditor.textContent).toMatch(/training_job:/i);
    });

    it("Should display Column Processing tab with correct headers and non-empty table cells", async () => {
        await renderPQTrainingOverview(defaultProps);
        await checkApiCall();

        const columnProcessingTab = await screen.findByRole("tab", {
            name: "Column processing"
        });
        await userEvent.click(columnProcessingTab);

        expect(
            screen.getByText(
                /This section shows the Encoders used to process your raw data/i
            )
        ).toBeInTheDocument();

        expect(screen.getByTestId("table-header")).toBeInTheDocument();

        ["Connector", "Table", "Column", "Encoder"].forEach((header) => {
            expect(screen.getByText(header)).toBeInTheDocument();
        });

        const allCells = document.querySelectorAll('[data-slot="table-cell"]');
        expect(allCells.length).toBeGreaterThan(0);

        allCells.forEach((cell) => {
            expect(cell.textContent?.trim()).toBeTruthy();
        });
    });

    it("Should hide content when show is false", async () => {
        await renderPQTrainingOverview({
            ...defaultProps,
            show: false
        });

        expect(document.querySelector("div.hidden")).toBeInTheDocument();
    });

    it("Should display trimmed model plan when isV2 is false and jobSource is WEB", async () => {
        await renderPQTrainingOverview({
            ...defaultProps,
            queryInfo: {
                ...defaultProps.queryInfo,
                isV2: false,
                jobSource: JobSource.WEB,
                modelPlannerYaml: `
    training_table_generation:
      split: SomeSplit
    training_job:
      num_experiments: 2
      metrics:
        - auroc
    `
            }
        });

        const modelPlanContainer = await waitFor(() =>
            document.getElementById("model_plan")
        );
        expect(modelPlanContainer).toBeInTheDocument();
        const planEditor = within(modelPlanContainer!).getByRole("textbox");
        expect(planEditor.textContent).toContain("training_job:");
        expect(planEditor.textContent).not.toContain(
            "training_table_generation"
        );
    });

    it("Should display the `Model Plan` header when modelPlannerYaml is set as null", async () => {
        const customQueryInfo = {
            ...defaultProps.queryInfo,
            isV2: false,
            jobSource: JobSource.WEB,
            modelPlannerYaml: null
        };
        await renderPQTrainingOverview({
            ...defaultProps,
            queryInfo: customQueryInfo
        });
        const modelHeading = await screen.findByRole("heading", {
            name: "Model Plan"
        });
        expect(modelHeading).toBeInTheDocument();
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });
});
