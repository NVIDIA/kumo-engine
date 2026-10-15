// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryType } from "gen/v1/types";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import mockSchemaResponse from "../../../../pages/NewJob/Prediction/__tests__/MockResponse/OutputSchema.json";

import { TaskType } from "types/predictiveQueries";

import PredictionSchema, {
    PredictionSchemaProps
} from "../Schema/PredictionSchema";

import { setPredictionTableSchemaTestResponseType } from "./MockFunctions/PredictionTableSchema.mock";
import mockSamplesResponse from "./MockResponses/SampleOutputSchema.json";

const queryId = "pquery-00027000000000000000000000000000";
const graphSnapshotId =
    "graph-0005b000000000000000000000000000@00033000000000000000000000000000";
const trainingTableId = "gen-traintable-job-00046000000000000000000000000000";

const defaultProps: PredictionSchemaProps = {
    queryId,
    graphSnapshotId,
    taskType: "" as TaskType,
    queryType: "" as QueryType,
    trainingTableId
};

describe("PredictionSchema", () => {
    const queryClient = createQueryClient();
    const renderPredictionSchema = async (props: PredictionSchemaProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <PredictionSchema {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const outputTableSelect = async () => {
        const tableRole = screen.getByRole("tablist");
        expect(tableRole).toBeInTheDocument();

        const tabList = screen.getAllByRole("tab");
        const outputSchema = tabList[1];
        expect(outputSchema).toBeInTheDocument();

        await userEvent.click(outputSchema);
    };

    beforeEach(() => {
        vi.resetAllMocks();
        queryClient.clear();
    });

    it("Should render `PredictionSchema` when all props are set empty", async () => {
        await renderPredictionSchema({
            ...defaultProps,
            queryId: "",
            graphSnapshotId: "",
            trainingTableId: ""
        });

        await outputTableSelect();

        expect(screen.getByText("Output table")).toBeInTheDocument();
        expect(screen.getByText("No data found.")).toBeInTheDocument();
    });

    it("Should render `PredictionSchema` when all props are set as null", async () => {
        await renderPredictionSchema({
            ...defaultProps,
            queryId: null as unknown as string,
            graphSnapshotId: null as unknown as string,
            trainingTableId: null as unknown as string
        });

        await outputTableSelect();

        expect(screen.getByText("Output table")).toBeInTheDocument();
        expect(screen.getByText("No data found.")).toBeInTheDocument();
    });

    it("Should render `Graph schema`", async () => {
        await renderPredictionSchema({
            ...defaultProps,
            queryId: "",
            trainingTableId: ""
        });

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/snapshots/graph`,
                {
                    "graph_snapshot_id": graphSnapshotId,
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

        await waitFor(() => {
            const skeleton = document.querySelector('[data-slot="skeleton"]');
            expect(skeleton).not.toBeInTheDocument();
        });

        const graphSchemaTab = screen.getByRole("tab", {
            name: /Graph/i
        });
        expect(graphSchemaTab).toHaveAttribute("aria-selected", "true");

        expect(
            screen.getByText(/Press enter or space to select a node/i)
        ).toBeInTheDocument();
        expect(
            screen.getByText(/Press enter or space to select an edge/i)
        ).toBeInTheDocument();
    });

    it("Should display `Predictions Table` under the Output table schema", async () => {
        await renderPredictionSchema({
            ...defaultProps,
            taskType: TaskType.BINARY_CLASSIFICATION,
            queryType: QueryType.TEMPORAL
        });

        await outputTableSelect();

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/batch_prediction_jobs/${queryId}/prediction_table_schema`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
            `v1/batch_prediction_jobs/samples_for_output_schema`,
            {
                train_table_job_id:
                    "gen-traintable-job-00046000000000000000000000000000",
                task_type: `${TaskType.BINARY_CLASSIFICATION}`
            },
            {
                headers: {
                    "Content-Type": "application/json"
                }
            }
        );

        const tabList = screen.getAllByRole("tab");
        expect(tabList[1]).toHaveTextContent("Output table");
        expect(tabList[0]).toHaveTextContent("Graph");

        expect(screen.getByText("Predictions table")).toBeInTheDocument();
        const predictionTableHeader = screen.getByTestId("table-header");
        expect(predictionTableHeader).toHaveTextContent("Column name");
        expect(predictionTableHeader).toHaveTextContent("Data type");

        expect(screen.queryByText("Column samples")).toBeInTheDocument();
        mockSchemaResponse.forEach((col) => {
            expect(screen.getByText(col.name)).toBeInTheDocument();
            expect(screen.getAllByText(col.dtype).length).toBeGreaterThan(0);
        });

        mockSamplesResponse.forEach((col) => {
            const joined = col.sample_rows.join(", ");
            expect(screen.getByText(joined)).toBeInTheDocument();
        });
    });

    it("Should display `No data found.` when the api `/prediction_table_schema` return error", async () => {
        setPredictionTableSchemaTestResponseType(testResponseType.INVALID);
        await renderPredictionSchema({
            ...defaultProps,
            taskType: TaskType.BINARY_CLASSIFICATION,
            queryType: QueryType.TEMPORAL
        });

        await outputTableSelect();

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/batch_prediction_jobs/${queryId}/prediction_table_schema`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        expect(screen.getByText(`No data found.`)).toBeInTheDocument();
    });
});
