// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { QueryType } from "gen/v1/types";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { setValidateOutputSchemaResponseType } from "pages/NewJob/Prediction/__tests__/MockFunction/OutputSchema.mock";
import mockSchemaResponse from "../../../../pages/NewJob/Prediction/__tests__/MockResponse/OutputSchema.json";

import { TaskType } from "types/predictiveQueries";

import OutputTableSchema, {
    OutputTableSchemaProps
} from "../Schema/OutputTableSchema";

import { setPredictionTableSchemaTestResponseType } from "./MockFunctions/PredictionTableSchema.mock";
import { setSamplesForOutputTestResponseType } from "./MockFunctions/SampleForOutput.mock";
import mockSamplesResponse from "./MockResponses/SampleOutputSchema.json";

const trainingTableID = "gen-traintable-job-00046000000000000000000000000000";
const queryId = "pquery-00027000000000000000000000000000";
const taskType = TaskType.BINARY_CLASSIFICATION;
const queryType = QueryType.TEMPORAL;
const defaultProps: OutputTableSchemaProps = {
    queryId: "",
    taskType: undefined,
    queryType: undefined,
    trainingTableID: ""
};

describe("OutputTableSchema", () => {
    const queryClient = createQueryClient();
    const renderOutputTableSchema = async (props: OutputTableSchemaProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <OutputTableSchema {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async (
        taskType: TaskType,
        trainingTableID?: string
    ) => {
        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v1/batch_prediction_jobs/output_schema`,
                {
                    task_type: `${taskType}`,
                    query_type: "temporal"
                },
                {
                    headers: {
                        "Content-Type": "application/json"
                    }
                }
            );

            if (trainingTableID) {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    `v1/batch_prediction_jobs/samples_for_output_schema`,
                    {
                        train_table_job_id: `${trainingTableID}`,
                        task_type: `${taskType}`
                    },
                    {
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }
                );
            }
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    it("Should display `No data found.` message when the props are set empty/undefined", async () => {
        await renderOutputTableSchema(defaultProps);

        expect(screen.getByText("No data found.")).toBeInTheDocument();
    });

    it("Should display `No data found.` message when the props are set null/undefined", async () => {
        await renderOutputTableSchema({
            ...defaultProps,
            queryId: null as unknown as string,
            trainingTableID: null as unknown as string
        });

        expect(screen.getByText("No data found.")).toBeInTheDocument();
    });

    it("Should call `/prediction_tables_schema` api when only the queryId prop is set", async () => {
        await renderOutputTableSchema({
            ...defaultProps,
            queryId
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/batch_prediction_jobs/${queryId}/prediction_table_schema`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        expect(
            await screen.findByText("Predictions table")
        ).toBeInTheDocument();
        const predictionTableHeader = screen.getByTestId("table-header");
        expect(predictionTableHeader).toHaveTextContent("Column name");
        expect(predictionTableHeader).toHaveTextContent("Data type");
    });

    it("Should call `/output_schema` api when the taskType and queryType are set", async () => {
        await renderOutputTableSchema({
            ...defaultProps,
            taskType,
            queryType
        });

        await checkApiCall(taskType);

        expect(
            await screen.findByText("Predictions table")
        ).toBeInTheDocument();
        const predictionTableHeader = screen.getByTestId("table-header");
        expect(predictionTableHeader).toHaveTextContent("Column name");
        expect(predictionTableHeader).toHaveTextContent("Data type");
    });

    it("Should render `OutputTableSchema` with both `/output_schema` and `/sample_output_schema` API when the taskType, queryType and trainingJobId props are set", async () => {
        await renderOutputTableSchema({
            ...defaultProps,
            taskType,
            queryType,
            trainingTableID
        });

        await checkApiCall(taskType, trainingTableID);

        expect(
            await screen.findByText("Predictions table")
        ).toBeInTheDocument();
        const predictionTableHeader = screen.getByTestId("table-header");
        expect(predictionTableHeader).toHaveTextContent("Column name");
        expect(predictionTableHeader).toHaveTextContent("Data type");
    });

    it("Should display Output schema and Output samples", async () => {
        await renderOutputTableSchema({
            ...defaultProps,
            taskType,
            queryType,
            trainingTableID
        });

        await checkApiCall(taskType, trainingTableID);

        expect(
            await screen.findByText("Predictions table")
        ).toBeInTheDocument();

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

    it("Should display `No data found.` when the API `/output_schema` return error", async () => {
        setValidateOutputSchemaResponseType(testResponseType.ERROR);
        await renderOutputTableSchema({
            ...defaultProps,
            taskType: TaskType.REGRESSION,
            queryType
        });

        await checkApiCall(TaskType.REGRESSION);

        expect(await screen.findByText(`No data found.`)).toBeInTheDocument();

        setValidateOutputSchemaResponseType(testResponseType.ALL);
    });

    it("Should not display `Column samples` when the api `/sample_output_schema` return error", async () => {
        setSamplesForOutputTestResponseType(testResponseType.ERROR);
        await renderOutputTableSchema({
            ...defaultProps,
            taskType: TaskType.REGRESSION,
            queryType,
            trainingTableID
        });

        await checkApiCall(TaskType.REGRESSION, trainingTableID);

        expect(screen.queryByText("Column samples")).not.toBeInTheDocument();
        mockSchemaResponse.forEach((col) => {
            expect(screen.getByText(col.name)).toBeInTheDocument();
            expect(screen.getAllByText(col.dtype).length).toBeGreaterThan(0);
        });
    });

    it("Should display `No data found.` when the api `/prediction_table_schema` return error", async () => {
        setPredictionTableSchemaTestResponseType(testResponseType.ERROR);
        await renderOutputTableSchema({
            queryId,
            taskType,
            queryType
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/batch_prediction_jobs/${queryId}/prediction_table_schema`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        expect(await screen.findByText(`No data found.`)).toBeInTheDocument();
    });
});
