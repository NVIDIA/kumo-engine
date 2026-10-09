// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EntityOutput, HistoricalItems } from "gen/v0/types";
import { createQueryClient } from "tests/testUtil";

import {
    ClassificationEntityOutput,
    NodeData,
    RegressionEntityOutput
} from "types/explorer";
import { TaskType } from "types/predictiveQueries";

import Prediction, { PredictionDisplayProps } from "../Prediction/Prediction";

import {
    binaryEntityOutput,
    multilabelEntityOutput,
    regressionEntityOutput,
    staticLpEntityOutput
} from "./MockResponses/EntityOutputResponses";
import { staticLpHistoricalItems } from "./MockResponses/HistoricalItemsResponses";
import { staticLpTarget } from "./MockResponses/TargetResponses";

const defaultProps: PredictionDisplayProps = {
    entityOutput: {} as EntityOutput,
    taskType: TaskType.NONE,
    historicalItems: {} as HistoricalItems,
    isDataLoading: false,
    query: "",
    anchorTime: "",
    showGroundTruth: false
};

const staticLpProps: PredictionDisplayProps = {
    ...defaultProps,
    entityOutput: {
        task_type: TaskType.STATIC_LINK_PREDICTION,
        prediction: staticLpEntityOutput as NodeData[],
        target: staticLpTarget as NodeData[]
    } as EntityOutput,
    taskType: TaskType.STATIC_LINK_PREDICTION,
    historicalItems: staticLpHistoricalItems as HistoricalItems
};

const regressionProps: PredictionDisplayProps = {
    ...defaultProps,
    entityOutput: regressionEntityOutput as RegressionEntityOutput,
    taskType: TaskType.REGRESSION,
    historicalItems: {} as HistoricalItems,
    query: "PREDICT customers_hnm_downsampled.age\nFOR EACH customers_hnm_downsampled.customer_id",
    showGroundTruth: true
};

const binaryClassificationProps: PredictionDisplayProps = {
    ...defaultProps,
    entityOutput: binaryEntityOutput as RegressionEntityOutput,
    taskType: TaskType.BINARY_CLASSIFICATION,
    historicalItems: {} as HistoricalItems,
    query: "PREDICT COUNT(FINANCE_TRANS.*, 0, 30, days) = 0 \nFOR EACH FINANCE_ACCOUNT.ACCOUNT_ID",
    showGroundTruth: true
};

const multilabelProps: PredictionDisplayProps = {
    ...defaultProps,
    entityOutput: multilabelEntityOutput as ClassificationEntityOutput,
    taskType: TaskType.MULTILABEL_CLASSIFICATION,
    historicalItems: {} as HistoricalItems,
    showGroundTruth: true
};

describe("Prediction", () => {
    const queryClient = createQueryClient();
    const renderPrediction = async (props: PredictionDisplayProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <Prediction {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const expectSkeletonCount = (count: number) => {
        const skeleton = document.querySelectorAll('[data-slot="skeleton"]');
        expect(skeleton.length).toBe(count);
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render `Prediction` when all the props are set empty", async () => {
        await renderPrediction(defaultProps);

        expect(screen.queryByText("Prediction")).not.toBeInTheDocument();
    });

    it.each([
        {
            taskType: TaskType.STATIC_LINK_PREDICTION,
            name: "Static Link Prediction"
        },
        {
            taskType: TaskType.TEMPORAL_LINK_PREDICTION,
            name: "Temporal Link Prediction"
        }
    ])(
        "Should render `Prediction` when the taskType is `$name`",
        async ({ taskType }) => {
            await renderPrediction({
                ...staticLpProps,
                taskType,
                entityOutput: {
                    ...staticLpProps.entityOutput,
                    task_type: taskType
                } as EntityOutput,
                showGroundTruth: true
            });

            expect(screen.getByText("Columns")).toBeInTheDocument();
            // customer_id and FN are the two initial selected columns — shown as chips
            expect(screen.getByTestId("chip-customer_id")).toBeInTheDocument();
            expect(screen.getByTestId("chip-FN")).toBeInTheDocument();
            // Active is not selected initially — not a chip
            expect(screen.queryByTestId("chip-Active")).not.toBeInTheDocument();

            expect(
                screen.getByText("Historical Items (1 items)")
            ).toBeInTheDocument();
            expect(
                screen.getByText(staticLpHistoricalItems.nodes[0].id)
            ).toBeInTheDocument();

            expect(
                screen.getByText("Ground Truth (1 items)")
            ).toBeInTheDocument();
            expect(
                screen.getByText(staticLpEntityOutput[0].id)
            ).toBeInTheDocument();

            expect(
                screen.getByText("Predictions (10 items)")
            ).toBeInTheDocument();
            expect(
                screen.getAllByText(staticLpTarget[0].id).length
            ).toBeGreaterThan(0);
        }
    );

    it("Should not display `Ground Truth` field when `showGroundTruth` is set false", async () => {
        await renderPrediction(staticLpProps);

        expect(screen.queryByText("Ground Truth")).not.toBeInTheDocument();
    });

    it("Should display `skeleton` when the `isDataLoading` is set true", async () => {
        await renderPrediction({
            ...staticLpProps,
            isDataLoading: true,
            showGroundTruth: true
        });

        expectSkeletonCount(3);
    });

    it("Should filter the Predictions based on Search input", async () => {
        await renderPrediction({ ...staticLpProps, showGroundTruth: true });

        expect(screen.getByText("Predictions (10 items)")).toBeInTheDocument();

        const searchInputs = screen.getAllByPlaceholderText("Search");
        const predictionsSearchInput = searchInputs[2];
        const inputToSearch = staticLpEntityOutput[7].id;
        const otherPredictionId = staticLpEntityOutput[4].id;

        expect(screen.getByText(otherPredictionId)).toBeInTheDocument();

        await userEvent.click(predictionsSearchInput);
        await userEvent.type(predictionsSearchInput, inputToSearch);

        expect(screen.getByText(inputToSearch)).toBeInTheDocument();
        expect(screen.queryByText(otherPredictionId)).not.toBeInTheDocument();

        await userEvent.clear(predictionsSearchInput);
        expect(screen.queryByText(otherPredictionId)).toBeInTheDocument();
    });

    it("Should display additional content on the table when the checkbox is checked", async () => {
        await renderPrediction({ ...staticLpProps, showGroundTruth: true });

        expect(screen.getByText("Columns")).toBeInTheDocument();

        expect(screen.queryByTestId("chip-Active")).not.toBeInTheDocument();
        await userEvent.click(screen.getByText("Add column"));
        await userEvent.click(await screen.findByText("Active"));

        expect(screen.getByTestId("chip-Active")).toBeInTheDocument();

        const columnHeaders = screen.getAllByRole("columnheader");
        const activeColumnExists = columnHeaders.some((header) =>
            header.textContent?.includes("Active")
        );
        expect(activeColumnExists).toBe(true);
    });

    it("Should render `Prediction` when the taskType is `Regression`", async () => {
        await renderPrediction({ ...regressionProps });

        const outputQuery = "customers_hnm_downsampled.age";
        expect(screen.getByText("Output")).toBeInTheDocument();
        expect(screen.getByText(outputQuery));

        expect(screen.getByText("Prediction")).toBeInTheDocument();
        expect(
            screen.getByText(regressionEntityOutput.prediction.toFixed(2))
        ).toBeInTheDocument();

        expect(screen.getByText("Ground Truth")).toBeInTheDocument();
        expect(
            screen.getByText(regressionEntityOutput.target.toFixed(2))
        ).toBeInTheDocument();
    });

    it("Should display `Anchor Time` when the taskType is `Regression`", async () => {
        const anchorTime = "Thu Mar 6, 2025";
        await renderPrediction({
            ...regressionProps,
            isDataLoading: true,
            anchorTime
        });

        expectSkeletonCount(3);

        expect(screen.getByText("Anchor Time")).toBeInTheDocument();
        expect(screen.getByText(anchorTime)).toBeInTheDocument();
    });

    it("Should not display Ground Truth when the `showGroundTruth` is set false for the taskType is `Regression`", async () => {
        await renderPrediction({
            ...regressionProps,
            isDataLoading: true,
            showGroundTruth: false
        });

        expect(screen.getByText("Prediction")).toBeInTheDocument();
        expect(screen.queryByText("Ground Truth")).not.toBeInTheDocument();
        expectSkeletonCount(2);
    });

    it("Should not display anchor time and output field when the props are set empty for the taskType `Regression`", async () => {
        const anchorTime = "Thu Mar 6, 2025";
        await renderPrediction({
            ...regressionProps,
            query: "",
            anchorTime: ""
        });

        expect(screen.getByText("Output")).toBeInTheDocument();
        expect(screen.getByText("—")).toBeInTheDocument();

        expect(screen.getByText("Anchor Time")).toBeInTheDocument();
        expect(screen.queryByText(anchorTime)).not.toBeInTheDocument();
    });

    it("Should display empty state when `entityOuput` is set null for the taskType is `Regression`", async () => {
        await renderPrediction({
            ...regressionProps,
            entityOutput: null as unknown as RegressionEntityOutput
        });

        expect(screen.queryByText("Prediction")).not.toBeInTheDocument();
    });

    it("Should render `Prediction` when the taskType is `Binary Classification`", async () => {
        const anchorTime = "Sun Nov 1, 1998";
        await renderPrediction({ ...binaryClassificationProps, anchorTime });

        const outputQuery = "COUNT(FINANCE_TRANS.*, 0, 30, days) = 0";
        expect(screen.getByText("Output")).toBeInTheDocument();
        expect(screen.getByText(outputQuery));

        expect(screen.getByText("Prediction")).toBeInTheDocument();
        expect(screen.getByText("Ground Truth")).toBeInTheDocument();
        expect(
            screen.getAllByText(binaryEntityOutput.prediction.toFixed(2)).length
        ).toBe(2);

        expect(screen.getByText("Anchor Time")).toBeInTheDocument();
        expect(screen.getByText(anchorTime)).toBeInTheDocument();
    });

    it("Should display `skeleton` when the `isDataLoading` is set true for the taskType `Binary Classification`", async () => {
        const anchorTime = "Sun Nov 1, 1998";
        await renderPrediction({
            ...binaryClassificationProps,
            isDataLoading: true,
            anchorTime
        });

        expectSkeletonCount(3);

        expect(screen.getByText("Anchor Time")).toBeInTheDocument();
        expect(screen.getByText(anchorTime)).toBeInTheDocument();
    });

    it("Should not display Ground Truth when the `showGroundTruth` is set false for the taskType is `Binary Classification`", async () => {
        await renderPrediction({
            ...binaryClassificationProps,
            isDataLoading: true,
            showGroundTruth: false
        });

        expect(screen.getByText("Prediction")).toBeInTheDocument();
        expect(screen.queryByText("Ground Truth")).not.toBeInTheDocument();
        expectSkeletonCount(2);
    });

    it("Should not display anchor time and output field when the props are set empty for the taskType `Binary Classification`", async () => {
        const anchorTime = "Sun Nov 1, 1998";
        await renderPrediction({
            ...binaryClassificationProps,
            query: "",
            anchorTime: ""
        });

        expect(screen.getByText("Output")).toBeInTheDocument();
        expect(screen.getByText("—")).toBeInTheDocument();

        expect(screen.getByText("Anchor Time")).toBeInTheDocument();
        expect(screen.queryByText(anchorTime)).not.toBeInTheDocument();
    });

    it("Should display empty state when `entityOuput` is set null for the taskType is `Binary Classification`", async () => {
        await renderPrediction({
            ...binaryClassificationProps,
            entityOutput: null as unknown as RegressionEntityOutput
        });

        expect(screen.queryByText("Prediction")).not.toBeInTheDocument();
    });

    it.each([
        {
            taskType: TaskType.MULTILABEL_CLASSIFICATION,
            name: "Multilabel Classification"
        },
        {
            taskType: TaskType.MULTICLASS_CLASSIFICATION,
            name: "Multiclass Classification"
        }
    ])(
        "Should render `Prediction` when the taskType is `$name`",
        async ({ taskType }) => {
            await renderPrediction({ ...multilabelProps, taskType });

            expect(screen.getByText("Predictions")).toBeInTheDocument();
            expect(screen.getByText("Ground Truth")).toBeInTheDocument();

            const table = screen.getAllByRole("table");
            expect(table.length).toBe(2);

            const tableHeader = screen.getAllByTestId("table-header");
            expect(tableHeader[0]).toHaveTextContent("LabelValue");
            expect(tableHeader[1]).toHaveTextContent("LabelValue");

            const cells = screen.getAllByRole("cell");
            expect(cells[1]).toHaveTextContent(
                multilabelEntityOutput.prediction[1].toFixed(2)
            );
            expect(cells[3]).toHaveTextContent(
                multilabelEntityOutput.prediction[2].toFixed(2)
            );
            expect(cells[5]).toHaveTextContent(
                multilabelEntityOutput.target[1].toFixed(2)
            );
            expect(cells[7]).toHaveTextContent(
                multilabelEntityOutput.target[2].toFixed(2)
            );
        }
    );

    it("Should display `skeleton` with Ground Truth when the `isDataLoading` is set true for the taskType `Multilabel Classification`", async () => {
        await renderPrediction({
            ...multilabelProps,
            isDataLoading: true
        });

        expectSkeletonCount(20);
    });

    it("Should display `skeleton` without Ground Truth when `showGroundTruth` is false for the taskType `Multilabel Classification`", async () => {
        await renderPrediction({
            ...multilabelProps,
            isDataLoading: true,
            showGroundTruth: false
        });

        expect(screen.getByText("Predictions")).toBeInTheDocument();
        expect(screen.queryByText("Ground Truth")).not.toBeInTheDocument();
        expectSkeletonCount(10);
    });

    it("Should display empty state when `entityOuput` is set null for the taskType is `Multilabel Classification`", async () => {
        await renderPrediction({
            ...multilabelProps,
            entityOutput: null as unknown as RegressionEntityOutput
        });

        expect(screen.queryByText("Prediction")).not.toBeInTheDocument();
    });

    it("Should display `Something went wrong` when invalid output is set under `entityOutput`", async () => {
        await renderPrediction({
            ...binaryClassificationProps,
            entityOutput:
                multilabelEntityOutput as unknown as RegressionEntityOutput
        });

        expect(
            screen.getByText(/Something went wrong in/i)
        ).toBeInTheDocument();
        expect(
            screen.getByText(/BinaryClassificationPrediction/i)
        ).toBeInTheDocument();
        expect(
            screen.getByText(/Please contact Kumo support/i)
        ).toBeInTheDocument();
    });
});
