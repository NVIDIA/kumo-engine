// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { TaskType } from "types/predictiveQueries";

import {
    binaryEntityOutput,
    multiclassEntityOutput,
    nullRegressionEntityOutput,
    regressionEntityOutput,
    staticLpEntityOutput
} from "../../__tests__/MockResponses/EntityOutputResponses";
import { staticLpHistoricalItems } from "../../__tests__/MockResponses/HistoricalItemsResponses";
import { staticLpTarget } from "../../__tests__/MockResponses/TargetResponses";
import Prediction, { PredictionDisplayProps } from "../Prediction";

const reg_query = `PREDICT COUNT(transaction.Quantity
WHERE transaction.Description STARTS WITH "CARD", 0, 90,days)
FOR EACH customer.CustomerID`;

const binary_query = `PREDICT COUNT(FINANCE_TRANS.*, 0, 30,days)=0
FOR EACH FINANCE_ACCOUNT.ACCOUNT_ID`;

const anchorTime = "Fri Sep 9, 2011, 1:46:00 PM";

const defaultProps: PredictionDisplayProps = {
    entityOutput: regressionEntityOutput,
    taskType: TaskType.REGRESSION,
    historicalItems: undefined,
    isDataLoading: false,
    query: reg_query,
    anchorTime: anchorTime,
    showGroundTruth: true
};

const staticLpProps: PredictionDisplayProps = {
    ...defaultProps,
    taskType: TaskType.STATIC_LINK_PREDICTION,
    entityOutput: {
        task_type: TaskType.STATIC_LINK_PREDICTION,
        prediction: staticLpEntityOutput,
        target: staticLpTarget
    },
    historicalItems: staticLpHistoricalItems
};

describe("Prediction", () => {
    const renderPrediction = async (
        props: PredictionDisplayProps = defaultProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <Prediction {...props} />
                </MemoryRouter>
            );
        });
    };

    const getSectionByLabel = (labelText: string) => {
        const label = screen.getByText(labelText);
        return label.closest("div")!;
    };

    const expectSectionToContain = (
        labelText: string,
        expectedText: string
    ) => {
        const section = getSectionByLabel(labelText);
        expect(within(section).getByText(expectedText)).toBeInTheDocument();
    };

    const expectTableRows = (tableId: string, expectedRows: string[][]) => {
        const table = document.getElementById(tableId);
        expect(table).toBeInTheDocument();
        const rows = within(table!).getAllByRole("row").slice(1);

        expectedRows.forEach(([label, value], idx) => {
            const cells = within(rows[idx]).getAllByRole("cell");
            expect(cells[0]).toHaveTextContent(label);
            expect(cells[1]).toHaveTextContent(value);
        });
    };

    it("Should render RegressionPrediction component for REGRESSION task type", async () => {
        await renderPrediction(defaultProps);

        expectSectionToContain("Output", "—");
        expectSectionToContain("Prediction", "137.75");
        expectSectionToContain("Anchor Time", anchorTime);
    });

    it("Should render LinkPrediction component for LINK_PREDICTION task type", async () => {
        const props = {
            ...defaultProps,
            taskType: TaskType.STATIC_LINK_PREDICTION,
            entityOutput: {
                task_type: TaskType.STATIC_LINK_PREDICTION,
                prediction: staticLpEntityOutput,
                target: staticLpTarget
            },
            historicalItems: staticLpHistoricalItems
        };

        await renderPrediction(props);

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

        expect(screen.getByText("Ground Truth (1 items)")).toBeInTheDocument();
        expect(
            screen.getByText(staticLpEntityOutput[0].id)
        ).toBeInTheDocument();

        expect(screen.getByText("Predictions (10 items)")).toBeInTheDocument();
        expect(
            screen.getAllByText(staticLpTarget[0].id).length
        ).toBeGreaterThan(0);
    });

    it("Should render BinaryClassificationPrediction component for BINARY_CLASSIFICATION task type", async () => {
        const props = {
            ...defaultProps,
            taskType: TaskType.BINARY_CLASSIFICATION,
            entityOutput: binaryEntityOutput,
            query: binary_query
        };

        await renderPrediction(props);

        expectSectionToContain("Prediction", "0.00");
        expectSectionToContain("Ground Truth", "0.00");
        expectSectionToContain("Anchor Time", anchorTime);
    });

    it("Should render MultiClassificationPrediction component for MULTICLASS_CLASSIFICATION task type", async () => {
        const props = {
            ...defaultProps,
            taskType: TaskType.MULTICLASS_CLASSIFICATION,
            entityOutput: multiclassEntityOutput
        };

        await renderPrediction(props);

        expect(screen.getByText("Predictions")).toBeInTheDocument();
        expect(screen.getByText("Ground Truth")).toBeInTheDocument();

        const tables = screen.getAllByRole("table");
        expect(tables.length).toBe(2);

        const headers = screen.getAllByTestId("table-header");
        expect(headers.length).toBe(2);
        headers.forEach((header) => {
            expect(header).toHaveTextContent("Label");
            expect(header).toHaveTextContent("Value");
        });

        expectTableRows("prediction-table", [
            ["SIPO", "0.99"],
            ["UVER", "0.01"],
            ["POJISTNE", "0.00"],
            ["LEASING", "0.00"]
        ]);

        expectTableRows("ground-truth-table", [
            ["SIPO", "1"],
            ["UVER", "0"],
            ["POJISTNE", "0"],
            ["LEASING", "0"]
        ]);
    });

    it("Should render NoDataAlert when both prediction and target are null", async () => {
        const props = {
            ...defaultProps,
            entityOutput: nullRegressionEntityOutput
        };

        await renderPrediction(props);

        const alertDescription = document.querySelector(
            '[data-slot="alert-description"]'
        );

        expect(alertDescription).toHaveTextContent(
            "No prediction data or explainability scores are available for this entity at the moment. If this issue persists across all entities, please consider retraining the model or contact your support team at Kumo for further assistance."
        );
    });

    it("Should render Skeleton loader when data is loading", async () => {
        const props = {
            ...defaultProps,
            isDataLoading: true
        };

        await renderPrediction(props);

        ["Prediction", "Ground Truth"].forEach((label) => {
            const section = getSectionByLabel(label);
            const skeleton = section.querySelector('[data-slot="skeleton"]');
            expect(skeleton).toBeInTheDocument();
        });
    });

    it("Should pass selectedColumns to LinkPrediction and display correct checkboxes", async () => {
        const selectedColumns = ["customer_id", "FN", "Active"];
        const setSelectedColumns = vi.fn();

        await renderPrediction({
            ...staticLpProps,
            showGroundTruth: true,
            selectedColumns,
            setSelectedColumns
        });

        expect(screen.getByTestId("chip-customer_id")).toBeInTheDocument();
        expect(screen.getByTestId("chip-FN")).toBeInTheDocument();
        expect(screen.getByTestId("chip-Active")).toBeInTheDocument();
        expect(screen.queryByTestId("chip-age")).not.toBeInTheDocument();
    });

    it("Should call setSelectedColumns when checkbox is toggled in LinkPrediction", async () => {
        const selectedColumns = ["customer_id"];
        const setSelectedColumns = vi.fn();

        await renderPrediction({
            ...staticLpProps,
            showGroundTruth: true,
            selectedColumns,
            setSelectedColumns
        });

        expect(screen.queryByTestId("chip-FN")).not.toBeInTheDocument();

        await userEvent.click(screen.getByText("Add column"));
        await userEvent.click(await screen.findByText("FN"));

        expect(setSelectedColumns).toHaveBeenCalledWith(
            expect.arrayContaining(["customer_id", "FN"])
        );
    });
});
