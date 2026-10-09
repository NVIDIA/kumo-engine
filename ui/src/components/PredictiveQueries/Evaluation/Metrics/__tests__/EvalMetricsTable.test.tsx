// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

import { PredictiveQueryInfo, TaskType } from "types/predictiveQueries";

import EvalMetricsTable, { EvalMetricsProps } from "../EvalMetricsTable";

import {
    eval_query_info_reg as eval_query_info_reg_raw,
    kumo_model_results
} from "./MockResponses/QueryInfo.json";

const eval_query_info_reg =
    eval_query_info_reg_raw as unknown as PredictiveQueryInfo;

const defaultProps: EvalMetricsProps = {
    queryInfo: {
        ...eval_query_info_reg,
        taskType: TaskType.REGRESSION
    },
    results: {
        test_metrics: kumo_model_results,
        val_metrics: kumo_model_results,
        train_metrics: kumo_model_results
    },
    tuneMetric: "MAE"
};

describe("EvalMetricsTable", () => {
    const renderEvalMetricsTable = async (props: EvalMetricsProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <EvalMetricsTable {...props} />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render EvalMetricsTable without crashing", async () => {
        await renderEvalMetricsTable(defaultProps);

        const table = screen.getByRole("table");
        expect(table).toBeInTheDocument();

        const rowsOriginal = screen.getAllByRole("row");
        expect(rowsOriginal.length).equal(4);

        await waitFor(() => {
            expect(screen.getByText("Metric")).toBeInTheDocument();
            expect(screen.getByText("Holdout")).toBeInTheDocument();
            expect(screen.getByText("Validation")).toBeInTheDocument();
            expect(screen.getByText("Training")).toBeInTheDocument();
        });

        await waitFor(() => {
            expect(screen.getByText("MAE")).toBeInTheDocument();
            expect(screen.getByText("MSE")).toBeInTheDocument();
            expect(screen.getByText("RMSE")).toBeInTheDocument();
        });
    });

    it("Should handle undefined results", async () => {
        await renderEvalMetricsTable({
            ...defaultProps,
            results: undefined
        });

        await waitFor(() => {
            expect(screen.getByText(/No data found/)).toBeInTheDocument();
        });
    });
});
