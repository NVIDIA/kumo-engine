// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { createQueryClient } from "tests/testUtil";

import { jobStatus } from "utils/Constants";

import ModelPlanner, { ModelPlannerProps } from "../ModelPlanner/ModelPlanner";

const trainId = "trainingjob-4a6b21215a6b42f3854c2abeabe75c26";
const selectedOption = {
    "value": "2",
    "label": "Experiment 3 (BEST)"
};
const setOptions = vi.fn();
const setSelectedOption = vi.fn();
const defaultProps: ModelPlannerProps = {
    trainingJobId: "",
    bestExperimentIndex: 0,
    jobStatus: jobStatus.Done,
    setOptions,
    selectedOption: selectedOption,
    setSelectedOption
};

describe("ModelPlanner", () => {
    const queryClient = createQueryClient();
    const renderModelPlanner = async (props: ModelPlannerProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <ModelPlanner {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async () => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/training_jobs/${trainId}/model_plans`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    it("Should not render the ModelPlanner content when selectedOption is empty", async () => {
        await renderModelPlanner({
            ...defaultProps,
            trainingJobId: trainId,
            selectedOption: {}
        });

        const editor = screen.queryByRole("textbox");
        expect(editor?.textContent ?? "").not.toMatch(
            /training_table_generation:/i
        );
    });

    it("Should render the ModelPlanner with the expected content", async () => {
        await renderModelPlanner({
            ...defaultProps,
            trainingJobId: trainId,
            bestExperimentIndex: 2,
            jobStatus: jobStatus.Done
        });
        await checkApiCall();

        const editor = screen.getByRole("textbox");
        expect(editor.textContent).toMatch(/training_table_generation:/i);
        expect(editor.textContent).toMatch(/forecast_length: 1/i);
    });

    it("Should call setOptions and setSelectedOption after data fetch", async () => {
        await renderModelPlanner({
            ...defaultProps,
            trainingJobId: trainId,
            bestExperimentIndex: 2,
            jobStatus: jobStatus.Done
        });
        await checkApiCall();

        const editor = screen.getByRole("textbox");
        expect(editor.textContent).toMatch(/training_table_generation:/i);

        await waitFor(() => {
            expect(setOptions).toHaveBeenCalled();
            expect(setSelectedOption).toHaveBeenCalled();
        });
    });

    it("Should set selected option to last experiment when bestExperimentIndex is out of range", async () => {
        await renderModelPlanner({
            ...defaultProps,
            trainingJobId: trainId,
            bestExperimentIndex: 999,
            jobStatus: jobStatus.Done
        });
        await checkApiCall();

        await waitFor(() => {
            expect(setSelectedOption).toHaveBeenCalled();
            const calls = setSelectedOption.mock.calls;
            expect(calls.length).toBeGreaterThan(0);
            const lastCall = calls[calls.length - 1][0];
            expect(lastCall.value).toBe("3");
            expect(lastCall.label).toBe("Experiment 4");
        });
    });
});
