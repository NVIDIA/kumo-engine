// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import type { AxiosResponse } from "axios";
import { mockAxiosInstance } from "tests/mockAxios";
import { createMockResponse, createQueryClient } from "tests/testUtil";

import { jobStatus } from "utils/Constants";

import ModelPlanner, { ModelPlannerProps } from "../ModelPlanner/ModelPlanner";

const trainId = "trainingjob-00023000000000000000000000000000";
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

    const defaultGet = mockAxiosInstance.get.getMockImplementation();

    const mockModelPlans = (modelPlans: string[], trialIds: string[]) => {
        mockAxiosInstance.get.mockImplementation((url, data, options) =>
            url.endsWith("/model_plans")
                ? Promise.resolve(
                      createMockResponse({
                          model_plans: modelPlans,
                          advanced_options: [],
                          trial_ids: trialIds
                      }) as AxiosResponse<unknown>
                  )
                : defaultGet!(url, data, options)
        );
    };

    const lastCallArg = (mockFn: ReturnType<typeof vi.fn>) =>
        mockFn.mock.calls[mockFn.mock.calls.length - 1][0];

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    afterEach(() => {
        mockAxiosInstance.get.mockImplementation(defaultGet!);
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

    it("Should select the best plan by trial id instead of position", async () => {
        mockModelPlans(["plan-0", "plan-2", "plan-3"], ["0", "2", "3"]);

        await renderModelPlanner({
            ...defaultProps,
            trainingJobId: trainId,
            bestExperimentIndex: 1,
            bestTrialId: "3"
        });
        await checkApiCall();

        await waitFor(() => {
            expect(lastCallArg(setSelectedOption)).toEqual({
                value: "2",
                label: "Experiment 3 (BEST)"
            });
            expect(lastCallArg(setOptions)).toEqual([
                { value: "0", label: "Experiment 1" },
                { value: "1", label: "Experiment 2" },
                { value: "2", label: "Experiment 3 (BEST)" }
            ]);
        });
    });

    it("Should not mark any plan as best when the best trial has no plan", async () => {
        mockModelPlans(["plan-0", "plan-2"], ["0", "2"]);

        await renderModelPlanner({
            ...defaultProps,
            trainingJobId: trainId,
            bestExperimentIndex: 1,
            bestTrialId: "1"
        });
        await checkApiCall();

        await waitFor(() => {
            expect(lastCallArg(setSelectedOption)).toEqual({
                value: "1",
                label: "Experiment 2"
            });
            expect(lastCallArg(setOptions)).toEqual([
                { value: "0", label: "Experiment 1" },
                { value: "1", label: "Experiment 2" }
            ]);
        });
    });
});
