// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from "@testing-library/react";
import { graphInfo } from "tests/fixtures/MockGraphInformationPanel";
import { mockAxiosInstance } from "tests/mockAxios";
import { createMockResponse, renderWithQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { JSGraph } from "types/graphs";

import { jobStatus } from "utils/Constants";

import {
    modelPlanGraphDefinition,
    useDefaultModelPlan,
    useGetModelPlans
} from "../modelPlan";

const TRAINING_JOB_ID = "training-job-123";
const PROJECT_ID = "project-test";
const graphGroups = graphInfo.groups ?? [];
const graphWithPresentationFields: JSGraph = {
    ...graphInfo,
    isActive: true,
    groups: [
        ...graphGroups.map((group) => ({
            ...group,
            state: "saved" as const,
            updated: true
        })),
        {
            columns: [graphGroups[0]!.columns[0]!],
            state: "new",
            updated: true
        }
    ]
};

const projectState = vi.hoisted(() => ({
    projectId: "project-test" as string | null
}));

vi.mock("hooks/project/projects", () => ({
    useProjectId: () => projectState.projectId
}));

beforeEach(() => {
    projectState.projectId = PROJECT_ID;
});

describe("useGetModelPlans", () => {
    it.each([
        {
            desc: "no trainingJobId provided",
            trainingJobId: "",
            expectedPlans: 0,
            expectedOptions: 0
        },
        {
            desc: "trainingJobId provided",
            trainingJobId: TRAINING_JOB_ID,
            expectedPlans: 4,
            expectedOptions: 0
        }
    ])(
        "Should return $expectedPlans plans and $expectedOptions options when $desc",
        async ({ trainingJobId, expectedPlans, expectedOptions }) => {
            const { result } = renderWithQueryClient(() =>
                useGetModelPlans(trainingJobId, jobStatus.Done, 0)
            );

            await waitFor(() =>
                expect(result.current.plans.length).toBe(expectedPlans)
            );
            expect(result.current.options.length).toBe(expectedOptions);
        }
    );
});

describe("useDefaultModelPlan", () => {
    it("projects a UI graph to the strict public graph definition", () => {
        const definition = modelPlanGraphDefinition(
            graphWithPresentationFields
        );

        expect(definition).toEqual({
            tables: graphInfo.tables,
            col_groups: graphGroups.map(({ columns }) => ({ columns }))
        });
        expect(Object.keys(definition).sort()).toEqual([
            "col_groups",
            "tables"
        ]);
    });

    it("Should return null modelPlanYaml and not loading on mount", () => {
        const triggerError = vi.fn();
        const { result } = renderWithQueryClient(() =>
            useDefaultModelPlan(null, null, null, triggerError)
        );

        expect(result.current.isLoading).toBe(false);
        expect(result.current.modelPlanYaml).toBeNull();
        expect(typeof result.current.fetchModelPlan).toBe("function");
    });

    it("Should not trigger mutation when graph and queryText are null", () => {
        const triggerError = vi.fn();
        const { result } = renderWithQueryClient(() =>
            useDefaultModelPlan("normal", null, null, triggerError)
        );

        act(() => {
            result.current.fetchModelPlan();
        });

        expect(result.current.modelPlanYaml).toBeNull();
        expect(triggerError).not.toHaveBeenCalled();
    });

    it("keeps the current result when an older request finishes last", async () => {
        const triggerError = vi.fn();
        const stale = Promise.withResolvers<unknown>();
        const current = Promise.withResolvers<unknown>();
        mockAxiosInstance.post
            .mockImplementationOnce(
                () => stale.promise as ReturnType<typeof mockAxiosInstance.post>
            )
            .mockImplementationOnce(
                () =>
                    current.promise as ReturnType<typeof mockAxiosInstance.post>
            );
        let queryText = "PREDICT old_target FROM users";

        const { result, rerender } = renderWithQueryClient(() =>
            useDefaultModelPlan(
                "normal",
                graphWithPresentationFields,
                queryText,
                triggerError
            )
        );

        act(() => void result.current.fetchModelPlan());
        queryText = "PREDICT current_target FROM users";
        rerender();
        act(() => void result.current.fetchModelPlan());
        await act(async () => {
            current.resolve(
                createMockResponse({
                    suggested_model_plan: { task_type: "regression" },
                    suggested_model_plan_yaml: "current"
                })
            );
        });

        await waitFor(() => expect(result.current.taskType).toBe("regression"));
        expect(result.current.isLoading).toBe(false);
        expect(triggerError).not.toHaveBeenCalled();
        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
            "v1/predictive_queries/model_plan",
            {
                graph_definition: {
                    tables: graphInfo.tables,
                    col_groups: graphGroups.map(({ columns }) => ({
                        columns
                    }))
                },
                graph_id: "",
                query_string: queryText,
                run_mode: "normal"
            },
            {
                headers: { "Content-Type": "application/json" },
                params: {
                    dialect: "PQLANG_V2",
                    project_id: PROJECT_ID
                }
            }
        );

        await act(async () => {
            stale.resolve(
                createMockResponse({
                    suggested_model_plan: {
                        task_type: "binary_classification"
                    },
                    suggested_model_plan_yaml: "stale"
                })
            );
        });

        expect(result.current.taskType).toBe("regression");
        expect(result.current.modelPlanYaml).toBe("current");
    });
});
