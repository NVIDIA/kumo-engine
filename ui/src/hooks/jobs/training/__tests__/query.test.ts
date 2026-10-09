// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { renderWithQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { QueryType, TaskType } from "types/predictiveQueries";

import { jobStatus } from "utils/Constants";

import {
    useGetQueryWarnings,
    useQueryDefinition,
    useQueryMetaData,
    useQueryValidation
} from "../query";

const PQ_ID = "pq-123";
const TRAINING_JOB_ID = "trainingjob-123";
const GEN_TRAINTABLE_JOB_ID = "gen-traintable-job-123";
const PROJECT_ID = "project-test";

const projectState = vi.hoisted(() => ({
    projectId: "project-test" as string | null
}));

vi.mock("hooks/project/projects", () => ({
    useProjectId: () => projectState.projectId
}));

beforeEach(() => {
    projectState.projectId = PROJECT_ID;
});

describe("useQueryDefinition", () => {
    it.each([
        {
            desc: "queryId is null",
            queryId: null,
            queryYamlFromPage: undefined
        },
        {
            desc: "queryId provided with no queryYamlFromPage",
            queryId: PQ_ID,
            queryYamlFromPage: undefined
        },
        {
            desc: "queryId provided with queryYamlFromPage (disabled)",
            queryId: PQ_ID,
            queryYamlFromPage: "existing yaml"
        }
    ])(
        "Should return empty queryYaml when $desc",
        async ({ queryId, queryYamlFromPage }) => {
            const { result } = renderWithQueryClient(() =>
                useQueryDefinition(queryId, queryYamlFromPage)
            );

            await waitFor(() => expect(result.current.queryYaml).toBe(""));
            expect(result.current.setQueryYaml).toBeTypeOf("function");
        }
    );

    it("Should update queryYaml when setQueryYaml is called", () => {
        const { result } = renderWithQueryClient(() =>
            useQueryDefinition(null)
        );

        act(() => {
            result.current.setQueryYaml("PREDICT target FROM users");
        });

        expect(result.current.queryYaml).toBe("PREDICT target FROM users");
    });
});

describe("useQueryMetaData", () => {
    it("Should return default TaskType and QueryType when queryId is null", async () => {
        const { result } = renderWithQueryClient(() => useQueryMetaData(null));

        await waitFor(() => expect(result.current.isError).toBe(false));
        expect(result.current.taskType).toBe(TaskType.NONE);
        expect(result.current.queryType).toBe(QueryType.STATIC);
    });

    it("Should return taskType and queryType from metadata when queryId is provided", async () => {
        const { result } = renderWithQueryClient(() => useQueryMetaData(PQ_ID));

        await waitFor(() =>
            expect(result.current.taskType).toBe("binary_classification")
        );
        expect(result.current.queryType).toBe("temporal");
        expect(result.current.isError).toBe(false);
    });
});

describe("useGetQueryWarnings", () => {
    it.each([
        { desc: "id is null (query disabled)", id: null },
        {
            desc: "id is valid but warnings response is empty",
            id: TRAINING_JOB_ID
        }
    ])(
        "Should return null ctaMsgs and empty warnings when $desc",
        async ({ id }) => {
            const handleError = vi.fn();
            const { result } = renderWithQueryClient(() =>
                useGetQueryWarnings(id, handleError, jobStatus.Done)
            );

            await waitFor(() => {
                const [ctaMsgs, , warnings] = result.current;
                expect(ctaMsgs).toBeNull();
                expect(warnings).toEqual([]);
            });
            expect(handleError).not.toHaveBeenCalled();
        }
    );

    it("Should not throw when hideWarnings is called with null ctaMsgs", () => {
        const handleError = vi.fn();
        const { result } = renderWithQueryClient(() =>
            useGetQueryWarnings(null, handleError)
        );

        const [, hideWarnings] = result.current;
        expect(() => {
            act(() => hideWarnings());
        }).not.toThrow();
        const [ctaMsgs] = result.current;
        expect(ctaMsgs).toBeNull();
    });

    // Regression tests for kumo-ai/kumo#30119: the training-job warnings
    // route rejects every non-training-job id with a 404, so the request
    // must only ever be issued for a real `trainingjob-*` id.
    it("Should not request warnings for a gen-traintable id", async () => {
        const handleError = vi.fn();
        renderWithQueryClient(() =>
            useGetQueryWarnings(
                GEN_TRAINTABLE_JOB_ID,
                handleError,
                jobStatus.Done
            )
        );

        // Flush timers/microtasks so a (buggy) fetch would have fired.
        await act(async () => {
            await new Promise((resolve) => setTimeout(resolve, 0));
        });

        expect(mockAxiosInstance.post).not.toHaveBeenCalledWith(
            `v0/training_jobs/${GEN_TRAINTABLE_JOB_ID}/warnings`,
            undefined,
            expect.anything()
        );
        expect(handleError).not.toHaveBeenCalled();
    });

    it("Should request warnings for a training-job id", async () => {
        const handleError = vi.fn();
        renderWithQueryClient(() =>
            useGetQueryWarnings(TRAINING_JOB_ID, handleError, jobStatus.Done)
        );

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/training_jobs/${TRAINING_JOB_ID}/warnings`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: PROJECT_ID }
                }
            );
        });
        expect(handleError).not.toHaveBeenCalled();
    });
});

describe("useQueryValidation", () => {
    it("Should return empty errors and warnings on successful validation", async () => {
        const { result } = renderWithQueryClient(() => useQueryValidation());

        const args = {
            name: "test_pq",
            graph_id: "graph-123",
            query_yaml: "PREDICT target FROM users",
            query_dialect: "PQLANG_V2",
            model_planner_yaml: "",
            description: "",
            run_mode: null,
            graph_name: null,
            graph: { tables: {}, links: [], groups: [] } as never
        };

        let validationResult!: { error: unknown[]; warnings: unknown[] };
        await act(async () => {
            validationResult = await result.current.validateQuery(args);
        });

        expect(validationResult.error).toEqual([]);
        expect(validationResult.warnings).toEqual([]);
        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
            "v1/predictive_queries/validate",
            expect.any(Object),
            {
                headers: { "Content-Type": "application/json" },
                params: { project_id: PROJECT_ID }
            }
        );
    });
});
