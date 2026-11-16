// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from "@testing-library/react";
import { renderWithQueryClient } from "tests/testUtil";
import { describe, expect, it, vi } from "vitest";

import { jobStatus } from "utils/Constants";

import {
    useCancelTrainingJob,
    useCreateTrainingJob,
    useTrainingJobWorkflow
} from "../trainingJob";

const TRAINING_JOB_ID = "training-job-123";

describe("useCreateTrainingJob", () => {
    it("Should return jobId and no error on successful creation", async () => {
        const triggerAlert = vi.fn();
        const { result } = renderWithQueryClient(() =>
            useCreateTrainingJob(triggerAlert)
        );

        const args = {
            name: "test-job",
            graph_id: "",
            query_yaml: "PREDICT target FROM users",
            query_dialect: "PQLANG_V2",
            model_planner_yaml: "",
            description: "",
            run_mode: null,
            graph_name: null,
            graph: { tables: {}, links: [], groups: [] }
        } as never;

        let response!: { jobId: unknown; error: string };
        await act(async () => {
            response = await result.current.createTrainingJob(args, {});
        });

        expect(response.jobId).not.toBeNull();
        expect(response.error).toBe("");
        expect(triggerAlert).not.toHaveBeenCalled();
    });
});

describe("useCancelTrainingJob", () => {
    it("Should call onSuccess after cancelTraining is called", async () => {
        const onSuccess = vi.fn();
        const handleError = vi.fn();
        const { result } = renderWithQueryClient(() =>
            useCancelTrainingJob({
                jobId: TRAINING_JOB_ID,
                onSuccess,
                handleError
            })
        );

        expect(result.current.cancelTraining).toBeInstanceOf(Function);
        act(() => {
            result.current.cancelTraining({ id: TRAINING_JOB_ID });
        });

        await waitFor(() => expect(onSuccess).toHaveBeenCalled());
        expect(handleError).not.toHaveBeenCalled();
    });
});

describe("useTrainingJobWorkflow", () => {
    it("Should populate queryInfo and call setStatus on success", async () => {
        const setPageLoading = vi.fn();
        const handleError = vi.fn();
        const setStatus = vi.fn();

        const { result } = renderWithQueryClient(() =>
            useTrainingJobWorkflow({
                id: TRAINING_JOB_ID,
                setStatus,
                handleError,
                setPageLoading
            })
        );

        await waitFor(() => expect(result.current.isLoading).toBe(false));
        expect(result.current.queryInfo).not.toBeNull();
        expect(result.current.isError).toBe(false);
        expect(result.current.workflowResponse).toBeDefined();
        expect(result.current.refetch).toBeInstanceOf(Function);
        expect(setStatus).toHaveBeenCalledWith(jobStatus.Done);
        expect(setPageLoading).toHaveBeenCalledWith(false);
        expect(handleError).not.toHaveBeenCalled();
    });
});
