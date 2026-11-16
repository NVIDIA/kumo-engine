// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { removeFromLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { renderWithQueryClient } from "tests/testUtil";
import { describe, expect, it, vi } from "vitest";

import { storageKeys } from "utils/Constants";

import { useGetEvalStats } from "../binaryClassification";

const TRAINING_JOB_ID = "training-job-123";
const QUERY_ID = "query-456";

describe("useGetEvalStats", () => {
    it("does not fetch evaluation stats without a project", () => {
        removeFromLocalStorage(storageKeys.selectedProject);

        renderWithQueryClient(() =>
            useGetEvalStats(undefined, QUERY_ID, vi.fn())
        );

        expect(mockAxiosInstance.get).not.toHaveBeenCalled();
    });

    it.each([
        {
            desc: "no IDs provided",
            trainingJobId: undefined,
            queryId: undefined,
            responseIsUndefined: true
        },
        {
            desc: "trainingJobId is provided",
            trainingJobId: TRAINING_JOB_ID,
            queryId: undefined,
            responseIsUndefined: false
        },
        {
            desc: "trainingJobId overrides queryId",
            trainingJobId: TRAINING_JOB_ID,
            queryId: QUERY_ID,
            responseIsUndefined: false
        }
    ])(
        "Should return correct evaluationResponse when $desc",
        async ({ trainingJobId, queryId, responseIsUndefined }) => {
            const handleError = vi.fn();
            const { result } = renderWithQueryClient(() =>
                useGetEvalStats(trainingJobId, queryId, handleError)
            );

            await waitFor(() =>
                expect(result.current.isDataLoading).toBe(false)
            );
            if (responseIsUndefined) {
                expect(result.current.evaluationResponse).toBeUndefined();
            } else {
                expect(result.current.evaluationResponse).not.toBeUndefined();
            }
            expect(handleError).not.toHaveBeenCalled();
        }
    );
});
