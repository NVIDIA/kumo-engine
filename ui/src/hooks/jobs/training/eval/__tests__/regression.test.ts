// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { removeFromLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { renderWithQueryClient } from "tests/testUtil";
import { describe, expect, it, vi } from "vitest";

import { storageKeys } from "utils/Constants";

import { useRegressionHeatmap, useRegressionHistogram } from "../regression";

const TRAINING_JOB_ID = "training-job-123";

describe("useRegressionHeatmap", () => {
    it("does not fetch without a project", () => {
        removeFromLocalStorage(storageKeys.selectedProject);

        renderWithQueryClient(() =>
            useRegressionHeatmap({
                queryId: "query-123",
                handleError: vi.fn()
            })
        );

        expect(mockAxiosInstance.get).not.toHaveBeenCalled();
    });

    it.each([
        {
            desc: "no IDs provided",
            trainingJobId: undefined,
            queryId: "",
            dataIsUndefined: true
        },
        {
            desc: "trainingJobId is provided",
            trainingJobId: TRAINING_JOB_ID,
            queryId: "",
            dataIsUndefined: false
        },
        {
            desc: "trainingJobId overrides queryId",
            trainingJobId: TRAINING_JOB_ID,
            queryId: "query-123",
            dataIsUndefined: false
        }
    ])(
        "Should return correct data when $desc",
        async ({ trainingJobId, queryId, dataIsUndefined }) => {
            const { result } = renderWithQueryClient(() =>
                useRegressionHeatmap({
                    queryId,
                    trainingJobId,
                    handleError: vi.fn()
                })
            );

            await waitFor(() => expect(result.current.isLoading).toBe(false));
            if (dataIsUndefined) {
                expect(result.current.data).toBeUndefined();
            } else {
                expect(result.current.data).not.toBeUndefined();
            }
        }
    );
});

describe("useRegressionHistogram", () => {
    it("does not fetch without a project", () => {
        removeFromLocalStorage(storageKeys.selectedProject);

        renderWithQueryClient(() =>
            useRegressionHistogram({
                queryId: "query-123",
                handleError: vi.fn()
            })
        );

        expect(mockAxiosInstance.get).not.toHaveBeenCalled();
    });

    it.each([
        {
            desc: "no IDs provided",
            trainingJobId: undefined,
            queryId: "",
            dataIsUndefined: true
        },
        {
            desc: "trainingJobId is provided",
            trainingJobId: TRAINING_JOB_ID,
            queryId: "",
            dataIsUndefined: false
        },
        {
            desc: "trainingJobId overrides queryId",
            trainingJobId: TRAINING_JOB_ID,
            queryId: "query-123",
            dataIsUndefined: false
        }
    ])(
        "Should return correct data when $desc",
        async ({ trainingJobId, queryId, dataIsUndefined }) => {
            const { result } = renderWithQueryClient(() =>
                useRegressionHistogram({
                    queryId,
                    trainingJobId,
                    handleError: vi.fn()
                })
            );

            await waitFor(() => expect(result.current.isLoading).toBe(false));
            if (dataIsUndefined) {
                expect(result.current.data).toBeUndefined();
            } else {
                expect(result.current.data).not.toBeUndefined();
            }
        }
    );
});
