// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { removeFromLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { renderWithQueryClient } from "tests/testUtil";
import { describe, expect, it, vi } from "vitest";

import { TaskType } from "types/predictiveQueries";

import { jobStatus, storageKeys } from "utils/Constants";

import {
    useBpEntityCategories,
    useBpEntityData,
    useEntityCategories,
    useEntityData
} from "../explorer";

const TRAINING_JOB_ID = "training-job-123";
const BP_JOB_ID = "bp-job-123";

const assertCategoriesLoaded = (result: {
    current: { filteredCategories: object; filteredEntityIdCategories: object };
}) => {
    expect(
        Object.keys(result.current.filteredCategories).length
    ).toBeGreaterThan(0);
    expect(
        Object.keys(result.current.filteredEntityIdCategories).length
    ).toBeGreaterThan(0);
};

describe("useEntityCategories", () => {
    it("does not fetch without a project", () => {
        removeFromLocalStorage(storageKeys.selectedProject);

        renderWithQueryClient(() =>
            useEntityCategories("query-123", jobStatus.Done, vi.fn())
        );

        expect(mockAxiosInstance.get).not.toHaveBeenCalled();
    });

    it.each([
        {
            desc: "no IDs provided",
            queryId: "",
            status: jobStatus.Done,
            trainingJobId: undefined
        },
        {
            desc: "status is not Done",
            queryId: "",
            status: jobStatus.Running,
            trainingJobId: TRAINING_JOB_ID
        }
    ])(
        "Should return empty categories and isLoading false when $desc",
        ({ queryId, status, trainingJobId }) => {
            const { result } = renderWithQueryClient(() =>
                useEntityCategories(queryId, status, vi.fn(), trainingJobId)
            );

            expect(result.current.filteredCategories).toEqual({});
            expect(result.current.filteredEntityIdCategories).toEqual({});
            expect(result.current.isLoading).toBe(false);
        }
    );

    it("Should load categories when trainingJobId is provided and status is Done", async () => {
        const { result } = renderWithQueryClient(() =>
            useEntityCategories("", jobStatus.Done, vi.fn(), TRAINING_JOB_ID)
        );

        await waitFor(() => expect(result.current.isLoading).toBe(false));
        assertCategoriesLoaded(result);
    });
});

describe("useBpEntityCategories", () => {
    it.each([
        { desc: "jobId is empty", jobId: "", status: jobStatus.Done },
        {
            desc: "status is not Done",
            jobId: BP_JOB_ID,
            status: jobStatus.Running
        }
    ])(
        "Should return empty categories and isLoading true when $desc",
        ({ jobId, status }) => {
            const { result } = renderWithQueryClient(() =>
                useBpEntityCategories(jobId, status)
            );

            expect(result.current.filteredCategories).toEqual({});
            expect(result.current.filteredEntityIdCategories).toEqual({});
            expect(result.current.isLoading).toBe(true);
        }
    );

    it("Should load categories when jobId is provided and status is Done", async () => {
        const { result } = renderWithQueryClient(() =>
            useBpEntityCategories(BP_JOB_ID, jobStatus.Done)
        );

        await waitFor(() => expect(result.current.isLoading).toBe(false));
        assertCategoriesLoaded(result);
    });
});

describe("useEntityData", () => {
    it("does not fetch without a project", () => {
        removeFromLocalStorage(storageKeys.selectedProject);

        renderWithQueryClient(() =>
            useEntityData({
                queryId: "query-123",
                selectedEntity: "entity-1",
                taskType: TaskType.BINARY_CLASSIFICATION,
                status: jobStatus.Done
            })
        );

        expect(mockAxiosInstance.get).not.toHaveBeenCalled();
    });

    it("Should return undefined entityOutput and isLoading false when selectedEntity is empty", async () => {
        const { result } = renderWithQueryClient(() =>
            useEntityData({
                queryId: "pq-123",
                selectedEntity: "",
                taskType: TaskType.BINARY_CLASSIFICATION,
                status: jobStatus.Done,
                trainingJobId: TRAINING_JOB_ID
            })
        );

        await waitFor(() => expect(result.current.isLoading).toBe(false));
        expect(result.current.entityOutput).toBeUndefined();
        expect(result.current.isError).toBe(false);
    });
});

describe("useBpEntityData", () => {
    it.each([
        {
            desc: "selectedEntity is empty",
            selectedEntity: "",
            status: jobStatus.Done
        },
        {
            desc: "status is not Done",
            selectedEntity: "entity-1",
            status: jobStatus.Running
        }
    ])(
        "Should return null data and isLoading true when $desc",
        ({ selectedEntity, status }) => {
            const { result } = renderWithQueryClient(() =>
                useBpEntityData(
                    BP_JOB_ID,
                    selectedEntity,
                    TaskType.BINARY_CLASSIFICATION,
                    status
                )
            );

            expect(result.current.entityOutput).toBeNull();
            expect(result.current.subgraph).toBeNull();
            expect(result.current.historicalItems).toBeNull();
            expect(result.current.isLoading).toBe(true);
        }
    );
});
