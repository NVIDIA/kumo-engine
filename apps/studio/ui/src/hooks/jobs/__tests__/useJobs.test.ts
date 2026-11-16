// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from "@testing-library/react";
import { JobType, ListJobsQuery } from "gen/v0/types";
import { describe, expect, it } from "vitest";

import { useJobsListTableData } from "../list/jobList";

describe("useJobsTableData", () => {
    const mockQuery: ListJobsQuery = {
        project_id: "project_00000000000000000000000000000001",
        limit: 10,
        page: 1,
        job_type: [JobType.TRAINING_JOB],
        status: [],
        sort_by: undefined,
        sort_desc: true,
        search_keyword: "",
        tags_filter: [],
        tags: {}
    };

    it("Should initialize with correct default state", () => {
        const { result } = renderHook(() =>
            useJobsListTableData(mockQuery, 100)
        );

        expect(result.current.sorting).toEqual([{ id: undefined, desc: true }]);
    });

    it("Should calculate correct skeleton rows for middle page", () => {
        const query = { ...mockQuery, page: 2, limit: 10 };
        const { result } = renderHook(() => useJobsListTableData(query, 100));

        expect(result.current.numSkeletonRow).toBe(10);
    });

    it("Should calculate correct skeleton rows for last page", () => {
        const query = { ...mockQuery, page: 9, limit: 10 };
        const { result } = renderHook(() => useJobsListTableData(query, 95));

        expect(result.current.numSkeletonRow).toBe(5);
    });

    it("Should show test_metric column for training jobs", () => {
        const query = { ...mockQuery, job_type: [JobType.TRAINING_JOB] };
        const { result } = renderHook(() => useJobsListTableData(query, 100));

        expect(result.current.columnsVisibility.test_metric).toBe(true);
    });

    it("Should hide test_metric column for prediction jobs", () => {
        const query = {
            ...mockQuery,
            job_type: [JobType.BATCH_PREDICTION_JOB]
        };
        const { result } = renderHook(() => useJobsListTableData(query, 100));

        expect(result.current.columnsVisibility.test_metric).toBe(false);
    });

    it("Should provide setSorting function", () => {
        const { result } = renderHook(() =>
            useJobsListTableData(mockQuery, 100)
        );

        expect(typeof result.current.setSorting).toBe("function");
    });

    it("Should handle empty totalNumJobs", () => {
        const query = { ...mockQuery, page: 0, limit: 10 };
        const { result } = renderHook(() => useJobsListTableData(query, 0));

        expect(result.current.numSkeletonRow).toBe(0);
    });

    it("Should update when query changes", () => {
        const { result, rerender } = renderHook(
            ({ query }) => useJobsListTableData(query, 100),
            { initialProps: { query: mockQuery } }
        );

        const initialSorting = result.current.sorting;

        const newQuery: ListJobsQuery = {
            ...mockQuery,
            sort_by: undefined,
            sort_desc: false
        };
        rerender({ query: newQuery });

        expect(result.current.sorting).not.toEqual(initialSorting);
    });
});
