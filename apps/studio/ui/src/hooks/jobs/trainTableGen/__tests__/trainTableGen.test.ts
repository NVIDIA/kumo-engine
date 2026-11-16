// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook, waitFor } from "@testing-library/react";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { describe, expect, it, vi } from "vitest";

import { jobStatus } from "utils/Constants";

import {
    useCancelTrainTableJob,
    useSplitStats,
    useTrainingTableDetails,
    useTrainTableGenerationStats,
    useTraintableJobErrors
} from "../trainTableGen";

const renderWithQueryClient = <T>(hook: () => T) =>
    renderHook(hook, { wrapper: createWrapper(createQueryClient()) });

describe("useTrainingTableDetails", () => {
    it.each([
        ["id is not provided", undefined, true],
        ["valid id is provided", "gen-traintable-job-123", false]
    ])("Should return correct data when %s", async (_, id, dataIsUndefined) => {
        const { result } = renderWithQueryClient(() =>
            useTrainingTableDetails(id as never)
        );

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
            expect(result.current.data === undefined).toBe(dataIsUndefined);
        });
        expect(result.current.isError).toBe(false);
    });
});

describe("useTrainTableGenerationStats", () => {
    it.each([
        ["id is empty", "", true],
        ["id is valid", "gen-traintable-job-123", false]
    ])("Should return correct stats when %s", async (_, id, statsIsNull) => {
        const { result } = renderWithQueryClient(() =>
            useTrainTableGenerationStats(id, jobStatus.Done)
        );

        await waitFor(() => expect(result.current.receivedResponse).toBe(true));
        expect(result.current.stats === null).toBe(statsIsNull);
    });
});

describe("useSplitStats", () => {
    it.each([
        ["id is empty", "", false, false],
        ["isSourceSDK is false", "gen-traintable-job-123", false, true],
        ["isSourceSDK is true", "gen-traintable-job-123", true, true]
    ])(
        "Should return correct data when %s",
        async (_, id, isSourceSDK, dataIsDefined) => {
            const { result } = renderWithQueryClient(() =>
                useSplitStats(id, isSourceSDK, jobStatus.Done)
            );

            await waitFor(() =>
                expect(result.current !== undefined).toBe(dataIsDefined)
            );
        }
    );
});

describe("useCancelTrainTableJob", () => {
    it("Should return cancelTrainTable as a function", () => {
        const { result } = renderWithQueryClient(() =>
            useCancelTrainTableJob({
                jobId: "gen-traintable-job-123",
                handleError: vi.fn()
            })
        );

        expect(typeof result.current.cancelTrainTable).toBe("function");
    });
});

describe("useTraintableJobErrors", () => {
    it.each([
        ["jobId is empty", ""],
        ["jobId is valid", "gen-traintable-job-123"]
    ])("Should return empty errors and warnings when %s", async (_, jobId) => {
        const { result } = renderWithQueryClient(() =>
            useTraintableJobErrors({ jobId, trainTableDetails: null })
        );

        await waitFor(() => {
            expect(result.current.errors).toEqual([]);
            expect(result.current.warnings).toEqual([]);
            expect(result.current.fetchingErrors).toBe(false);
        });
    });
});
