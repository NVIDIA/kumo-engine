// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook, waitFor } from "@testing-library/react";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { describe, expect, it, vi } from "vitest";

import sampleOutputSchemaJson from "components/NewJob/Prediction/__tests__/MockResponses/SampleOutputSchema.json";

import outputSchemaJson from "pages/NewJob/Prediction/__tests__/MockResponse/OutputSchema.json";

import { jobStatus } from "utils/Constants";

import {
    useCancelGenpredtableJobs,
    useGenpredtableAnchorTime,
    useGenPredTableSummary,
    usePredictionTableNumEntities,
    usePredictionTableSchema,
    usePredictionTableSchemaSamples
} from "../predictionTableGen";

const expectedSchema = outputSchemaJson.map((col) => ({
    name: col.name,
    colType: col.dtype
}));

const renderWithQueryClient = <T>(hook: () => T) =>
    renderHook(hook, { wrapper: createWrapper(createQueryClient()) });

describe("usePredictionTableSchema", () => {
    it.each([
        ["no args provided", undefined, undefined, undefined, []],
        [
            "predictiveQueryID provided",
            "pq-123",
            undefined,
            undefined,
            expectedSchema
        ],
        [
            "taskType and queryType provided",
            undefined,
            "BINARY_CLASSIFICATION",
            "TOP_K",
            expectedSchema
        ]
    ])(
        "Should return correct schema when %s",
        async (
            _,
            predictiveQueryID,
            taskType,
            queryType,
            expectedSchemaResult
        ) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionTableSchema(
                    predictiveQueryID,
                    taskType as never,
                    queryType as never
                )
            );

            await waitFor(() =>
                expect(result.current[0]).toEqual(expectedSchemaResult)
            );
            expect(result.current[1]).toBe(false);
        }
    );
});

describe("usePredictionTableSchemaSamples", () => {
    it.each([
        ["no args provided", undefined, undefined, []],
        [
            "trainingTableID and taskType provided",
            "train-table-123",
            "BINARY_CLASSIFICATION",
            sampleOutputSchemaJson
        ]
    ])(
        "Should return correct samples when %s",
        async (_, trainingTableID, taskType, expectedSamples) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionTableSchemaSamples(
                    trainingTableID,
                    taskType as never
                )
            );

            await waitFor(() =>
                expect(result.current[0]).toEqual(expectedSamples)
            );
        }
    );
});

describe("usePredictionTableNumEntities", () => {
    it.each([
        ["id is empty", "", null],
        ["valid id is provided", "gen-job-123", "2"]
    ])(
        "Should return correct entity count when %s",
        async (_, id, expected) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionTableNumEntities({ id, status: jobStatus.Running })
            );

            await waitFor(() => expect(result.current).toBe(expected));
        }
    );
});

describe("useGenPredTableSummary", () => {
    it.each([
        ["id is empty", "", true],
        ["valid id is provided", "genpredtable-job-123", false]
    ])(
        "Should return correct details when %s",
        async (_, id, detailsIsNull) => {
            const handleError = vi.fn();
            const setPageLoading = vi.fn();
            const { result } = renderWithQueryClient(() =>
                useGenPredTableSummary(id, handleError, setPageLoading)
            );

            await waitFor(() => {
                expect(result.current.isDataLoading).toBe(false);
                expect(result.current.predictionTableDetails === null).toBe(
                    detailsIsNull
                );
            });
        }
    );
});

describe("useGenpredtableAnchorTime", () => {
    it.each([
        ["id is empty", "", null],
        ["valid id is provided", "genpredtable-job-123", "2011-12-08T13:46:00"]
    ])(
        "Should return correct anchorTime when %s",
        async (_, id, expectedAnchorTime) => {
            const handleError = vi.fn();
            const setPageLoading = vi.fn();
            const { result } = renderWithQueryClient(() =>
                useGenpredtableAnchorTime(id, null, handleError, setPageLoading)
            );

            await waitFor(() =>
                expect(result.current.isDataLoading).toBe(false)
            );
            expect(result.current.anchorTime).toBe(expectedAnchorTime);
        }
    );
});

describe("useCancelGenpredtableJobs", () => {
    it("Should return cancelGenpredtableJobs as a function", () => {
        const { result } = renderWithQueryClient(() =>
            useCancelGenpredtableJobs({
                bpId: "bp-job-123",
                id: "genpredtable-job-123",
                handleError: vi.fn()
            })
        );

        expect(typeof result.current.cancelGenpredtableJobs).toBe("function");
    });
});
