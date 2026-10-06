// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setSDKGraphResponseTestType } from "components/SDKCodeModal/__tests__/MockFunctions/SDKGraphCodeModal.mock";
import { setSDKPredictionResponseTestType } from "components/SDKCodeModal/__tests__/MockFunctions/SDKPredictionCodeModal.mock";
import { setSDKTableResponseTestType } from "components/SDKCodeModal/__tests__/MockFunctions/SDKTableCodeModal.mock";
import { setSDKTrainingResponseTestType } from "components/SDKCodeModal/__tests__/MockFunctions/SDKTrainingCodeModal.mock";

import { SDKObjectType } from "types/constants";

import { useSDKCode } from "../sdk";

import "tests/mockAxios";

let queryClient: QueryClient;

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
});

afterEach(() => {
    queryClient.clear();
});

describe("useSDKCode", () => {
    it.each([
        {
            description: "table",
            id: "table-123",
            type: SDKObjectType.Table,
            setter: setSDKTableResponseTestType
        },
        {
            description: "graph",
            id: "graph-456",
            type: SDKObjectType.Graph,
            setter: setSDKGraphResponseTestType
        },
        {
            description: "training job",
            id: "train-789",
            type: SDKObjectType.TrainingJob,
            setter: setSDKTrainingResponseTestType
        },
        {
            description: "prediction job",
            id: "pred-101",
            type: SDKObjectType.PredictionJob,
            setter: setSDKPredictionResponseTestType
        }
    ])(
        "Should fetch $description SDK code when type is appropriate",
        async ({ id, type, setter }) => {
            setter();

            const { result } = renderHook(() => useSDKCode(id, type), {
                wrapper: createWrapper(queryClient)
            });

            expect(result.current.isLoading).toBe(true);

            await waitFor(() => {
                expect(result.current.isLoading).toBe(false);
            });

            expect(result.current.data).toBeTruthy();
            expect(result.current.isError).toBe(false);
        }
    );

    it("Should handle error state for table SDK", async () => {
        setSDKTableResponseTestType(testResponseType.ERROR);

        const { result } = renderHook(
            () => useSDKCode("table-123", SDKObjectType.Table),
            {
                wrapper: createWrapper(queryClient)
            }
        );

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.isError).toBe(true);
        expect(result.current.error).toBeTruthy();
    });

    it("Should handle empty response", async () => {
        setSDKGraphResponseTestType(testResponseType.EMPTY);

        const { result } = renderHook(
            () => useSDKCode("graph-123", SDKObjectType.Graph),
            {
                wrapper: createWrapper(queryClient)
            }
        );

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.data?.data).toBeFalsy();
    });

    it("Should handle invalid response", async () => {
        setSDKTrainingResponseTestType(testResponseType.INVALID);

        const { result } = renderHook(
            () => useSDKCode("train-123", SDKObjectType.TrainingJob),
            {
                wrapper: createWrapper(queryClient)
            }
        );

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.isError).toBe(true);
    });
});
