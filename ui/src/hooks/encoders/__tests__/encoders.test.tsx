// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { removeFromLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClient } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { vi } from "vitest";

import { storageKeys } from "utils/Constants";

import { useGetQueryEncoders } from "../encoders";

import "tests/mockAxios";

let queryClient: QueryClient;

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
});

afterEach(() => {
    queryClient.clear();
});

describe("useGetQueryEncoders", () => {
    const renderUseGetQueryEncoders = (
        queryId?: string,
        trainingJobId?: string,
        trainingStatus?: string
    ) => {
        return renderHook(
            () => useGetQueryEncoders(queryId, trainingJobId, trainingStatus),
            { wrapper: createWrapper(queryClient) }
        );
    };

    it("Should return empty object initially when no IDs provided", () => {
        const { result } = renderUseGetQueryEncoders();

        expect(result.current).toEqual({});
    });

    it("Should fetch encoders from predictive queries API when only queryId is provided", async () => {
        const { result } = renderUseGetQueryEncoders(
            "predictive-query-123",
            undefined,
            undefined
        );

        await waitFor(() => {
            expect(Object.keys(result.current).length).toBeGreaterThan(0);
        });

        expect(mockAxiosInstance.get).toHaveBeenCalledWith(
            "v1/predictive_queries/predictive-query-123/encoders",
            undefined,
            {
                headers: undefined,
                params: { project_id: "project-tests" }
            }
        );
        expect(result.current).toHaveProperty(
            "customerltv_mini_integ_test.transaction.Description"
        );
    });

    it("Should not fetch predictive query encoders without a project", () => {
        removeFromLocalStorage(storageKeys.selectedProject);

        renderUseGetQueryEncoders("predictive-query-123");

        expect(mockAxiosInstance.get).not.toHaveBeenCalled();
    });

    it("Should prioritize trainingJobId over queryId when both are provided", async () => {
        const { result } = renderUseGetQueryEncoders(
            "predictive-query-123",
            "training-job-456",
            "DONE"
        );

        await waitFor(() => {
            expect(Object.keys(result.current).length).toBeGreaterThan(0);
        });

        expect(mockAxiosInstance.get).toHaveBeenCalledWith(
            "v0/training_jobs/training-job-456/encoders"
        );
        expect(mockAxiosInstance.get).not.toHaveBeenCalledWith(
            expect.stringContaining("v1/predictive_queries/")
        );
        expect(result.current).toHaveProperty(
            "customerltv_mini_integ_test.transaction.Description"
        );
    });

    it.each([
        {
            description: "with DONE status",
            trainingJobId: "trainingjob-00048000000000000000000000000000",
            status: "DONE"
        },
        {
            description: "with RUNNING status",
            trainingJobId: "trainingjob-0001b000000000000000000000000000",
            status: "RUNNING"
        },
        {
            description: "with QUEUED status",
            trainingJobId: "trainingjob-00036000000000000000000000000000",
            status: "QUEUED"
        }
    ])(
        "Should fetch training job encoders $description",
        async ({ trainingJobId, status }) => {
            const { result } = renderUseGetQueryEncoders(
                undefined,
                trainingJobId,
                status
            );

            await waitFor(() => {
                expect(Object.keys(result.current).length).toBeGreaterThan(0);
            });

            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/training_jobs/${trainingJobId}/encoders`
            );
        }
    );
});
