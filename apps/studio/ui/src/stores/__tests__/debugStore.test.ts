// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useDebugStore } from "stores/debugStore";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import { describe, expect, it } from "vitest";

const mockResponse = <T>(data: T) =>
    ({
        data,
        status: 200,
        statusText: "OK",
        headers: {},
        config: { headers: {} }
    }) as never;

describe("useDebugStore", () => {
    it("marks config fetched when startup config requests fail", async () => {
        useUserStore.getState().setUserDetails({ isLoggedIn: true });
        useDebugStore.setState({ configFetched: false });

        mockAxiosInstance.get.mockRejectedValueOnce(
            new Error("feature flags unavailable")
        );

        await useDebugStore.getState().fetchData();

        expect(useDebugStore.getState().configFetched).toBe(true);
    });

    it("marks config fetched and stores config when requests succeed", async () => {
        useUserStore.getState().setUserDetails({ isLoggedIn: true });
        useDebugStore.setState({ configFetched: false });

        mockAxiosInstance.get.mockResolvedValueOnce(
            mockResponse({
                feature_flags: {
                    ENABLE_COLUMN_ANALYSIS: true
                }
            })
        );

        await useDebugStore.getState().fetchData();

        expect(useDebugStore.getState().configFetched).toBe(true);
        expect(
            useDebugStore.getState().data.feature_flags.ENABLE_COLUMN_ANALYSIS
        ).toBe(true);
    });
});
