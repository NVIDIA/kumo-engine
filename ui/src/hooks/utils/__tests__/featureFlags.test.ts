// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from "@testing-library/react";
import { FeatureFlags } from "gen/v0/types";
import { useDebugStore } from "stores/debugStore";
import { beforeEach, describe, expect, it, vi } from "vitest";

import useFeatureFlags from "../featureFlags";

vi.mock("stores/debugStore");

describe("useFeatureFlags", () => {
    const mockStoreWithFlags = (flags: FeatureFlags) => {
        vi.mocked(useDebugStore).mockReturnValue({
            data: { feature_flags: flags }
        } as ReturnType<typeof useDebugStore>);
    };

    const renderFeatureFlags = () => renderHook(() => useFeatureFlags());

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should return feature flags from debug store", () => {
        const flags = {
            enable_new_feature: true,
            enable_beta_mode: false,
            enable_advanced_analytics: true,
            GCP_WORKLOAD_EXCHANGE_ENABLED: true
        } as FeatureFlags;

        mockStoreWithFlags(flags);

        const { result } = renderFeatureFlags();

        expect(result.current).toEqual(flags);
        expect(result.current.GCP_WORKLOAD_EXCHANGE_ENABLED).toBe(true);
    });

    it("Should return empty object when feature flags are not set", () => {
        mockStoreWithFlags({} as FeatureFlags);

        const { result } = renderFeatureFlags();

        expect(result.current).toEqual({});
    });

    it("Should update when debug store changes", () => {
        const initialFlags = { enable_new_feature: false } as FeatureFlags;
        mockStoreWithFlags(initialFlags);

        const { result, rerender } = renderFeatureFlags();

        expect(result.current).toEqual(initialFlags);

        const updatedFlags = { enable_new_feature: true } as FeatureFlags;
        mockStoreWithFlags(updatedFlags);

        rerender();

        expect(result.current).toEqual(updatedFlags);
    });
});
