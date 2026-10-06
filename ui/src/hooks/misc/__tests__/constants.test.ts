// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { DebugInfo } from "gen/v0/types";
import { setDebugInfoOverride } from "stores/__tests__/MockFunctions/DebugInfo.mock";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { useDebugInfo } from "../constants";

import "tests/mockAxios";

let queryClient: QueryClient;

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    setDebugInfoOverride(null);
});

afterEach(() => {
    queryClient.clear();
});

describe("useDebugInfo", () => {
    it("Should load and return debugInfo with profileOutputDir", async () => {
        setDebugInfoOverride({
            db_path: "/test/path/to",
            release_version: "1.2.3"
        } as Partial<DebugInfo>);

        const { result } = renderHook(() => useDebugInfo(), {
            wrapper: createWrapper(queryClient)
        });

        expect(result.current.isLoading).toBe(true);

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.debugInfo).toBeTruthy();
        expect(result.current.debugInfo?.db_path).toBe("/test/path/to");
        expect(result.current.debugInfo?.release_version).toBe("1.2.3");
        expect(result.current.profileOutputDir).toBe(
            "/test/path/to/profiling/1.2.3/"
        );
    });

    it("Should return empty/undefined when values are missing", async () => {
        setDebugInfoOverride({
            db_path: undefined,
            release_version: undefined
        } as Partial<DebugInfo>);

        const { result } = renderHook(() => useDebugInfo(), {
            wrapper: createWrapper(queryClient)
        });

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.profileOutputDir).toBe("");
        expect(result.current.debugInfo?.release_version).toBe(undefined);
    });

    it("Should provide refetch function", async () => {
        const { result } = renderHook(() => useDebugInfo(), {
            wrapper: createWrapper(queryClient)
        });

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.refetch).toBeDefined();
        expect(typeof result.current.refetch).toBe("function");
    });
});
