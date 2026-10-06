// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { usePrefetch } from "../prefetch";

let queryClient: QueryClient;

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
});

afterEach(() => {
    queryClient.clear();
});

const renderPrefetch = () =>
    renderHook(() => usePrefetch(), { wrapper: createWrapper(queryClient) });

describe("prefetch", () => {
    it.each([
        {
            extraArgs: ["arg1"],
            expectedCall: ["arg1", {}],
            description: "plain args — empty options appended"
        },
        {
            extraArgs: ["arg1", { query: { staleTime: 1000 } }],
            expectedCall: ["arg1", { query: { staleTime: 1000 } }],
            description: "options object as last arg — extracted from params"
        }
    ])(
        "Should call prefetchFn with queryClient and correct args when $description",
        async ({ extraArgs, expectedCall }) => {
            const { result } = renderPrefetch();
            const fn = vi.fn(async (qc: QueryClient) => qc);

            await result.current.prefetch(fn, ...extraArgs);

            expect(fn).toHaveBeenCalledWith(queryClient, ...expectedCall);
        }
    );

    it("Should deduplicate in-flight calls with the same args", async () => {
        const { result } = renderPrefetch();
        const fn = vi.fn(async (qc: QueryClient) => qc);

        result.current.prefetch(fn, "arg");
        result.current.prefetch(fn, "arg");

        await result.current.prefetch(fn, "arg");

        expect(fn).toHaveBeenCalledTimes(1);
    });

    it("Should mark query as prefetched and isPrefetched reflects the state", async () => {
        const { result } = renderPrefetch();
        const fn = vi.fn(async (qc: QueryClient) => qc);

        expect(result.current.isPrefetched(fn, "arg")).toBe(false);

        await result.current.prefetch(fn, "arg");

        expect(result.current.isPrefetched(fn, "arg")).toBe(true);
    });

    it("Should re-throw when prefetchFn fails", async () => {
        const { result } = renderPrefetch();
        const fn = vi.fn(async () => {
            throw new Error("prefetch failed");
        });

        await expect(result.current.prefetch(fn)).rejects.toThrow(
            "prefetch failed"
        );
    });
});

describe("prefetchWithConfig", () => {
    it.each([
        {
            params: ["id1"],
            options: { query: { staleTime: 1000 } },
            expectedCall: ["id1", { query: { staleTime: 1000 } }],
            description: "with params and options"
        },
        {
            params: undefined,
            options: undefined,
            expectedCall: [{}],
            description: "without params or options"
        }
    ])(
        "Should call prefetchFn correctly $description",
        async ({ params, options, expectedCall }) => {
            const { result } = renderPrefetch();
            const fn = vi.fn(async (qc: QueryClient) => qc);

            await result.current.prefetchWithConfig({
                prefetchFn: fn,
                params,
                options
            });

            expect(fn).toHaveBeenCalledWith(queryClient, ...expectedCall);
        }
    );
});

describe("prefetchParallel", () => {
    it("Should call all prefetch functions and return fulfilled results", async () => {
        const { result } = renderPrefetch();
        const fn1 = vi.fn(async (qc: QueryClient) => qc);
        const fn2 = vi.fn(async (qc: QueryClient) => qc);

        const results = await result.current.prefetchParallel([
            { prefetchFn: fn1, params: ["a"] },
            { prefetchFn: fn2, params: ["b"] }
        ]);

        expect(fn1).toHaveBeenCalled();
        expect(fn2).toHaveBeenCalled();
        expect(results).toHaveLength(2);
        expect(results[0].status).toBe("fulfilled");
        expect(results[1].status).toBe("fulfilled");
    });

    it("Should return rejected status without throwing when a prefetch fails", async () => {
        const { result } = renderPrefetch();
        const fn1 = vi.fn(async (qc: QueryClient) => qc);
        const fn2 = vi.fn(async () => {
            throw new Error("failed");
        });

        const results = await result.current.prefetchParallel([
            { prefetchFn: fn1, params: ["success"] },
            { prefetchFn: fn2, params: ["fail"] }
        ]);

        expect(results[0].status).toBe("fulfilled");
        expect(results[1].status).toBe("rejected");
    });
});

describe("prefetchSequential", () => {
    it("Should call prefetch functions in order and return success results", async () => {
        const { result } = renderPrefetch();
        const callOrder: number[] = [];
        const fn1 = vi.fn(async (qc: QueryClient) => {
            callOrder.push(1);
            return qc;
        });
        const fn2 = vi.fn(async (qc: QueryClient) => {
            callOrder.push(2);
            return qc;
        });

        const results = await result.current.prefetchSequential([
            { prefetchFn: fn1 },
            { prefetchFn: fn2 }
        ]);

        expect(results).toEqual([{ success: true }, { success: true }]);
        expect(callOrder).toEqual([1, 2]);
    });

    it("Should continue after failure and return error result for failed prefetch", async () => {
        const { result } = renderPrefetch();
        const fn1 = vi.fn(async () => {
            throw new Error("fn1 failed");
        });
        const fn2 = vi.fn(async (qc: QueryClient) => qc);

        const results = await result.current.prefetchSequential([
            { prefetchFn: fn1 },
            { prefetchFn: fn2 }
        ]);

        expect(results[0]).toEqual({
            success: false,
            error: expect.any(Error)
        });
        expect(results[1]).toEqual({ success: true });
        expect(fn2).toHaveBeenCalled();
    });
});

describe("prefetchConditional", () => {
    it("Should return undefined when no conditions pass", async () => {
        const { result } = renderPrefetch();
        const fn = vi.fn(async (qc: QueryClient) => qc);

        const output = await result.current.prefetchConditional([
            { prefetchFn: fn, condition: () => false }
        ]);

        expect(output).toBeUndefined();
        expect(fn).not.toHaveBeenCalled();
    });

    it("Should only prefetch for configs where condition returns true", async () => {
        const { result } = renderPrefetch();
        const fn1 = vi.fn(async (qc: QueryClient) => qc);
        const fn2 = vi.fn(async (qc: QueryClient) => qc);

        const results = await result.current.prefetchConditional([
            { prefetchFn: fn1, condition: () => true },
            { prefetchFn: fn2, condition: () => false }
        ]);

        expect(fn1).toHaveBeenCalled();
        expect(fn2).not.toHaveBeenCalled();
        expect(results).toHaveLength(1);
    });
});

describe("clearPrefetchCache", () => {
    it("Should reset isPrefetched to false after clearing the cache", async () => {
        const { result } = renderPrefetch();
        const fn = vi.fn(async (qc: QueryClient) => qc);

        await result.current.prefetch(fn, "arg");
        expect(result.current.isPrefetched(fn, "arg")).toBe(true);

        result.current.clearPrefetchCache();

        expect(result.current.isPrefetched(fn, "arg")).toBe(false);
    });
});
