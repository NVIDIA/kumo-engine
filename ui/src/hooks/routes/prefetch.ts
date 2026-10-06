// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable @typescript-eslint/no-explicit-any */
import { useCallback, useRef } from "react";

import {
    QueryClient,
    useQueryClient,
    UseQueryOptions
} from "@tanstack/react-query";

type PrefetchOptions = {
    query?: Partial<UseQueryOptions<any, any, any, readonly unknown[]>>;
    request?: any;
};

type PrefetchFn = (
    queryClient: QueryClient,
    ...args: any[]
) => Promise<QueryClient>;

export type PrefetchConfig = {
    prefetchFn: PrefetchFn;
    params?: any[];
    options?: PrefetchOptions;
};

type ConditionalPrefetchConfig = PrefetchConfig & {
    condition: () => boolean;
};

type PrefetchResult = {
    success: boolean;
    error?: Error;
};

export function usePrefetch() {
    const queryClient = useQueryClient();
    const prefetchedQueries = useRef<Set<string>>(new Set());
    const prefetchPromises = useRef<Map<string, Promise<QueryClient>>>(
        new Map()
    );

    const prefetch = useCallback(
        async (
            prefetchFn: PrefetchFn,
            ...args: any[]
        ): Promise<QueryClient> => {
            const cacheKey = JSON.stringify([prefetchFn.name, ...args]);

            const existingPromise = prefetchPromises.current.get(cacheKey);
            if (existingPromise) {
                return existingPromise;
            }

            const prefetchPromise = (async (): Promise<QueryClient> => {
                try {
                    let options: PrefetchOptions = {};
                    let params = [...args];

                    const lastArg = args[args.length - 1];
                    if (
                        lastArg &&
                        typeof lastArg === "object" &&
                        ("query" in lastArg || "request" in lastArg)
                    ) {
                        options = lastArg;
                        params = args.slice(0, -1);
                    }

                    const result = await prefetchFn(
                        queryClient,
                        ...params,
                        options
                    );

                    prefetchedQueries.current.add(cacheKey);
                    return result;
                } catch (error) {
                    console.warn("Prefetch failed:", error);
                    throw error;
                } finally {
                    prefetchPromises.current.delete(cacheKey);
                }
            })();

            prefetchPromises.current.set(cacheKey, prefetchPromise);
            return prefetchPromise;
        },
        [queryClient]
    );

    const prefetchWithConfig = useCallback(
        async (config: PrefetchConfig): Promise<QueryClient> => {
            const args: any[] = [];

            if (config.params && config.params.length > 0) {
                args.push(...config.params);
            }

            if (config.options) {
                args.push(config.options);
            }

            return prefetch(config.prefetchFn, ...args);
        },
        [prefetch]
    );

    const prefetchParallel = useCallback(
        async (
            prefetchConfigs: PrefetchConfig[]
        ): Promise<PromiseSettledResult<QueryClient>[]> => {
            const promises = prefetchConfigs.map((config) =>
                prefetchWithConfig(config)
            );
            return Promise.allSettled(promises);
        },
        [prefetchWithConfig]
    );

    const prefetchSequential = useCallback(
        async (
            prefetchConfigs: PrefetchConfig[]
        ): Promise<PrefetchResult[]> => {
            const results: PrefetchResult[] = [];

            for (const config of prefetchConfigs) {
                try {
                    await prefetchWithConfig(config);
                    results.push({ success: true });
                } catch (error) {
                    results.push({
                        success: false,
                        error: error as Error
                    });
                }
            }

            return results;
        },
        [prefetchWithConfig]
    );

    const prefetchConditional = useCallback(
        async (
            conditions: ConditionalPrefetchConfig[]
        ): Promise<PromiseSettledResult<QueryClient>[] | undefined> => {
            const validConfigs = conditions.filter(({ condition }) =>
                condition()
            );

            if (validConfigs.length > 0) {
                return prefetchParallel(validConfigs);
            }
        },
        [prefetchParallel]
    );

    const clearPrefetchCache = useCallback((): void => {
        prefetchedQueries.current.clear();
        prefetchPromises.current.clear();
    }, []);

    const isPrefetched = useCallback(
        (prefetchFn: PrefetchFn, ...args: any[]): boolean => {
            const cacheKey = JSON.stringify([prefetchFn.name, ...args]);
            return prefetchedQueries.current.has(cacheKey);
        },
        []
    );

    return {
        prefetch,
        prefetchWithConfig,
        prefetchParallel,
        prefetchSequential,
        prefetchConditional,
        clearPrefetchCache,
        isPrefetched
    };
}
