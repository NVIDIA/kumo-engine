// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    getFromSessionStorage,
    storeToLocalStorage
} from "@kumo-ai/kumo-ui-catalog";
import { create } from "zustand";

export interface CacheConfig {
    enabled: boolean;
    staleTimeShort: number;
    staleTimeMedium: number;
    staleTimeLong: number;
}

const CACHE_STORAGE_KEY = "kumo_cache_config";

const getDefaultCacheConfig = (): CacheConfig => ({
    enabled: true,
    staleTimeShort: 2,
    staleTimeMedium: 5,
    staleTimeLong: 30
});

const loadCacheConfigFromStorage = (): CacheConfig => {
    try {
        const stored = getFromSessionStorage<string>(CACHE_STORAGE_KEY);
        if (stored) {
            const parsed = JSON.parse(stored) as CacheConfig;
            // Validate the config has required fields
            if (
                typeof parsed.enabled === "boolean" &&
                typeof parsed.staleTimeShort === "number" &&
                typeof parsed.staleTimeMedium === "number" &&
                typeof parsed.staleTimeLong === "number"
            ) {
                return parsed;
            }
        }
    } catch (error) {
        console.warn("Failed to load cache config from localStorage:", error);
        localStorage.removeItem(CACHE_STORAGE_KEY);
    }

    return getDefaultCacheConfig();
};

const computeStaleTimesFromConfig = (cacheConfig: CacheConfig) => {
    if (!cacheConfig.enabled) {
        return {
            short: 0,
            medium: 0,
            long: 0
        };
    }

    return {
        short: cacheConfig.staleTimeShort * 1000 * 60,
        medium: cacheConfig.staleTimeMedium * 1000 * 60,
        long: cacheConfig.staleTimeLong * 1000 * 60
    };
};

export const getDefaultStaleTimes = () => {
    const defaultConfig = getDefaultCacheConfig();
    return computeStaleTimesFromConfig(defaultConfig);
};

type CacheState = {
    cacheConfig: CacheConfig;
    short: number;
    medium: number;
    long: number;
    setCacheConfig: (config: CacheConfig) => void;
    resetCacheConfig: () => void;
};

export const useCacheStore = create<CacheState>((set) => {
    const initialConfig = loadCacheConfigFromStorage();

    return {
        cacheConfig: initialConfig,
        ...computeStaleTimesFromConfig(initialConfig),

        setCacheConfig: (config: CacheConfig) => {
            storeToLocalStorage(CACHE_STORAGE_KEY, config);
            set({
                cacheConfig: config,
                ...computeStaleTimesFromConfig(config)
            });
        },

        resetCacheConfig: () => {
            // Delete the localStorage item to reset to defaults
            try {
                localStorage.removeItem(CACHE_STORAGE_KEY);
            } catch (error) {
                console.warn(
                    "Failed to remove cache config from localStorage:",
                    error
                );
            }

            const config = getDefaultCacheConfig();
            set({
                cacheConfig: config,
                ...computeStaleTimesFromConfig(config)
            });
        }
    };
});

export const useCacheStaleTime = () => {
    const short = useCacheStore((state) => state.short);
    const medium = useCacheStore((state) => state.medium);
    const long = useCacheStore((state) => state.long);

    return {
        short,
        medium,
        long
    };
};
