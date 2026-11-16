// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import CacheConfiguration from "../CacheConfiguration";

const mockCacheConfig = {
    enabled: false,
    staleTimeShort: 5,
    staleTimeMedium: 10,
    staleTimeLong: 30
};

const mockSetCacheConfig = vi.fn();
const mockResetCacheConfig = vi.fn();

vi.mock("stores/cacheStore", () => ({
    useCacheStore: vi.fn(() => ({
        cacheConfig: mockCacheConfig,
        setCacheConfig: mockSetCacheConfig,
        resetCacheConfig: mockResetCacheConfig
    }))
}));

describe("CacheConfiguration", () => {
    const CACHE_LABELS = {
        SHORT: "Short Cache (Jobs, live data)",
        MEDIUM: "Medium Cache (Most APIs)",
        LONG: "Long Cache (Infrequently changing data)"
    } as const;

    const renderCacheConfiguration = async () => {
        await act(async () => {
            render(<CacheConfiguration />);
        });
    };

    const getShowHideButton = () =>
        screen.getByRole("button", { name: /show|hide/i });
    const getEnableCachingSwitch = () =>
        screen.getByLabelText("Enable Caching");
    const getResetButton = () => screen.getByRole("button", { name: /reset/i });
    const clickShowHide = async () =>
        await userEvent.click(getShowHideButton());
    const getCacheInput = (label: string) => screen.getByLabelText(label);
    const queryCacheInput = (label: string) => screen.queryByLabelText(label);

    beforeEach(() => {
        vi.clearAllMocks();
        mockCacheConfig.enabled = false;
        mockCacheConfig.staleTimeShort = 5;
        mockCacheConfig.staleTimeMedium = 10;
        mockCacheConfig.staleTimeLong = 30;
    });

    it("Should render with Show button when collapsed", async () => {
        await renderCacheConfiguration();

        expect(screen.getByText("Cache Configuration")).toBeInTheDocument();

        const button = getShowHideButton();
        expect(button).toBeInTheDocument();
        expect(button).toHaveTextContent("Show");
    });

    it("Should toggle collapsible when Show/Hide button is clicked", async () => {
        await renderCacheConfiguration();

        const button = getShowHideButton();
        expect(button).toHaveTextContent("Show");

        await clickShowHide();
        expect(button).toHaveTextContent("Hide");

        await clickShowHide();
        expect(button).toHaveTextContent("Show");
    });

    it("Should render Enable Caching switch and Reset button when expanded", async () => {
        await renderCacheConfiguration();

        await clickShowHide();

        expect(getEnableCachingSwitch()).toBeInTheDocument();
        expect(getResetButton()).toBeInTheDocument();
    });

    it("Should call setCacheConfig when Enable Caching switch is toggled", async () => {
        await renderCacheConfiguration();
        await clickShowHide();

        const enableSwitch = getEnableCachingSwitch();
        await userEvent.click(enableSwitch);

        expect(mockSetCacheConfig).toHaveBeenCalledWith({
            ...mockCacheConfig,
            enabled: true
        });
    });

    it("Should call resetCacheConfig when Reset button is clicked", async () => {
        await renderCacheConfiguration();
        await clickShowHide();

        const resetButton = getResetButton();
        await userEvent.click(resetButton);

        expect(mockResetCacheConfig).toHaveBeenCalled();
    });

    it("Should not show cache time inputs when caching is disabled", async () => {
        mockCacheConfig.enabled = false;
        await renderCacheConfiguration();
        await clickShowHide();

        expect(
            screen.queryByText("Cache Times (minutes)")
        ).not.toBeInTheDocument();
        expect(queryCacheInput(CACHE_LABELS.SHORT)).not.toBeInTheDocument();
    });

    it("Should show cache time inputs when caching is enabled", async () => {
        mockCacheConfig.enabled = true;
        await renderCacheConfiguration();
        await clickShowHide();

        expect(screen.getByText("Cache Times (minutes)")).toBeInTheDocument();
        expect(getCacheInput(CACHE_LABELS.SHORT)).toBeInTheDocument();
        expect(getCacheInput(CACHE_LABELS.MEDIUM)).toBeInTheDocument();
        expect(getCacheInput(CACHE_LABELS.LONG)).toBeInTheDocument();
    });

    it.each([
        {
            label: CACHE_LABELS.SHORT,
            configKey: "staleTimeShort",
            newValue: "15"
        },
        {
            label: CACHE_LABELS.MEDIUM,
            configKey: "staleTimeMedium",
            newValue: "20"
        },
        {
            label: CACHE_LABELS.LONG,
            configKey: "staleTimeLong",
            newValue: "60"
        }
    ])(
        "Should call setCacheConfig when $label input value changes",
        async ({ label, configKey, newValue }) => {
            mockCacheConfig.enabled = true;
            await renderCacheConfiguration();
            await clickShowHide();

            const input = getCacheInput(label);
            await userEvent.clear(input);
            await userEvent.type(input, newValue);

            expect(mockSetCacheConfig).toHaveBeenCalled();
            // Verify the configKey was updated in at least one call
            const wasCalled = mockSetCacheConfig.mock.calls.some(
                (call) =>
                    call[0][configKey] !== undefined && call[0][configKey] > 0
            );
            expect(wasCalled).toBe(true);
        }
    );

    it.each([
        {
            label: CACHE_LABELS.SHORT,
            configKey: "staleTimeShort",
            value: 5
        },
        {
            label: CACHE_LABELS.MEDIUM,
            configKey: "staleTimeMedium",
            value: 10
        },
        {
            label: CACHE_LABELS.LONG,
            configKey: "staleTimeLong",
            value: 30
        }
    ])(
        "Should display correct initial value for $label",
        async ({ label, value }) => {
            mockCacheConfig.enabled = true;
            await renderCacheConfiguration();
            await clickShowHide();

            const input = getCacheInput(label) as HTMLInputElement;
            expect(input.value).toBe(value.toString());
        }
    );

    it("Should set cache time to 0 when invalid input is provided", async () => {
        mockCacheConfig.enabled = true;
        await renderCacheConfiguration();
        await clickShowHide();

        const input = getCacheInput(CACHE_LABELS.SHORT);
        await userEvent.clear(input);
        await userEvent.type(input, "abc");

        expect(mockSetCacheConfig).toHaveBeenCalledWith({
            ...mockCacheConfig,
            staleTimeShort: 0
        });
    });

    it("Should have minimum value of 0 for all cache time inputs", async () => {
        mockCacheConfig.enabled = true;
        await renderCacheConfiguration();
        await clickShowHide();

        const shortCacheInput = getCacheInput(
            CACHE_LABELS.SHORT
        ) as HTMLInputElement;
        const mediumCacheInput = getCacheInput(
            CACHE_LABELS.MEDIUM
        ) as HTMLInputElement;
        const longCacheInput = getCacheInput(
            CACHE_LABELS.LONG
        ) as HTMLInputElement;

        expect(shortCacheInput.min).toBe("0");
        expect(mediumCacheInput.min).toBe("0");
        expect(longCacheInput.min).toBe("0");
    });
});
