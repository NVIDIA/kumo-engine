// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    fireEvent,
    render,
    screen,
    waitFor
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { setDebugInfoOverride } from "stores/__tests__/MockFunctions/DebugInfo.mock";
import { useCacheStore } from "stores/cacheStore";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import tokenService from "services/token_service";

import DebugConsole from "../DebugConsole";

describe("DebugConsole", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });

    beforeEach(async () => {
        vi.clearAllMocks();
        localStorage.clear();

        tokenService.setToken("access-token", "refresh-token");

        useCacheStore.getState().resetCacheConfig();
    });

    const jobId = "trainingjob-bed43f8f2fc54ea985439b026dbe0865";
    const renderDebugConsole = async (initialPath = `/jobs/${jobId}`) => {
        const queryClient = createQueryClient();
        await act(async () => {
            render(
                <MemoryRouter initialEntries={[initialPath]}>
                    <QueryClientProvider client={queryClient}>
                        <DebugConsole />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async () => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/constants/flags`
            );
        });
    };

    const openDebugModal = async () => {
        await act(async () => {
            fireEvent.keyDown(window, { key: "d", metaKey: true });
        });

        await screen.findByText("Debug Console");
    };

    const openCacheConfiguration = async () => {
        await openDebugModal();

        const showButton = await screen.findByText("Show");
        await userEvent.click(showButton);

        await screen.findByText("Enable Caching");
    };

    const expectCacheInputsVisible = () => {
        expect(screen.getByText("Cache Times (minutes)")).toBeVisible();
        expect(screen.getByText("Short Cache (Jobs, live data)")).toBeVisible();
        expect(screen.getByText("Medium Cache (Most APIs)")).toBeVisible();
        expect(
            screen.getByText("Long Cache (Infrequently changing data)")
        ).toBeVisible();
    };

    const getCacheSwitch = async () => {
        const switches = await screen.findAllByRole("switch");
        return switches[0];
    };

    const toggleCacheSwitch = async (switchElement: HTMLElement) => {
        await userEvent.click(switchElement);
    };

    it("Should not render modal initially", async () => {
        await renderDebugConsole();

        expect(screen.queryByText("Debug Console")).not.toBeInTheDocument();
    });

    it("Should open modal when toggle function is called", async () => {
        await renderDebugConsole();
        await openDebugModal();

        await checkApiCall();

        expect(await screen.findByText("Debug Console")).toBeInTheDocument();
    });

    it("Should close modal when close button is clicked", async () => {
        await renderDebugConsole();
        await openDebugModal();

        // Close modal
        const closeButton = screen.getByRole("button", { name: "Close" });
        await userEvent.click(closeButton);

        await waitFor(() => {
            expect(screen.queryByText("Debug Console")).not.toBeInTheDocument();
        });
    });

    it("Should display all debug information when modal is open", async () => {
        useUserStore.getState().setUserDetails({ user_id: "test-user-123" });

        await renderDebugConsole();
        await openDebugModal();

        // Test user details
        expect(screen.getByText("test-user-123")).toBeInTheDocument();
        expect(screen.getByText("access-token")).toBeInTheDocument();
    });

    it("Should handle copy all functionality", async () => {
        const mockWriteText = vi.fn().mockResolvedValue(undefined);
        Object.defineProperty(navigator, "clipboard", {
            value: { writeText: mockWriteText },
            writable: true
        });

        await renderDebugConsole();
        await openDebugModal();

        const copyAllButton = await screen.findByText("Copy All");
        await userEvent.click(copyAllButton);

        expect(mockWriteText).toHaveBeenCalled();
        expect(
            await screen.findByText("Copied all debug info!")
        ).toBeInTheDocument();

        await waitFor(
            () => {
                expect(
                    screen.queryByText("Copied all debug info!")
                ).not.toBeInTheDocument();
            },
            { timeout: 3000 }
        );

        Object.defineProperty(navigator, "clipboard", {
            value: undefined,
            writable: true
        });
    });

    it("Should copy all debug info with comprehensive data", async () => {
        const mockWriteText = vi.fn().mockResolvedValue(undefined);
        Object.defineProperty(navigator, "clipboard", {
            value: { writeText: mockWriteText },
            writable: true
        });

        // Set up comprehensive test data
        useUserStore.getState().setUserDetails({ user_id: "test-user-123" });

        await renderDebugConsole(`/jobs/${jobId}`);
        await openDebugModal();

        const copyAllButton = await screen.findByText("Copy All");
        await userEvent.click(copyAllButton);

        expect(mockWriteText).toHaveBeenCalled();
        const copiedText = mockWriteText.mock.calls[0][0];

        // Verify all expected fields are included
        expect(copiedText).toContain("User ID: test-user-123");
        expect(copiedText).toContain("Access Token: access-token");
        expect(copiedText).toContain("REST API URL:");

        Object.defineProperty(navigator, "clipboard", {
            value: undefined,
            writable: true
        });
    });

    it("Should exclude DB path in production", async () => {
        const mockWriteText = vi.fn().mockResolvedValue(undefined);
        Object.defineProperty(navigator, "clipboard", {
            value: { writeText: mockWriteText },
            writable: true
        });

        const originalEnv = process.env.NODE_ENV;
        process.env.NODE_ENV = "production";
        try {
            setDebugInfoOverride({
                db_path: "/secret/db/path"
            });

            await renderDebugConsole();
            await openDebugModal();

            const copyAllButton = await screen.findByText("Copy All");
            await userEvent.click(copyAllButton);

            const copiedText = mockWriteText.mock.calls[0][0];
            expect(copiedText).not.toContain("DB Path:");
        } finally {
            process.env.NODE_ENV = originalEnv;
            setDebugInfoOverride(null);

            Object.defineProperty(navigator, "clipboard", {
                value: undefined,
                writable: true
            });
        }
    });

    it("Should handle API errors gracefully", async () => {
        mockAxiosInstance.get.mockRejectedValue(new Error("API Error"));

        await renderDebugConsole();
        await openDebugModal();

        await waitFor(() => {
            expect(screen.getAllByText("N/A").length).toBeGreaterThan(0);
        });
    });

    it("Should display access token from token service", async () => {
        const tokenService = await import("services/token_service");
        tokenService.default.getAccessToken = vi.fn(() => "special-test-token");

        await renderDebugConsole();
        await openDebugModal();

        expect(
            await screen.findByText("special-test-token")
        ).toBeInTheDocument();
    });

    describe("Cache Configuration", () => {
        it("Should display cache configuration section", async () => {
            await renderDebugConsole();
            await openDebugModal();

            expect(
                await screen.findByText("Cache Configuration")
            ).toBeInTheDocument();
            expect(screen.getByText("Show")).toBeInTheDocument();
        });

        it("Should toggle cache configuration visibility", async () => {
            await renderDebugConsole();
            await openCacheConfiguration();

            const hideButton = await screen.findByText("Hide");
            await userEvent.click(hideButton);

            expect(await screen.findByText("Show")).toBeInTheDocument();

            // Check that the collapsible container is properly collapsed
            await waitFor(() => {
                const collapsibleContainer = screen
                    .getByText("Cache Configuration")
                    .closest('[data-slot="collapsible"]');
                expect(collapsibleContainer).toHaveAttribute(
                    "data-state",
                    "closed"
                );
            });
        });

        it("Should toggle cache enable/disable switch", async () => {
            await renderDebugConsole();
            await openCacheConfiguration();

            const cacheSwitch = await getCacheSwitch();
            expect(cacheSwitch).toBeChecked();
            expectCacheInputsVisible();

            // Disable caching
            await toggleCacheSwitch(cacheSwitch);
            await waitFor(() => {
                expect(cacheSwitch).not.toBeChecked();
            });

            // Re-enable caching
            await toggleCacheSwitch(cacheSwitch);
            await waitFor(() => {
                expect(cacheSwitch).toBeChecked();
            });
            expectCacheInputsVisible();
        });

        it("Should update cache time values and reset to defaults", async () => {
            await renderDebugConsole();
            await openCacheConfiguration();

            expect(await screen.findByDisplayValue("2")).toBeInTheDocument(); // Default short cache time

            // Modify values
            const shortCacheInput = screen.getByDisplayValue("2");
            await userEvent.clear(shortCacheInput);
            await userEvent.type(shortCacheInput, "10");
            expect(shortCacheInput).toHaveValue(10);

            const mediumCacheInput = screen.getByDisplayValue("5"); // Default medium cache time
            await userEvent.clear(mediumCacheInput);
            await userEvent.type(mediumCacheInput, "60");
            expect(mediumCacheInput).toHaveValue(60);

            // Reset to defaults
            const resetButton = screen.getByText("Reset");
            await userEvent.click(resetButton);

            expect(await screen.findByDisplayValue("2")).toBeInTheDocument();
            expect(screen.getByDisplayValue("5")).toBeInTheDocument();
        });
    });
});
