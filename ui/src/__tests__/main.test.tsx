// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createBrowserRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { useUserStore } from "stores/userStore";
import { createQueryClient } from "tests/testUtil";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mockRootRender = vi.hoisted(() => vi.fn());
const mockInitDirectConnect = vi.hoisted(() =>
    vi.fn<() => Promise<boolean>>().mockResolvedValue(false)
);
const mockLogoutDirectConnect = vi.hoisted(() =>
    vi.fn<() => Promise<boolean>>().mockResolvedValue(false)
);

vi.mock("react-dom/client", () => ({
    default: {
        createRoot: () => ({
            render: mockRootRender
        })
    },
    createRoot: () => ({
        render: mockRootRender
    })
}));

vi.mock("react-router", async () => {
    const actual = await vi.importActual("react-router");
    return {
        ...actual,
        RouterProvider: () => <div data-testid="browser-router" />
    };
});

vi.mock("components/Debug/ReactQueryDevTool", () => ({
    default: () => null
}));

vi.mock("utils/storage", async () => {
    const actual = await vi.importActual("utils/storage");
    return {
        ...actual,
        inDebugMode: () =>
            JSON.parse(localStorage.getItem("DEBUG") || "false") === "true"
    };
});

vi.mock("hooks/auth/directConnectAuth", () => ({
    clearDirectConnectLoggedOut: vi.fn(),
    completeDirectConnectPasswordReset: vi.fn(),
    initDirectConnect: mockInitDirectConnect,
    isDirectConnectLoggedOut: vi.fn(() => false),
    loginDirectConnect: vi.fn(),
    loginDirectConnectWithPassword: vi.fn(),
    logoutDirectConnect: mockLogoutDirectConnect
}));

vi.mock("utils/reportWebVitals", () => ({
    default: vi.fn()
}));

const mockTokenService = {
    clear: vi.fn(),
    setToken: vi.fn(),
    setSSOFlow: vi.fn()
};

vi.mock("services/token_service", () => ({
    default: mockTokenService
}));

const setupWindowEnv = (env: Record<string, string | boolean | undefined>) => {
    const filteredEnv = Object.entries(env).reduce(
        (acc, [key, value]) => {
            if (value !== undefined) {
                acc[key] = value;
            }
            return acc;
        },
        {} as Record<string, string | boolean>
    );
    (
        window as Window &
            typeof globalThis & { _env_: Record<string, string | boolean> }
    )._env_ = filteredEnv;
};

describe("main", () => {
    const renderBase = (
        Base: React.ComponentType<{
            router: ReturnType<typeof createBrowserRouter>;
        }>
    ) => {
        const queryClient = createQueryClient();
        const router = createBrowserRouter([
            { path: "*", element: <div data-testid="test-route" /> }
        ]);
        const result = render(
            <QueryClientProvider client={queryClient}>
                <Base router={router} />
            </QueryClientProvider>
        );
        return { ...result, queryClient, router };
    };

    beforeEach(() => {
        vi.resetModules();
        useUserStore.getState().clearUserDetails();
        localStorage.clear();
        setupWindowEnv({});

        const root = document.createElement("div");
        root.setAttribute("id", "root");
        document.body.appendChild(root);

        vi.clearAllMocks();
        mockInitDirectConnect.mockResolvedValue(false);
        mockLogoutDirectConnect.mockResolvedValue(false);
    });

    afterEach(() => {
        cleanup();
        const root = document.getElementById("root");
        if (root) document.body.removeChild(root);
    });

    it("renders Base without an auth provider wrapper", async () => {
        const { Base } = await import("../main");

        renderBase(Base);

        expect(screen.getByTestId("browser-router")).toBeInTheDocument();
    });

    it("initializes Direct Connect on startup", async () => {
        await import("../main");

        await waitFor(() => {
            expect(mockInitDirectConnect).toHaveBeenCalledTimes(1);
        });
    });

    it("seeds auth-disabled session without initializing Direct Connect", async () => {
        setupWindowEnv({ AUTH_DISABLED: true });

        await import("../main");

        expect(mockInitDirectConnect).not.toHaveBeenCalled();
        expect(mockTokenService.setToken).toHaveBeenCalledWith(
            "AUTH_DISABLED",
            "AUTH_DISABLED"
        );
        const { useUserStore: initializedUserStore } =
            await import("stores/userStore");
        await waitFor(() => {
            expect(initializedUserStore.getState().userDetails.isLoggedIn).toBe(
                true
            );
        });
    });
});
