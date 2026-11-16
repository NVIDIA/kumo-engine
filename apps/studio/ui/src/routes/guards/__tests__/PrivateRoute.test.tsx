// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import * as storage from "@kumo-ai/kumo-ui-catalog";
import { render, screen } from "@testing-library/react";
import { useUserStore } from "stores/userStore";
import { vi } from "vitest";

import * as constants from "utils/Constants";

import PrivateRoute from "../PrivateRoute";

vi.mock("components/Layout/MainLayout", () => ({
    default: () => <div>MainLayout</div>
}));

// PrivateRoute calls usePreloadRoutes(), which schedules tiered route preloads
// via requestIdleCallback + setTimeout chains. Those timers fire dynamic
// import()s of lazy route modules that resolve *after* Vitest tears down the
// test environment, producing "Cannot load ... after the environment was torn
// down" unhandled rejections. This unit test only exercises auth/redirect
// logic, so stub the preload hook to a no-op to keep it isolated.
vi.mock("hooks/routes/preloadRoutes", () => ({
    __esModule: true,
    default: vi.fn(),
    usePreloadRoutes: vi.fn()
}));

vi.mock("react-router", async () => {
    const actual = await vi.importActual("react-router");
    return {
        ...actual,
        Navigate: ({ to }: { to: string }) => <div>Redirect to {to}</div>
    };
});

// Storage helpers now live in the catalog package, whose ESM namespace
// exports are not configurable — so they can't be spied on directly. Replace
// them with vi.fn()s via a partial module mock instead.
vi.mock("@kumo-ai/kumo-ui-catalog", async (importOriginal) => {
    const actual =
        await importOriginal<typeof import("@kumo-ai/kumo-ui-catalog")>();
    return {
        ...actual,
        storeToLocalStorage: vi.fn()
    };
});

describe("PrivateRoute", () => {
    const renderPrivateRoute = async () => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <PrivateRoute />
                </MemoryRouter>
            );
        });
    };

    it("Should render MainLayout if user is logged in", async () => {
        useUserStore.getState().setUserDetails({ isLoggedIn: true });
        await renderPrivateRoute();

        expect(screen.getByText("MainLayout")).toBeInTheDocument();
    });

    it("Should redirect to '/login' if user is not logged in", async () => {
        useUserStore.getState().setUserDetails({ isLoggedIn: false });
        await renderPrivateRoute();

        expect(screen.getByText("Redirect to /login")).toBeInTheDocument();
    });

    it("Should store current path if not logged in and path is private", async () => {
        useUserStore.getState().setUserDetails({ isLoggedIn: false });

        const privatePath = "/some-private-page";
        Object.defineProperty(window, "location", {
            value: { pathname: privatePath, search: "" },
            writable: true
        });

        vi.spyOn(constants, "publicPaths", "get").mockReturnValue(["/"]);
        const storeSpy = vi.mocked(storage.storeToLocalStorage);

        await renderPrivateRoute();

        expect(storeSpy).toHaveBeenCalledWith(
            constants.storageKeys.navPath,
            privatePath
        );
    });

    it("Should not store current path if not logged in and path is public", async () => {
        useUserStore.getState().setUserDetails({ isLoggedIn: false });

        const publicPath = "/public-page";
        Object.defineProperty(window, "location", {
            value: { pathname: publicPath, search: "" },
            writable: true
        });

        vi.spyOn(constants, "publicPaths", "get").mockReturnValue([publicPath]);
        const storeSpy = vi.mocked(storage.storeToLocalStorage);

        await renderPrivateRoute();

        expect(storeSpy).not.toHaveBeenCalled();
    });

    it("Should store empty pathname if it's not in publicPaths", async () => {
        useUserStore.getState().setUserDetails({ isLoggedIn: false });

        Object.defineProperty(window, "location", {
            value: { pathname: "", search: "" },
            writable: true
        });

        vi.spyOn(constants, "publicPaths", "get").mockReturnValue(["/"]);
        const storeSpy = vi.mocked(storage.storeToLocalStorage);

        await renderPrivateRoute();

        expect(storeSpy).toHaveBeenCalledWith(
            constants.storageKeys.navPath,
            ""
        );
    });

    it("Should not store path if user is logged in", async () => {
        useUserStore.getState().setUserDetails({ isLoggedIn: true });

        Object.defineProperty(window, "location", {
            value: { pathname: "/private-page", search: "" },
            writable: true
        });

        const storeSpy = vi.mocked(storage.storeToLocalStorage);

        await renderPrivateRoute();

        expect(storeSpy).not.toHaveBeenCalled();
    });

    it("Should redirect if publicPaths is empty", async () => {
        useUserStore.getState().setUserDetails({ isLoggedIn: false });

        Object.defineProperty(window, "location", {
            value: { pathname: "/some-private-page", search: "" },
            writable: true
        });

        vi.spyOn(constants, "publicPaths", "get").mockReturnValue([]);
        const storeSpy = vi.mocked(storage.storeToLocalStorage);

        await renderPrivateRoute();

        expect(storeSpy).toHaveBeenCalled();
        expect(screen.getByText("Redirect to /login")).toBeInTheDocument();
        expect(storeSpy).toHaveBeenCalledWith(
            constants.storageKeys.navPath,
            "/some-private-page"
        );
    });

    it("Should store path with query parameters when present", async () => {
        useUserStore.getState().setUserDetails({ isLoggedIn: false });

        const privatePath = "/models/";
        const searchParams = "?page=1&job_type=TRAINING_JOB";
        Object.defineProperty(window, "location", {
            value: { pathname: privatePath, search: searchParams },
            writable: true
        });

        vi.spyOn(constants, "publicPaths", "get").mockReturnValue(["/"]);
        const storeSpy = vi.mocked(storage.storeToLocalStorage);

        await renderPrivateRoute();

        expect(storeSpy).toHaveBeenCalledWith(
            constants.storageKeys.navPath,
            privatePath + searchParams
        );
    });
});
