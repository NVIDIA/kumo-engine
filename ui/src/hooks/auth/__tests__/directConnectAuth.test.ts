// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    clearSessionStorage,
    storeToLocalStorage
} from "@kumo-ai/kumo-ui-catalog";
import { type UserDetails, useUserStore } from "stores/userStore";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
    isDirectConnectLoggedOut,
    loginDirectConnect,
    logoutDirectConnect
} from "hooks/auth/directConnectAuth";

import tokenService from "services/token_service";

import { storageKeys } from "utils/Constants";

import "tests/mockAxios";

const mocks = vi.hoisted(() => ({
    fetch: vi.fn()
}));

const mockAssign = vi.fn();
const tokenClearSpy = vi.spyOn(tokenService, "clear");

const loggedInUserDetails: UserDetails = {
    first_name: "Alice",
    last_name: "Example",
    email: "alice@example.com",
    isLoggedIn: true,
    picture: "",
    user_id: "direct_connect|alice",
    name: "Alice Example"
};

describe("directConnectAuth session", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        localStorage.clear();
        clearSessionStorage();
        useUserStore.setState({
            userDetails: { ...loggedInUserDetails },
            adminInfo: null
        });
        Object.defineProperty(globalThis, "fetch", {
            value: mocks.fetch,
            writable: true
        });
        Object.defineProperty(window, "location", {
            value: {
                assign: mockAssign,
                origin: "https://app.example.com",
                pathname: "/graphs/",
                search: "?project=demo"
            },
            writable: true
        });
    });

    describe("loginDirectConnect", () => {
        it("sends a relative return_to for the current path", () => {
            loginDirectConnect();

            expect(mockAssign).toHaveBeenCalledWith(
                "/v0/auth/login?return_to=%2Fgraphs%2F%3Fproject%3Ddemo"
            );
        });

        it("ignores unsafe stored paths", () => {
            storeToLocalStorage(
                storageKeys.navPath,
                "https://evil.example.com/"
            );

            loginDirectConnect();

            expect(mockAssign).toHaveBeenCalledWith(
                "/v0/auth/login?return_to=%2Fgraphs%2F%3Fproject%3Ddemo"
            );
        });
    });

    describe("logoutDirectConnect", () => {
        it("clears client state after backend logout succeeds", async () => {
            mocks.fetch.mockResolvedValue({ ok: true } as Response);

            await expect(logoutDirectConnect()).resolves.toBe(true);

            expect(mocks.fetch).toHaveBeenCalledWith("/v0/auth/logout", {
                method: "POST",
                credentials: "include"
            });
            expect(isDirectConnectLoggedOut()).toBe(true);
            expect(useUserStore.getState().userDetails.isLoggedIn).toBe(false);
            expect(tokenClearSpy).toHaveBeenCalled();
            expect(mockAssign).toHaveBeenCalledWith("/login?logged_out=1");
        });

        it("keeps client state when backend logout fails", async () => {
            mocks.fetch.mockResolvedValue({
                ok: false,
                status: 500
            } as Response);

            await expect(logoutDirectConnect()).resolves.toBe(false);

            expect(isDirectConnectLoggedOut()).toBe(false);
            expect(useUserStore.getState().userDetails).toEqual(
                loggedInUserDetails
            );
            expect(tokenClearSpy).not.toHaveBeenCalled();
            expect(mockAssign).not.toHaveBeenCalled();
        });
    });
});
