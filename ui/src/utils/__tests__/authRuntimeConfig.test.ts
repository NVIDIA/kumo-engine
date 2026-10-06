// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
    getAuthClientConfig,
    loadAuthClientConfig,
    setAuthClientConfig
} from "utils/authRuntimeConfig";

describe("authRuntimeConfig", () => {
    const originalApiTarget = process.env.API_TARGET;

    beforeEach(() => {
        vi.clearAllMocks();
        setAuthClientConfig(null);
        process.env.API_TARGET = "https://api.example.test/";
    });

    afterEach(() => {
        if (originalApiTarget === undefined) {
            delete process.env.API_TARGET;
        } else {
            process.env.API_TARGET = originalApiTarget;
        }
        setAuthClientConfig(null);
        vi.restoreAllMocks();
    });

    it("loads Direct Connect client config with cookie credentials", async () => {
        vi.spyOn(globalThis, "fetch").mockResolvedValue({
            ok: true,
            json: async () => ({
                auth_disabled: false,
                direct_connect_login_method: "password"
            })
        } as Response);

        await expect(loadAuthClientConfig()).resolves.toEqual({
            auth_disabled: false,
            direct_connect_login_method: "password"
        });

        expect(globalThis.fetch).toHaveBeenCalledWith(
            "https://api.example.test/v0/authentication/client-config",
            { credentials: "include" }
        );
        expect(getAuthClientConfig()).toEqual({
            auth_disabled: false,
            direct_connect_login_method: "password"
        });
    });

    it("does not store malformed client config responses", async () => {
        vi.spyOn(globalThis, "fetch").mockResolvedValue({
            ok: true,
            json: async () => ({
                auth_disabled: "false",
                direct_connect_login_method: "password"
            })
        } as Response);

        await expect(loadAuthClientConfig()).resolves.toBeNull();

        expect(getAuthClientConfig()).toBeNull();
    });
});
