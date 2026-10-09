// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it } from "vitest";

import { setAuthClientConfig } from "utils/authRuntimeConfig";
import {
    getDirectConnectLoginMethod,
    validateDirectConnectPassword
} from "utils/directConnectAuth";

describe("directConnectAuth", () => {
    afterEach(() => {
        setAuthClientConfig(null);
        (window as unknown as { _env_?: Record<string, string> })._env_ = {};
    });

    describe("validateDirectConnectPassword", () => {
        it("accepts a strong password", () => {
            expect(
                validateDirectConnectPassword(
                    "FinalPass67890",
                    "alice@example.com"
                )
            ).toBeNull();
        });

        it("rejects weak passwords", () => {
            expect(validateDirectConnectPassword("short1")).toBe(
                "Password must be at least 12 characters"
            );
            expect(validateDirectConnectPassword("NoDigitsHere")).toBe(
                "Password must include at least one letter and one number"
            );
            expect(validateDirectConnectPassword("Password12345")).toBe(
                "Password is too easy to guess"
            );
        });

        it("rejects passwords containing the email local part", () => {
            expect(
                validateDirectConnectPassword(
                    "AliceStrong12345",
                    "alice@example.com"
                )
            ).toBe("Password must not contain your email address");
        });
    });

    describe("getDirectConnectLoginMethod", () => {
        it("uses runtime config when available", () => {
            setAuthClientConfig({
                auth_disabled: false,
                direct_connect_login_method: "password"
            });

            expect(getDirectConnectLoginMethod()).toBe("password");
        });

        it("defaults to oidc without runtime config", () => {
            expect(getDirectConnectLoginMethod()).toBe("oidc");
        });
    });
});
