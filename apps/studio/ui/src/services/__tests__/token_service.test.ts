// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    getFromLocalStorage,
    storeToLocalStorage
} from "@kumo-ai/kumo-ui-catalog";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { storageKeys } from "utils/Constants";

import tokenService from "../token_service";

describe("TokenService", () => {
    beforeEach(() => {
        localStorage.clear();
        tokenService.accessToken = "";
        tokenService.ssoFlow = false;
    });

    afterEach(() => {
        localStorage.clear();
    });

    describe("initialization", () => {
        it("Should have default ssoFlow as false", () => {
            expect(tokenService.ssoFlow).toBe(false);
        });

        it("Should load access token from localStorage on initialization", () => {
            storeToLocalStorage(storageKeys.accessToken, "stored-token", true);

            const token = getFromLocalStorage(storageKeys.accessToken, true);

            expect(token).toBe("stored-token");
        });
    });

    describe("setToken", () => {
        it("Should set access token and refresh token in localStorage for non-SSO flow", () => {
            tokenService.setToken("new-access-token", "new-refresh-token");

            expect(tokenService.accessToken).toBe("new-access-token");
            expect(getFromLocalStorage(storageKeys.accessToken, true)).toBe(
                "new-access-token"
            );
            expect(getFromLocalStorage(storageKeys.refreshToken, true)).toBe(
                "new-refresh-token"
            );
        });

        it("Should only set access token in localStorage for SSO flow", () => {
            tokenService.setSSOFlow();

            tokenService.setToken("sso-access-token", "sso-refresh-token");

            expect(tokenService.accessToken).toBe("sso-access-token");
            expect(getFromLocalStorage(storageKeys.accessToken, true)).toBe(
                "sso-access-token"
            );
        });

        it("Should handle undefined refresh token", () => {
            tokenService.setToken("access-token-only");

            expect(tokenService.accessToken).toBe("access-token-only");
            expect(getFromLocalStorage(storageKeys.accessToken, true)).toBe(
                "access-token-only"
            );
        });
    });

    describe("setSSOFlow", () => {
        it("Should set SSO flow to true", () => {
            tokenService.setSSOFlow();

            expect(tokenService.ssoFlow).toBe(true);
        });
    });

    describe("getSSOFlow", () => {
        it("Should return true when ssoFlow is true", () => {
            tokenService.setSSOFlow();

            expect(tokenService.getSSOFlow()).toBe(true);
        });

        it("Should return true when accessToken exists but refreshToken does not", () => {
            storeToLocalStorage(storageKeys.accessToken, "access-token", true);

            expect(tokenService.getSSOFlow()).toBe(true);
        });

        it("Should return false when both tokens exist", () => {
            storeToLocalStorage(storageKeys.accessToken, "access-token", true);
            storeToLocalStorage(
                storageKeys.refreshToken,
                "refresh-token",
                true
            );

            expect(tokenService.getSSOFlow()).toBe(false);
        });
    });

    describe("clear", () => {
        it("Should clear access token and remove tokens from localStorage", () => {
            tokenService.setToken("access-token", "refresh-token");

            tokenService.clear();

            expect(tokenService.accessToken).toBe("");
            expect(
                getFromLocalStorage(storageKeys.accessToken, true)
            ).toBeNull();
            expect(
                getFromLocalStorage(storageKeys.refreshToken, true)
            ).toBeNull();
        });
    });

    describe("getAccessToken", () => {
        it.each([
            {
                scenario: "SSO flow",
                setup: () => {
                    tokenService.accessToken = "sso-access-token";
                    tokenService.setSSOFlow();
                },
                expected: "sso-access-token"
            },
            {
                scenario: "valid refresh token",
                setup: () =>
                    tokenService.setToken(
                        "access-token",
                        "valid-refresh-token"
                    ),
                expected: "access-token"
            },
            {
                scenario: "invalid refresh token",
                setup: () => {
                    tokenService.accessToken = "access-token";
                },
                expected: "access-token"
            }
        ])(
            "Should return access token for $scenario",
            ({ setup, expected }) => {
                setup();

                const result = tokenService.getAccessToken();

                expect(result).toBe(expected);
            }
        );
    });

    describe("getRefreshToken", () => {
        it("Should return refresh token when valid", () => {
            tokenService.setToken("access", "valid-refresh-token");

            const result = tokenService.getRefreshToken();

            expect(result).toBe("valid-refresh-token");
        });

        it.each([
            {
                scenario: "token not in localStorage",
                setup: () => {}
            },
            {
                scenario: "token is empty string",
                setup: () =>
                    storeToLocalStorage(storageKeys.refreshToken, "", true)
            }
        ])("Should return empty string when $scenario", ({ setup }) => {
            setup();

            const result = tokenService.getRefreshToken();

            expect(result).toBe("");
        });
    });

    describe("integration: full auth flow", () => {
        it("Should handle complete login and logout flow", () => {
            tokenService.setToken("access-123", "refresh-456");

            expect(tokenService.getAccessToken()).toBe("access-123");
            expect(tokenService.getRefreshToken()).toBe("refresh-456");
            expect(tokenService.getSSOFlow()).toBe(false);

            tokenService.clear();

            expect(tokenService.getAccessToken()).toBe("");
            expect(tokenService.getRefreshToken()).toBe("");
        });

        it("Should handle SSO login flow", () => {
            tokenService.setSSOFlow();
            tokenService.setToken("sso-access-token");

            expect(tokenService.getAccessToken()).toBe("sso-access-token");
            expect(tokenService.getSSOFlow()).toBe(true);
        });

        it("Should persist tokens in localStorage", () => {
            tokenService.setToken("persist-access", "persist-refresh");

            expect(getFromLocalStorage(storageKeys.accessToken, true)).toBe(
                "persist-access"
            );
            expect(getFromLocalStorage(storageKeys.refreshToken, true)).toBe(
                "persist-refresh"
            );
        });
    });
});
