// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import axios, { AxiosError, AxiosRequestConfig } from "axios";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setAuthClientConfig } from "../../utils/authRuntimeConfig";
import {
    storageKeys,
    UNEXPECTED_RESPONSE_MESSAGE
} from "../../utils/Constants";

import { useUserStore } from "../../stores/userStore";
import axiosClient, {
    getBaseURL,
    NETWORK_ERROR_MESSAGE,
    setDirectConnectUnauthorizedHandler
} from "../axios_client";

vi.unmock("services/axios_client");

let mockAdapter: ReturnType<typeof vi.fn>;

const loggedInUser = {
    isLoggedIn: true,
    name: "testuser",
    email: "test@example.com",
    first_name: "Test",
    last_name: "User",
    picture: "",
    user_id: "test-user-id"
};

const loggedOutUser = {
    isLoggedIn: false,
    name: "",
    email: "",
    first_name: "",
    last_name: "",
    picture: "",
    user_id: ""
};

const setUserState = (
    userDetails: typeof loggedInUser | typeof loggedOutUser
) => {
    useUserStore.setState({ userDetails });
};

const getRequestConfig = () => mockAdapter.mock.calls[0][0];

describe("axios_client", () => {
    beforeEach(() => {
        localStorage.clear();
        sessionStorage.clear();
        setAuthClientConfig(null);

        mockAdapter = vi.fn((config: AxiosRequestConfig) => {
            return Promise.resolve({
                data: { success: true },
                status: 200,
                statusText: "OK",
                headers: {},
                config
            });
        });
        axiosClient.defaults.adapter = mockAdapter as never;
    });

    afterEach(() => {
        setAuthClientConfig(null);
        setDirectConnectUnauthorizedHandler(null);
        vi.clearAllMocks();
        localStorage.clear();
        sessionStorage.clear();
    });

    it("Should return test URL in test environment", () => {
        expect(getBaseURL()).toBe("http://localhost:10001");
    });

    it("Should define network error message correctly", () => {
        expect(NETWORK_ERROR_MESSAGE).toContain(
            "Could not connect to the Kumo API server"
        );
        expect(NETWORK_ERROR_MESSAGE).toContain("internet connection");
        expect(NETWORK_ERROR_MESSAGE).toContain("ad-blockers");
    });

    describe("axiosClient configuration", () => {
        it.each([
            ["get", "function"],
            ["post", "function"]
        ])("Should have %s method of type %s", (method, type) => {
            expect(typeof axiosClient[method as keyof typeof axiosClient]).toBe(
                type
            );
        });

        it("Should have correct default headers", () => {
            expect(axiosClient.defaults.headers["Content-type"]).toBe(
                "application/json"
            );
        });

        it("Should have baseURL configured", () => {
            expect(axiosClient.defaults.baseURL).toBe("http://localhost:10001");
        });

        it.each([["request"], ["response"]])(
            "Should have %s interceptor configured",
            (interceptorType) => {
                expect(
                    axiosClient.interceptors[
                        interceptorType as "request" | "response"
                    ]
                ).toBeDefined();
            }
        );
    });

    describe("request interceptor - Direct Connect credentials", () => {
        it("Should use cookie credentials without bearer header when user is logged in", async () => {
            setUserState(loggedInUser);

            await axiosClient.get("/api/test");

            const config = getRequestConfig();
            expect(config.withCredentials).toBe(true);
            expect(config.headers?.Authorization).toBeUndefined();
        });

        it("Should use cookie credentials without bearer header when user is not logged in", async () => {
            setUserState(loggedOutUser);

            await axiosClient.get("/api/test");

            const config = getRequestConfig();
            expect(config.withCredentials).toBe(true);
            expect(config.headers?.Authorization).toBeUndefined();
        });

        it("Should not send cookie credentials when auth is disabled via runtime config", async () => {
            setAuthClientConfig({
                auth_disabled: true,
                direct_connect_login_method: null
            });

            await axiosClient.get("/api/test");

            const config = getRequestConfig();
            expect(config.withCredentials).toBe(false);
        });

        it("Should not send cookie credentials when auth is disabled via window env", async () => {
            (window as unknown as { _env_?: object })._env_ = {
                AUTH_DISABLED: true
            };
            try {
                await axiosClient.get("/api/test");

                const config = getRequestConfig();
                expect(config.withCredentials).toBe(false);
            } finally {
                delete (window as unknown as { _env_?: object })._env_;
            }
        });
    });

    describe("request interceptor - profile mode", () => {
        it.each([
            ["ALL", "/api/test", true],
            ["v0/", "v0/users", true],
            ["v1/", "v0/users", false]
        ])(
            "Should handle profile mode %s for URL %s",
            async (profileMode, url, shouldAddProfile) => {
                sessionStorage.setItem(
                    "PROFILE_MODE",
                    JSON.stringify(profileMode)
                );

                await axiosClient.get(url);

                const config = getRequestConfig();
                if (shouldAddProfile) {
                    expect(config.params?.profile).toBe("true");
                } else {
                    expect(config.params?.profile).toBeUndefined();
                }
            }
        );

        it("Should preserve existing params when adding profile", async () => {
            sessionStorage.setItem("PROFILE_MODE", JSON.stringify("ALL"));

            await axiosClient.get("/api/test", { params: { page: 1 } });

            const config = getRequestConfig();
            expect(config.params?.profile).toBe("true");
            expect(config.params?.page).toBe(1);
        });
    });

    describe("response interceptor - success", () => {
        it("Should return response on success", async () => {
            const response = await axiosClient.get("/api/test");

            expect(response.data).toEqual({ success: true });
            expect(response.status).toBe(200);
        });
    });

    describe("response interceptor - cancel handling", () => {
        it("Should reject cancelled requests", async () => {
            const cancelError = new axios.Cancel("Request cancelled");
            mockAdapter.mockRejectedValueOnce(cancelError);

            await expect(axiosClient.get("/api/test")).rejects.toThrow();
        });
    });

    describe("response interceptor - 401 handling", () => {
        const create401Error = (url = "/api/data"): AxiosError =>
            ({
                config: { url },
                response: { status: 401 },
                message: "Unauthorized",
                isAxiosError: true
            }) as AxiosError;

        beforeEach(() => {
            Object.defineProperty(window, "location", {
                value: {
                    pathname: "/dashboard",
                    search: "?tab=overview"
                },
                writable: true,
                configurable: true
            });
        });

        it("Should redirect to login on 401", async () => {
            const assign = vi.fn();
            const directConnectUnauthorizedHandler = vi.fn();
            Object.defineProperty(window, "location", {
                value: {
                    assign,
                    pathname: "/dashboard",
                    search: "?tab=overview"
                },
                writable: true,
                configurable: true
            });
            setDirectConnectUnauthorizedHandler(
                directConnectUnauthorizedHandler
            );
            setUserState(loggedInUser);
            mockAdapter.mockRejectedValueOnce(create401Error());

            await expect(axiosClient.get("/api/data")).rejects.toThrow();

            expect(directConnectUnauthorizedHandler).toHaveBeenCalled();
            expect(assign).toHaveBeenCalledWith("/login");
            expect(
                JSON.parse(localStorage.getItem(storageKeys.navPath) as string)
            ).toBe("/dashboard?tab=overview");
        });

        it("Should not store path when on public path", async () => {
            const assign = vi.fn();
            Object.defineProperty(window, "location", {
                value: {
                    assign,
                    pathname: "login",
                    search: ""
                },
                writable: true,
                configurable: true
            });

            setUserState(loggedInUser);
            mockAdapter.mockRejectedValue(create401Error());

            await expect(axiosClient.get("/api/data")).rejects.toThrow();

            expect(localStorage.getItem(storageKeys.navPath)).toBeNull();
            expect(assign).toHaveBeenCalledWith("/login");
        });

        it("Should not handle 401 for authentication endpoints", async () => {
            setUserState(loggedInUser);
            mockAdapter.mockRejectedValueOnce(
                create401Error("v0/authentication/login")
            );

            await expect(
                axiosClient.post("v0/authentication/login", {
                    username: "test",
                    password: "wrong"
                })
            ).rejects.toThrow();
        });
    });

    describe("response interceptor - Network Error", () => {
        it("Should replace Network Error message", async () => {
            mockAdapter.mockRejectedValueOnce(new Error("Network Error"));

            await expect(axiosClient.get("/api/test")).rejects.toThrow(
                NETWORK_ERROR_MESSAGE
            );
        });

        it("Should not modify other error messages", async () => {
            const error = {
                config: { url: "/api/test" },
                response: { status: 500 },
                message: "Internal Server Error",
                isAxiosError: true
            } as AxiosError;

            mockAdapter.mockRejectedValueOnce(error);

            await expect(axiosClient.get("/api/test")).rejects.toThrow(
                "Internal Server Error"
            );
        });
    });

    describe("response interceptor - HTML error sanitization", () => {
        it.each([
            {
                description: "Should sanitize DOCTYPE HTML error",
                status: 503,
                html: "<!DOCTYPE html><html><body>503 Service Unavailable</body></html>"
            },
            {
                description: "Should sanitize DOCTYPE with leading whitespace",
                status: 503,
                html: "  <!DOCTYPE html><html><body>Error</body></html>"
            },
            {
                // nginx gateway pages start with a bare <html>, not <!DOCTYPE -
                // this is the exact body that leaked in #30031.
                description:
                    "Should sanitize bare <html> nginx 504 gateway page",
                status: 504,
                html: "<html>\n<head><title>504 Gateway Time-out</title></head>\n<body><center><h1>504 Gateway Time-out</h1></center></body>\n</html>"
            },
            {
                description:
                    "Should sanitize bare <html> with leading whitespace",
                status: 502,
                html: "  \n<html><body>502 Bad Gateway</body></html>"
            }
        ])("$description", async ({ status, html }) => {
            const error = {
                config: { url: "/api/test" },
                response: { status, data: html },
                message: "Gateway error",
                isAxiosError: true
            } as AxiosError;

            mockAdapter.mockRejectedValueOnce(error);

            try {
                await axiosClient.get("/api/test");
                throw new Error("expected request to reject");
            } catch (err) {
                const data = (err as AxiosError).response?.data;
                expect(data).toBe(UNEXPECTED_RESPONSE_MESSAGE);
                expect(data).not.toContain("<");
            }
        });

        it.each([
            {
                description: "Should not sanitize body tag without DOCTYPE",
                data: "<body>Proxy Error</body>"
            },
            {
                description: "Should not sanitize non-HTML string errors",
                data: "Invalid request parameters"
            }
        ])("$description", async ({ data }) => {
            const error = {
                config: { url: "/api/test" },
                response: { status: 400, data },
                message: "Bad Request",
                isAxiosError: true
            } as AxiosError;

            mockAdapter.mockRejectedValueOnce(error);

            try {
                await axiosClient.get("/api/test");
            } catch (err) {
                expect((err as AxiosError).response?.data).toBe(data);
            }
        });

        it("Should not sanitize object errors", async () => {
            const errorData = { error: "Something went wrong" };
            const error = {
                config: { url: "/api/test" },
                response: { status: 500, data: errorData },
                message: "Internal Server Error",
                isAxiosError: true
            } as AxiosError;

            mockAdapter.mockRejectedValueOnce(error);

            try {
                await axiosClient.get("/api/test");
            } catch (err) {
                expect((err as AxiosError).response?.data).toEqual(errorData);
            }
        });
    });

    it("Should handle Direct Connect request with profile mode", async () => {
        setUserState(loggedInUser);
        sessionStorage.setItem("PROFILE_MODE", JSON.stringify("ALL"));

        mockAdapter.mockResolvedValueOnce({
            data: { users: [] },
            status: 200,
            statusText: "OK",
            headers: {},
            config: {}
        });

        const response = await axiosClient.get("/api/users");

        const config = getRequestConfig();
        expect(config.headers?.Authorization).toBeUndefined();
        expect(config.withCredentials).toBe(true);
        expect(config.params?.profile).toBe("true");
        expect(response.data).toEqual({ users: [] });
    });
});
