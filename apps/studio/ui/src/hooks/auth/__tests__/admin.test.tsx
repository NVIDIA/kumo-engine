// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";
import { vi } from "vitest";

import { setAdminInfoResponseType } from "pages/Admin/__tests__/MockFunctions/AdminInfo.mock";
import { setOrgMembersResponseType } from "pages/Admin/__tests__/MockFunctions/OrgMembers.mock";

import { ReactQueryKeys } from "utils/reactQueryKeys";

import { useAdminInfo, useAdminMembers } from "../admin";
import { useDirectConnectUsers } from "../directConnectAdmin";

let queryClient: QueryClient;

const createWrapper = (queryClient: QueryClient) => {
    return ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={queryClient}>
            {children}
        </QueryClientProvider>
    );
};

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();

    setAdminInfoResponseType();
    setOrgMembersResponseType();
});

afterEach(() => {
    queryClient.clear();
});

const handleError = vi.fn();
describe("useAdminInfo", () => {
    const renderUseAdminInfo = (handleErrorFn?: (error: unknown) => void) => {
        return renderHook(() => useAdminInfo(handleErrorFn), {
            wrapper: createWrapper(queryClient)
        });
    };

    it("Should return isAdmin as true when user is customer admin", async () => {
        const { result } = renderUseAdminInfo();

        // Initial state
        expect(result.current.isLoading).toBe(true);

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.isAdmin).toBe(true);
        expect(result.current.adminInfo).toEqual({
            user_is_customer_admin: true,
            customer_admin_emails: ["test1@kumo.ai", "test2@kumo.ai"],
            customer_admin_info: {
                api_enabled: true
            },
            organization_name: "test"
        });
    });

    it("Should call handleError when error occurs", async () => {
        setAdminInfoResponseType(testResponseType.ERROR);

        renderUseAdminInfo(handleError);

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });

    it("Should log error to console when no handleError provided", async () => {
        const consoleErrorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});

        setAdminInfoResponseType(testResponseType.ERROR);

        renderUseAdminInfo();

        await waitFor(() => {
            expect(consoleErrorSpy).toHaveBeenCalled();
        });

        consoleErrorSpy.mockRestore();
    });

    it("Should clear admin info cache when clearAdminInfoCache is called", async () => {
        const invalidateQueriesSpy = vi.spyOn(queryClient, "invalidateQueries");

        const { result } = renderUseAdminInfo();

        await act(async () => {
            result.current.clearAdminInfoCache();
        });

        expect(invalidateQueriesSpy).toHaveBeenCalledWith({
            queryKey: [ReactQueryKeys.AdminInfo]
        });
    });
});

describe("useAdminMembers", () => {
    const renderUseAdminMembers = (
        handleErrorFn?: (error: unknown) => void
    ) => {
        return renderHook(() => useAdminMembers(handleErrorFn), {
            wrapper: createWrapper(queryClient)
        });
    };

    it("Should return members when data is available", async () => {
        const { result } = renderUseAdminMembers();

        expect(result.current.isLoading).toBe(true);

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.members).toEqual([
            {
                name: "test",
                email: "test@kumo.ai",
                uid: "test@121254452",
                roles: ["Admin"]
            },
            {
                name: "test2",
                email: "test@manipal.net",
                uid: "test@34353",
                roles: ["Admin"]
            },
            {
                name: "test3",
                email: "test3@manipal.net",
                uid: "test@34353",
                roles: ["User"]
            },
            {
                name: "test4",
                email: "test4@manipal.net",
                uid: "test@3435344",
                roles: ["Admin", "User"]
            },
            {
                name: "test5",
                email: "test5@manipal.net",
                uid: "test@3435355",
                roles: []
            }
        ]);
    });

    it("Should call handleError when error occurs", async () => {
        setOrgMembersResponseType(testResponseType.ERROR);

        renderUseAdminMembers(handleError);

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });

    it("Should log error to console when no handleError provided", async () => {
        const consoleErrorSpy = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});

        setOrgMembersResponseType(testResponseType.ERROR);

        renderUseAdminMembers();

        await waitFor(() => {
            expect(consoleErrorSpy).toHaveBeenCalled();
        });

        consoleErrorSpy.mockRestore();
    });

    it("Should clear admin members cache when clearAdminMembersCache is called", async () => {
        const invalidateQueriesSpy = vi.spyOn(queryClient, "invalidateQueries");

        const { result } = renderUseAdminMembers();

        await act(async () => {
            result.current.clearAdminMembersCache();
        });

        expect(invalidateQueriesSpy).toHaveBeenCalledWith({
            queryKey: [ReactQueryKeys.AdminMembers]
        });
    });
});

describe("useDirectConnectUsers", () => {
    const renderUseDirectConnectUsers = (
        enabled = true,
        handleErrorFn?: (error: unknown) => void
    ) => {
        return renderHook(() => useDirectConnectUsers(enabled, handleErrorFn), {
            wrapper: createWrapper(queryClient)
        });
    };

    it("Should fetch Direct Connect users when enabled", async () => {
        const { result } = renderUseDirectConnectUsers();

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(mockAxiosInstance.get).toHaveBeenCalledWith("v0/admin/users");
        expect(result.current.users).toEqual([
            {
                user_id: "direct_connect|alice",
                email: "alice@example.com",
                display_name: "Alice Example",
                role: "admin",
                disabled: false
            },
            {
                user_id: "direct_connect|bob",
                email: "bob@example.com",
                display_name: "Bob Example",
                role: "user",
                disabled: false
            }
        ]);
    });

    it("Should not fetch Direct Connect users when disabled", async () => {
        const { result } = renderUseDirectConnectUsers(false);

        expect(result.current.users).toEqual([]);
        expect(mockAxiosInstance.get).not.toHaveBeenCalledWith(
            "v0/admin/users"
        );
    });

    it("Should set a Direct Connect user temporary password", async () => {
        const { result } = renderUseDirectConnectUsers();

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        await act(async () => {
            await result.current.setUserPassword(
                "direct_connect|alice",
                "SeedPass12345"
            );
        });

        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
            "v0/admin/users/direct_connect%7Calice/password",
            { password: "SeedPass12345" }
        );
    });

    it("Should create a Direct Connect user with a temporary password", async () => {
        const { result } = renderUseDirectConnectUsers();

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        await act(async () => {
            await result.current.createUser({
                email: "charlie@example.com",
                display_name: "Charlie Example",
                role: "admin",
                disabled: false,
                password: "SeedPass12345"
            });
        });

        expect(mockAxiosInstance.post).toHaveBeenCalledWith("v0/admin/users", {
            email: "charlie@example.com",
            display_name: "Charlie Example",
            role: "admin",
            disabled: false,
            password: "SeedPass12345"
        });
        expect(mockAxiosInstance.post).not.toHaveBeenCalledWith(
            "v0/admin/users/direct_connect%7Ccharlie/password",
            { password: "SeedPass12345" }
        );
    });

    it("Should patch encoded user ids and invalidate Direct Connect caches", async () => {
        const invalidateQueriesSpy = vi.spyOn(queryClient, "invalidateQueries");
        const { result } = renderUseDirectConnectUsers();

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        await act(async () => {
            await result.current.updateUsers([
                {
                    user_id: "direct_connect|alice",
                    role: "user",
                    disabled: true
                }
            ]);
        });

        expect(mockAxiosInstance.patch).toHaveBeenCalledWith(
            "v0/admin/users/direct_connect%7Calice",
            { role: "user", disabled: true },
            expect.objectContaining({
                headers: { "Content-Type": "application/json" }
            })
        );
        expect(invalidateQueriesSpy).toHaveBeenCalledWith({
            queryKey: [ReactQueryKeys.AdminMembers, "directConnect"]
        });
        expect(invalidateQueriesSpy).toHaveBeenCalledWith({
            queryKey: [ReactQueryKeys.AdminMembers]
        });
        // A role change can demote the current user, so the admin status
        // consumed by AdminRouteGuard and the sidebar must refetch too.
        expect(invalidateQueriesSpy).toHaveBeenCalledWith({
            queryKey: [ReactQueryKeys.AdminInfo]
        });
    });
});
