// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClient } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import { setDeploymentType } from "stores/__tests__/MockFunctions/DebugInfo.mock";
import { useUserStore } from "stores/userStore";
import { mockLocationStateValue, mockNavigateCall } from "tests/mockRouter";
import { deploymentType } from "tests/testType";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { storageKeys } from "utils/Constants";

import { useSidebarData } from "../useSidebarData";

import "tests/mockAxios";

let queryClient: QueryClient;

beforeEach(async () => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    mockNavigateCall();

    setDeploymentType(deploymentType.SAAS);
    mockLocationStateValue(null, "/models/");
    storeToLocalStorage(storageKeys.userDetails, {
        first_name: "John",
        last_name: "Doe",
        email: "john.doe@example.com",
        picture: "http://example.com/pic.jpg",
        user_id: "",
        name: "",
        isLoggedIn: false
    });
    useUserStore.getInitialState().initializeUser();
});

afterEach(() => {
    queryClient.clear();
});

describe("useSidebarData", () => {
    describe("user profile", () => {
        it("Should return user profile with full name when first_name and last_name exist", () => {
            const { result } = renderHook(
                () => useSidebarData({ hideMenu: false }),
                { wrapper: createWrapper(queryClient) }
            );
            expect(result.current.userProfile.name).toBe("John Doe");
            expect(result.current.userProfile.email).toBe(
                "john.doe@example.com"
            );
            expect(result.current.userProfile.avatar).toBe(
                "http://example.com/pic.jpg"
            );
            expect(result.current.userProfile.show).toBe(true);
        });
    });

    describe("navigation menu", () => {
        it("Should return empty navMainMenuGroup when hideMenu is true", () => {
            const { result } = renderHook(
                () => useSidebarData({ hideMenu: true }),
                { wrapper: createWrapper(queryClient) }
            );

            expect(result.current.navMainMenuGroup).toEqual([]);
            expect(result.current.navQuickStart).toEqual([]);
        });

        it("Should generate navMainMenuGroup with all items when hideMenu is false", () => {
            const { result } = renderHook(
                () => useSidebarData({ hideMenu: false }),
                { wrapper: createWrapper(queryClient) }
            );

            expect(result.current.navMainMenuGroup).toHaveLength(3);
            expect(result.current.navMainMenuGroup[0].items).toHaveLength(2);
            expect(result.current.navMainMenuGroup[0].items[0].title).toBe(
                "Models"
            );
            expect(result.current.navMainMenuGroup[0].items[1].title).toBe(
                "Predictions"
            );
        });

        it.each([
            {
                pathname: "/project/project-tests/models",
                expectedActive: "Models"
            },
            {
                pathname: "/project/project-tests/predictions",
                expectedActive: "Predictions"
            },
            {
                pathname: "/project/project-tests/graphs",
                expectedActive: "Graphs"
            },
            {
                pathname: "/project/project-tests/tables",
                expectedActive: "Tables"
            },
            {
                pathname: "/project/project-tests/connectors",
                expectedActive: "Connectors"
            },
            { pathname: "/admin", expectedActive: "Admin" }
        ])(
            "Should mark $expectedActive as active when pathname is $pathname",
            ({ pathname, expectedActive }) => {
                mockLocationStateValue(null, pathname);

                const { result } = renderHook(
                    () => useSidebarData({ hideMenu: false }),
                    { wrapper: createWrapper(queryClient) }
                );

                const allItems = result.current.navMainMenuGroup.flatMap(
                    (group) => group.items
                );
                const activeItem = allItems.find((item) => item.isActive);

                expect(activeItem?.title).toBe(expectedActive);
            }
        );
    });

    describe("admin visibility", () => {
        it("Should show admin menu item when user is admin", () => {
            const { result } = renderHook(
                () => useSidebarData({ hideMenu: false, isAdmin: true }),
                { wrapper: createWrapper(queryClient) }
            );

            const adminItem = result.current.navMainMenuGroup[2].items.find(
                (item) => item.title === "Admin"
            );
            expect(adminItem).toBeDefined();
            expect(adminItem?.to).toBe("/admin/members");
        });
    });

    describe("quick start menu", () => {
        it("Should return empty navQuickStart when hideMenu is true", () => {
            const { result } = renderHook(
                () => useSidebarData({ hideMenu: true }),
                { wrapper: createWrapper(queryClient) }
            );

            expect(result.current.navQuickStart).toEqual([]);
        });

        it("Should generate navQuickStart with New", () => {
            const { result } = renderHook(
                () => useSidebarData({ hideMenu: false }),
                { wrapper: createWrapper(queryClient) }
            );

            expect(result.current.navQuickStart).toHaveLength(1);
            expect(result.current.navQuickStart[0].title).toBe("New");
            expect(result.current.navQuickStart[0].childs).toHaveLength(2);
        });
    });
});
