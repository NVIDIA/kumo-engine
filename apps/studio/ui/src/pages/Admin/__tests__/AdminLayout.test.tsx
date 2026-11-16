// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
    mockLocationStateValue,
    mockNavigateCall,
    mockUseNavigate
} from "tests/mockRouter";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { navRoutes } from "utils/Constants";

import AdminLayout from "../AdminLayout";

describe("AdminLayout", () => {
    const renderLayout = async (pathname = "") => {
        mockLocationStateValue(null, pathname);
        await act(async () => {
            render(
                <MemoryRouter>
                    <AdminLayout />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        mockNavigateCall();
    });

    it.each([
        { label: "Admin heading", query: () => screen.getByText("Admin") },
        {
            label: "Back button",
            query: () => screen.getByRole("button", { name: /back/i })
        }
    ])("Should render $label", async ({ query }) => {
        await renderLayout();
        expect(query()).toBeInTheDocument();
    });

    it.each([{ title: "Users" }, { title: "Groups" }, { title: "Identities" }])(
        "Should render '$title' nav item",
        async ({ title }) => {
            await renderLayout();
            expect(
                screen.getByRole("button", { name: title })
            ).toBeInTheDocument();
        }
    );

    it("Should not render an 'API Keys' admin nav item", async () => {
        await renderLayout();
        expect(
            screen.queryByRole("button", { name: "API Keys" })
        ).not.toBeInTheDocument();
    });

    it("Should navigate to Connectors when Back button is clicked", async () => {
        await renderLayout();
        await userEvent.click(screen.getByRole("button", { name: /back/i }));
        expect(mockUseNavigate).toHaveBeenCalledWith(navRoutes.Home);
    });

    it.each([
        { title: "Users", path: navRoutes.AdminMembers },
        { title: "Groups", path: navRoutes.AdminGroups },
        { title: "Identities", path: navRoutes.AdminIdentitySettings }
    ])(
        "Should navigate to $path when '$title' nav item is clicked",
        async ({ title, path }) => {
            await renderLayout();
            await userEvent.click(screen.getByRole("button", { name: title }));
            expect(mockUseNavigate).toHaveBeenCalledWith(path);
        }
    );

    it.each([
        {
            title: "Users",
            path: navRoutes.AdminMembers,
            inactiveTitle: "Identities"
        },
        {
            title: "Groups",
            path: navRoutes.AdminGroups,
            inactiveTitle: "Identities"
        },
        {
            title: "Identities",
            path: navRoutes.AdminIdentitySettings,
            inactiveTitle: "Groups"
        }
    ])(
        "Should mark '$title' as active and '$inactiveTitle' as inactive when on its path",
        async ({ title, path, inactiveTitle }) => {
            await renderLayout(path);
            expect(screen.getByRole("button", { name: title })).toHaveAttribute(
                "data-active",
                "true"
            );
            expect(
                screen.getByRole("button", { name: inactiveTitle })
            ).toHaveAttribute("data-active", "false");
        }
    );
});
