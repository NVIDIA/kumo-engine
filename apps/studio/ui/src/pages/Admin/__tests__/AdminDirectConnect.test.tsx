// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import {
    fireEvent,
    render,
    screen,
    waitFor,
    within
} from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import DirectConnectUsersAdmin from "components/Admin/DirectConnectUsersAdmin";

import { setAuthClientConfig } from "utils/authRuntimeConfig";
import { isDirectConnectPasswordLogin } from "utils/directConnectAuth";

import { setAdminInfoResponseType } from "./MockFunctions/AdminInfo.mock";
import { setOrgMembersResponseType } from "./MockFunctions/OrgMembers.mock";

const setDirectConnectEnv = (loginMethod?: "password" | "oidc") => {
    setAuthClientConfig({
        auth_disabled: false,
        direct_connect_login_method: loginMethod ?? "oidc"
    });
};

const resetDirectConnectEnv = () => {
    setAuthClientConfig(null);
};

describe("Admin Direct Connect users", () => {
    const queryClient = createQueryClient();
    const renderPage = async () => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <DirectConnectUsersAdmin
                            isPasswordLogin={isDirectConnectPasswordLogin()}
                            handleError={vi.fn()}
                            setLoading={vi.fn()}
                        />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
        resetDirectConnectEnv();
        setAdminInfoResponseType(testResponseType.ALL);
        setOrgMembersResponseType(testResponseType.ALL);
        setDirectConnectEnv();
    });

    it("Should render users and save role/status updates", async () => {
        await renderPage();

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                "v0/admin/users"
            );
        });
        expect(
            await screen.findByRole("button", { name: "Add user" })
        ).toBeInTheDocument();
        const searchInput = screen.getByPlaceholderText("Search");
        expect(searchInput).toBeInTheDocument();
        expect(await screen.findByText("Alice Example")).toBeInTheDocument();
        expect(await screen.findByText("bob@example.com")).toBeInTheDocument();
        fireEvent.change(searchInput, { target: { value: "bob" } });
        expect(screen.queryByText("Alice Example")).not.toBeInTheDocument();
        expect(screen.getByText("bob@example.com")).toBeInTheDocument();
        fireEvent.change(searchInput, { target: { value: "" } });
        expect(await screen.findByText("Alice Example")).toBeInTheDocument();
        expect(
            screen.queryByText("Organization Members")
        ).not.toBeInTheDocument();
        expect(screen.queryByText("API Keys")).not.toBeInTheDocument();

        fireEvent.click(
            screen.getByTestId(
                "testid-direct-connect-user-role-direct_connect|alice"
            )
        );
        const roleListbox = await screen.findByRole("listbox");
        fireEvent.click(
            await within(roleListbox).findByRole("option", { name: "User" })
        );

        const saveButton = await screen.findByRole("button", {
            name: "Save"
        });
        expect(saveButton).toBeEnabled();
        fireEvent.click(saveButton);

        await waitFor(() => {
            expect(mockAxiosInstance.patch).toHaveBeenCalledWith(
                "v0/admin/users/direct_connect%7Calice",
                { role: "user" },
                expect.objectContaining({
                    headers: { "Content-Type": "application/json" }
                })
            );
        });
    });

    it("Should block weak temporary passwords", async () => {
        setDirectConnectEnv("password");
        await renderPage();

        const setPasswordButtons = await screen.findAllByRole("button", {
            name: "Set password"
        });
        fireEvent.click(setPasswordButtons[0]);
        fireEvent.change(await screen.findByLabelText("Temporary password"), {
            target: { value: "short1" }
        });

        expect(
            await screen.findByText("Password must be at least 12 characters")
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: "Save password" })
        ).toBeDisabled();
        expect(mockAxiosInstance.post).not.toHaveBeenCalledWith(
            "v0/admin/users/direct_connect%7Calice/password",
            { password: "short1" }
        );
    });

    it("Should set temporary passwords in password mode", async () => {
        setDirectConnectEnv("password");
        await renderPage();

        const setPasswordButtons = await screen.findAllByRole("button", {
            name: "Set password"
        });
        fireEvent.click(setPasswordButtons[0]);
        fireEvent.change(await screen.findByLabelText("Temporary password"), {
            target: { value: "SeedPass12345" }
        });
        fireEvent.click(
            await screen.findByRole("button", { name: "Save password" })
        );

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                "v0/admin/users/direct_connect%7Calice/password",
                { password: "SeedPass12345" }
            );
        });
    });

    it("Should create users in password mode", async () => {
        setDirectConnectEnv("password");
        await renderPage();

        fireEvent.click(
            await screen.findByRole("button", { name: "Add user" })
        );
        fireEvent.change(await screen.findByLabelText("Email"), {
            target: { value: "charlie@example.com" }
        });
        fireEvent.change(await screen.findByLabelText("Display name"), {
            target: { value: "Charlie Example" }
        });
        fireEvent.click(await screen.findByLabelText("Role"));
        const roleListbox = await screen.findByRole("listbox");
        expect(
            screen.queryByPlaceholderText("Search...")
        ).not.toBeInTheDocument();
        fireEvent.click(
            await within(roleListbox).findByRole("option", { name: "Admin" })
        );
        fireEvent.change(await screen.findByLabelText("Temporary password"), {
            target: { value: "SeedPass12345" }
        });
        fireEvent.click(
            await screen.findByRole("button", { name: "Create user" })
        );

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                "v0/admin/users",
                {
                    email: "charlie@example.com",
                    display_name: "Charlie Example",
                    role: "admin",
                    disabled: false,
                    password: "SeedPass12345"
                }
            );
            expect(mockAxiosInstance.post).not.toHaveBeenCalledWith(
                "v0/admin/users/direct_connect%7Ccharlie/password",
                { password: "SeedPass12345" }
            );
        });
    });
});
