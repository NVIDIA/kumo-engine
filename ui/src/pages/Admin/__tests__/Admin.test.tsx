// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { verifyTextExists } from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";
import { describe, expect, it, vi } from "vitest";

import { setAuthClientConfig } from "utils/authRuntimeConfig";

import Admin from "../Admin";

import { setAdminInfoResponseType } from "./MockFunctions/AdminInfo.mock";

describe("Admin Component", () => {
    const queryClient = createQueryClient();
    const renderPage = async () => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <Admin />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };
    const pageLoadTest = async () => {
        await waitFor(() => {
            const APIS_TO_CALL = ["v0/admin/info", "v0/admin/users"];
            APIS_TO_CALL.forEach((API) => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(`${API}`);
            });
        });
    };

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
        setAuthClientConfig({
            auth_disabled: false,
            direct_connect_login_method: "oidc"
        });
        setAdminInfoResponseType(testResponseType.ALL);
    });

    it("Should render Direct Connect users in the shared admin layout", async () => {
        await renderPage();
        await pageLoadTest();

        expect(
            await screen.findByRole("heading", { name: "Users" })
        ).toBeInTheDocument();
        expect(screen.getByTestId("admin-page")).toHaveAttribute(
            "data-admin-page-title",
            "Users"
        );
        expect(
            await screen.findByRole("button", { name: "Add user" })
        ).toBeInTheDocument();
        expect(
            screen.queryByText(
                "Manage organization members and your own API keys."
            )
        ).not.toBeInTheDocument();
    });

    it.each([
        {
            type: testResponseType.INVALID,
            description: "invalid admin info responses",
            expectedError: "Failed to get Admin Info"
        },
        {
            type: testResponseType.ERROR,
            description: "error admin info responses",
            expectedError: "Something went wrong!"
        },
        {
            type: testResponseType.EMPTY,
            description: "empty admin info responses",
            expectedError: null
        }
    ])(
        "Should show correct UI for $description",
        async ({ type, expectedError }) => {
            setAdminInfoResponseType(type);
            await renderPage();
            await pageLoadTest();

            await verifyTextExists(/Users/);

            if (expectedError) {
                await verifyTextExists(expectedError);
            }
        }
    );
});
