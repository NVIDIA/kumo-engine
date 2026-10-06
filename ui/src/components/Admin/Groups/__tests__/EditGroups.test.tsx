// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter, Route, Routes } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { setAuthClientConfig } from "utils/authRuntimeConfig";

import EditGroup from "../EditGroup";

import groupsResponse from "./MockResponses/Groups.response.json";

import "tests/mockAxios";

describe("EditGroup Page", () => {
    // mockAxios returns group "group-001" / "Engineering Team" for the detail
    // endpoint regardless of the requested id.
    const group = groupsResponse[0];

    const renderPage = async (groupId = group.id) => {
        const queryClient = createQueryClient();
        await act(async () => {
            render(
                <MemoryRouter initialEntries={[`/admin/groups/${groupId}`]}>
                    <QueryClientProvider client={queryClient}>
                        <Routes>
                            <Route
                                path="/admin/groups/:groupId"
                                element={<EditGroup />}
                            />
                        </Routes>
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        setAuthClientConfig(null);
    });

    describe("Layout", () => {
        it("Should render the Edit Group heading and Cancel/Save actions", async () => {
            await renderPage();
            expect(
                await screen.findByRole("heading", { name: "Edit Group" })
            ).toBeInTheDocument();
            expect(
                screen.getByRole("button", { name: "Cancel" })
            ).toBeInTheDocument();
            // The page header "Save" persists group details (id="save").
            expect(document.getElementById("save")).toBeInTheDocument();
        });

        it("Should render the three section headings", async () => {
            await renderPage();
            for (const heading of [
                "Group details",
                "Users",
                "LDAP/SSO groups"
            ]) {
                expect(
                    await screen.findByRole("heading", { name: heading })
                ).toBeInTheDocument();
            }
        });

        it("Should pre-populate the group name from the detail response", async () => {
            await renderPage();
            expect(
                await screen.findByDisplayValue("Engineering Team")
            ).toBeInTheDocument();
        });

        it("Should surface the mapping actions in the LDAP/SSO section", async () => {
            await renderPage();
            expect(
                await screen.findByPlaceholderText("Enter identity provider ID")
            ).toBeInTheDocument();
            expect(
                screen.getByPlaceholderText("Enter external group ID")
            ).toBeInTheDocument();
            expect(
                screen.getByRole("button", { name: "Add mapping" })
            ).toBeInTheDocument();
            expect(
                screen.getByRole("button", { name: "Save LDAP/SSO groups" })
            ).toBeInTheDocument();
        });
    });

    describe("Data hydration", () => {
        // Regression: the LDAP/SSO draft must hydrate from the (async) group
        // detail response, not freeze on the empty first-render placeholder.
        it("Should display existing LDAP/SSO mappings from the server", async () => {
            await renderPage();
            expect(await screen.findByText("engineering")).toBeInTheDocument();
            expect(await screen.findByText("data-science")).toBeInTheDocument();
            expect(screen.getAllByText("corporate-oidc")).toHaveLength(2);
        });

        // Regression: saving a subsection refetches the group detail, which
        // must not overwrite unsaved name/description edits.
        it("Should keep unsaved metadata edits when a subsection saves", async () => {
            await renderPage();
            const nameInput =
                await screen.findByDisplayValue("Engineering Team");
            await userEvent.clear(nameInput);
            await userEvent.type(nameInput, "Renamed Team");

            await userEvent.type(
                screen.getByPlaceholderText("Enter identity provider ID"),
                "okta"
            );
            await userEvent.type(
                screen.getByPlaceholderText("Enter external group ID"),
                "eng-all"
            );
            await userEvent.click(
                screen.getByRole("button", { name: "Add mapping" })
            );
            await userEvent.click(
                screen.getByRole("button", { name: "Save LDAP/SSO groups" })
            );

            await waitFor(() => {
                expect(mockAxiosInstance.put).toHaveBeenCalledWith(
                    "v1/groups/group-001/external-mappings",
                    expect.objectContaining({
                        mappings: expect.arrayContaining([
                            expect.objectContaining({
                                identity_provider_id: "okta",
                                external_group_id: "eng-all"
                            })
                        ])
                    }),
                    expect.anything()
                );
            });
            // The refetch triggered by the save must not reset the field.
            expect(
                screen.getByDisplayValue("Renamed Team")
            ).toBeInTheDocument();
        });
    });
});
