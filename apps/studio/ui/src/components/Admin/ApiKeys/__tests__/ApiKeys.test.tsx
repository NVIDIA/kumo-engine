// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useDebugStore } from "stores/debugStore";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import { findAndClickButtonByRole, getDialog } from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import {
    mockAdminApiKeysList,
    setAdminApiKeysResponseType,
    setCreateAdminApiKeyResponseType,
    setDeleteAdminApiKeyResponseType
} from "hooks/apiKeys/__tests__/MockFunctions/AdminApiKeys.mock";
import {
    mockCreateUserApiKeyResponse,
    mockUserApiKeysList,
    setCreateUserApiKeyResponseType,
    setDeleteUserApiKeyResponseType,
    setUserApiKeysResponseType
} from "hooks/apiKeys/__tests__/MockFunctions/UserApiKeys.mock";

import {
    setAdminInfoResponseType,
    setMockAdminUserIsAdmin
} from "pages/Admin/__tests__/MockFunctions/AdminInfo.mock";

import ApiKeys from "../ApiKeys";

import "tests/mockAxios";

describe("ApiKeys", () => {
    // The single /api-keys module is admin-aware at runtime via useAdminInfo();
    // the test drives that through the mocked /v0/admin/info response.
    const renderApiKeys = async (isAdmin = false) => {
        setMockAdminUserIsAdmin(isAdmin);
        const queryClient = createQueryClient();
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <ApiKeys />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const openDeleteDialog = async () => {
        await renderApiKeys();
        await waitFor(() => {
            expect(
                screen.getByText(mockUserApiKeysList[0].name)
            ).toBeInTheDocument();
        });
        const deleteButtons = screen.getAllByRole("button", {
            name: "Delete"
        });
        await userEvent.click(deleteButtons[0]);
        await waitFor(() => {
            getDialog();
        });
    };

    const openCreateDialog = async () => {
        await renderApiKeys();
        await findAndClickButtonByRole("Create API key");
        await waitFor(() => {
            getDialog();
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        useDebugStore.getState().reset();
        useUserStore.getState().setUserDetails({ isLoggedIn: true });
        setMockAdminUserIsAdmin(false);
        setAdminInfoResponseType();
        setAdminApiKeysResponseType();
        setCreateAdminApiKeyResponseType();
        setDeleteAdminApiKeyResponseType();
        setUserApiKeysResponseType();
        setCreateUserApiKeyResponseType();
        setDeleteUserApiKeyResponseType();
    });

    it("Should filter the table based on search", async () => {
        await renderApiKeys();
        await waitFor(() => {
            expect(
                screen.getByText(mockUserApiKeysList[0].name)
            ).toBeInTheDocument();
        });

        const searchBar = screen.getByPlaceholderText("Search");
        await userEvent.type(searchBar, "nonexistent-key-name");

        await waitFor(() => {
            expect(
                screen.queryByText(mockUserApiKeysList[0].name)
            ).not.toBeInTheDocument();
        });
    });

    it("Should surface an admin-info fetch failure in the page validation UI", async () => {
        setAdminInfoResponseType(testResponseType.ERROR);
        await renderApiKeys();

        await waitFor(() => {
            expect(
                screen.getByText("Something went wrong!")
            ).toBeInTheDocument();
        });
    });

    describe("Non-admin user", () => {
        it("Should not show the Admin API keys tab", async () => {
            await renderApiKeys();

            expect(screen.getByText("API Keys")).toBeInTheDocument();
            expect(screen.getByTestId("admin-page")).toHaveAttribute(
                "data-admin-page-title",
                "API Keys"
            );
            expect(
                screen.getByTestId("admin-search-container")
            ).toBeInTheDocument();

            expect(screen.getByPlaceholderText("Search")).toBeInTheDocument();
            expect(
                screen.getByRole("button", { name: "Create API key" })
            ).toBeInTheDocument();

            expect(
                screen.queryByRole("tab", { name: "Admin API keys" })
            ).not.toBeInTheDocument();
        });

        it("Should load and display user API keys", async () => {
            await renderApiKeys();
            await waitFor(() => {
                expect(
                    screen.getByText(mockUserApiKeysList[0].name)
                ).toBeInTheDocument();
            });
        });

        it("Should not render the Admin/User tabs or the owner column", async () => {
            await renderApiKeys();
            // Wait for the table rows so the column-visibility assertion runs
            // against a fully rendered table, not a loading state.
            await waitFor(() => {
                expect(
                    screen.getByText(mockUserApiKeysList[0].name)
                ).toBeInTheDocument();
            });

            expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
            expect(
                screen.queryByRole("tab", { name: "Admin API keys" })
            ).not.toBeInTheDocument();
            expect(
                screen.queryByRole("tab", { name: "User API keys" })
            ).not.toBeInTheDocument();
            expect(
                screen.queryByRole("columnheader", { name: "User" })
            ).not.toBeInTheDocument();
        });
    });

    describe("Admin mode", () => {
        it("Should show the supported admin and user tabs", async () => {
            await renderApiKeys(true);

            await waitFor(() => {
                expect(
                    screen.getByRole("tab", { name: "Admin API keys" })
                ).toBeInTheDocument();
                expect(
                    screen.getByRole("tab", { name: "User API keys" })
                ).toBeInTheDocument();
                expect(
                    screen.queryByRole("tab", { name: "Legacy API keys" })
                ).not.toBeInTheDocument();
            });
            expect(
                screen.queryByRole("tab", { name: "Legacy API keys" })
            ).not.toBeInTheDocument();
        });

        it("Should show the Create API key button so an admin can create on either tab", async () => {
            await renderApiKeys(true);
            await waitFor(() => {
                expect(
                    screen.getByRole("tab", { name: "Admin API keys" })
                ).toBeInTheDocument();
            });
            expect(
                screen.getByRole("button", { name: "Create API key" })
            ).toBeInTheDocument();
        });

        it("Should default to the Admin API keys tab", async () => {
            await renderApiKeys(true);
            await waitFor(() => {
                expect(
                    screen.getByRole("tab", {
                        name: "Admin API keys",
                        selected: true
                    })
                ).toBeInTheDocument();
            });
        });

        it("Should load and display admin API keys with the owner column", async () => {
            await renderApiKeys(true);
            await waitFor(() => {
                expect(
                    screen.getByRole("tab", { name: "Admin API keys" })
                ).toBeInTheDocument();
                expect(
                    screen.getByText(mockAdminApiKeysList[0].name)
                ).toBeInTheDocument();
            });
            // Positive twin of the non-admin negative test: it keeps the
            // "User" columnheader query honest if the header is renamed.
            expect(
                screen.getByRole("columnheader", { name: "User" })
            ).toBeInTheDocument();
        });
    });

    describe("User mode", () => {
        it("Should show the Create API key button and should not show admin api keys tab", async () => {
            await renderApiKeys();
            expect(
                screen.getByRole("button", { name: "Create API key" })
            ).toBeInTheDocument();
            expect(
                screen.queryByRole("tab", { name: "Admin API keys" })
            ).not.toBeInTheDocument();
        });

        it("Should use user API key endpoints outside the admin route", async () => {
            await renderApiKeys();

            await waitFor(() => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    "v0/api-keys"
                );
            });
            expect(mockAxiosInstance.get).not.toHaveBeenCalledWith(
                "v0/api-keys/admin",
                expect.anything()
            );
            expect(
                await screen.findByText(mockUserApiKeysList[0].name)
            ).toBeInTheDocument();
        });
    });

    describe("Delete API Keys flow", () => {
        it("Should close the delete dialog when Cancel is clicked", async () => {
            await openDeleteDialog();

            expect(
                screen.getByText(/are you sure you want to delete/i)
            ).toBeInTheDocument();

            await findAndClickButtonByRole("Cancel");

            await waitFor(() => {
                expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
            });
        });

        it("Should call the delete API and close dialog on confirm", async () => {
            await openDeleteDialog();

            await findAndClickButtonByRole("Delete");

            await waitFor(() => {
                expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
            });
        });

        it("Should show `Internal Server error` message when API returns error", async () => {
            setDeleteUserApiKeyResponseType(testResponseType.ERROR);

            await openDeleteDialog();

            await findAndClickButtonByRole("Delete");

            await waitFor(() => {
                expect(screen.getByText("Internal error")).toBeInTheDocument();
            });
        });

        it("Should show an error dialog when delete fails", async () => {
            setDeleteUserApiKeyResponseType(testResponseType.INVALID);

            await openDeleteDialog();

            await findAndClickButtonByRole("Delete");

            await waitFor(() => {
                expect(
                    screen.getByText("API Key deletion failed!")
                ).toBeInTheDocument();
            });
        });
    });

    describe("Create API key flow", () => {
        it("Should open the create dialog when Create API key is clicked", async () => {
            await openCreateDialog();

            const dialog = screen.getByRole("dialog");
            expect(dialog).toBeInTheDocument();
            // The action-row button reads "Create API key"; scope the title
            // assertion to the dialog so we match the title, not the button.
            expect(
                within(dialog).getByText("Create API Key")
            ).toBeInTheDocument();
            expect(
                screen.getByRole("button", { name: "Create" })
            ).toBeDisabled();
        });

        it("Should enable Create button after entering a name", async () => {
            await openCreateDialog();

            await userEvent.type(
                screen.getByRole("textbox", { name: /api key name/i }),
                "My New Key"
            );

            expect(
                screen.getByRole("button", { name: "Create" })
            ).toBeEnabled();
        });

        it("Should show the created key and success message after creation", async () => {
            await openCreateDialog();

            await userEvent.type(
                screen.getByRole("textbox", { name: /api key name/i }),
                "My New Key"
            );

            await findAndClickButtonByRole("Create");

            await waitFor(() => {
                expect(
                    screen.getByText("Save Your API Key")
                ).toBeInTheDocument();
                expect(
                    screen.getByText(
                        new RegExp(
                            `API key "${mockCreateUserApiKeyResponse.name}" created successfully`
                        )
                    )
                ).toBeInTheDocument();
            });
        });

        it("Should show `Internal Server error` message when API returns error", async () => {
            setCreateUserApiKeyResponseType(testResponseType.ERROR);

            await openCreateDialog();

            await userEvent.type(
                screen.getByRole("textbox", { name: /api key name/i }),
                "My New Key"
            );

            await findAndClickButtonByRole("Create");

            await waitFor(() => {
                expect(
                    screen.getByText(/internal server error/i)
                ).toBeInTheDocument();
            });
        });

        it("Should show an inline error and disable Create when the API returns 422", async () => {
            setCreateUserApiKeyResponseType(testResponseType.INVALID);

            await openCreateDialog();

            await userEvent.type(
                screen.getByRole("textbox", { name: /api key name/i }),
                "My New Key"
            );

            await findAndClickButtonByRole("Create");

            await waitFor(() => {
                expect(
                    screen.getByText("API key name already exists")
                ).toBeInTheDocument();
            });

            expect(
                screen.getByRole("button", { name: "Create" })
            ).toBeDisabled();
        });

        it("Should close the create dialog when Cancel is clicked", async () => {
            await openCreateDialog();

            await findAndClickButtonByRole("Cancel");

            await waitFor(() => {
                expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
            });
        });
    });
});
