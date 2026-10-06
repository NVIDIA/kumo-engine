// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { verifySkeletonExists, verifyTextExists } from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";
import { describe, expect, it, vi } from "vitest";

import Groups from "../Groups";

import { setGroupsResponseType } from "./MockFunctions/Groups.mock";
import groupsResponse from "./MockResponses/Groups.response.json";

const mockNavigate = vi.fn();
vi.mock("react-router", async () => {
    const actual =
        await vi.importActual<typeof import("react-router")>("react-router");
    return { ...actual, useNavigate: () => mockNavigate };
});

describe("Groups Component", () => {
    const queryClient = createQueryClient();

    const renderPage = async () => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <Groups />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const waitForGroupsLoad = async () => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith("v0/groups");
        });
    };

    const typeInSearch = async (term: string) => {
        const searchInput = await screen.findByPlaceholderText("Search");
        await userEvent.type(searchInput, term);
        return searchInput;
    };

    const closeDialog = async (dialog: HTMLElement) => {
        const closeButtons = within(dialog).getAllByRole("button", {
            name: /close/i
        });
        await userEvent.click(closeButtons.at(-1)!);
        await waitFor(() => {
            expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
        });
    };

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
        setGroupsResponseType(testResponseType.ALL);
    });

    describe("Initial Load", () => {
        it("Should show skeleton cells in the table while loading", async () => {
            await renderPage();

            verifySkeletonExists();
        });

        it("Should render column headers, search bar and New Group button", async () => {
            await renderPage();

            await screen.findByRole("heading", { name: "Groups" });

            ["Groups", "Description"].forEach((header) => {
                expect(
                    screen.getByRole("columnheader", { name: header })
                ).toBeInTheDocument();
            });
            expect(screen.getByPlaceholderText("Search")).toBeInTheDocument();
            expect(screen.getByTestId("admin-page")).toHaveAttribute(
                "data-admin-page-title",
                "Groups"
            );
            expect(
                screen.getByTestId("admin-search-container")
            ).toBeInTheDocument();
            expect(
                screen.getByRole("button", { name: /New Group/i })
            ).toBeInTheDocument();
        });
    });

    describe("Data Display", () => {
        it.each(groupsResponse)(
            "Should display row for group '$display_name'",
            async ({ display_name, description }) => {
                await renderPage();
                await waitForGroupsLoad();

                await verifyTextExists(display_name);
                await verifyTextExists(description);
            }
        );
    });

    describe("API Response States", () => {
        it("Should show 'No data found' when API returns empty list", async () => {
            setGroupsResponseType(testResponseType.EMPTY);
            await renderPage();
            await waitForGroupsLoad();

            expect(screen.getByText("No data found.")).toBeInTheDocument();
        });

        it("Should show error dialog on server error (500)", async () => {
            setGroupsResponseType(testResponseType.ERROR);
            await renderPage();
            await waitForGroupsLoad();

            expect(await screen.findByRole("dialog")).toBeInTheDocument();
            expect(
                await screen.findByText("Something went wrong!")
            ).toBeInTheDocument();
            expect(
                await screen.findByText("No data found.")
            ).toBeInTheDocument();
        });

        it("Should show error alert banner on invalid response (422)", async () => {
            setGroupsResponseType(testResponseType.INVALID);
            await renderPage();
            await waitForGroupsLoad();

            expect(await screen.findByRole("alert")).toBeInTheDocument();
            expect(
                await screen.findByText(
                    "Sorry, there is an error with one of the inputs"
                )
            ).toBeInTheDocument();
            expect(
                await screen.findByText("Failed to get Groups")
            ).toBeInTheDocument();
            expect(
                await screen.findByText("No data found.")
            ).toBeInTheDocument();
        });
    });

    describe("Search / Filtering", () => {
        it.each([
            {
                field: "display name",
                searchTerm: "Engineering",
                expectedVisible: "Engineering Team",
                expectedHidden: ["Data Science Team", "Administrators"]
            },
            {
                field: "description",
                searchTerm: "Platform administrators",
                expectedVisible: "Platform administrators",
                expectedHidden: [
                    "All engineers in the organization",
                    "Data scientists and analysts"
                ]
            }
        ])(
            "Should filter rows by $field when searching '$searchTerm'",
            async ({ searchTerm, expectedVisible, expectedHidden }) => {
                await renderPage();
                await waitForGroupsLoad();

                await typeInSearch(searchTerm);

                await waitFor(() => {
                    expect(
                        screen.getByText(expectedVisible)
                    ).toBeInTheDocument();
                    expectedHidden.forEach((text) => {
                        expect(
                            screen.queryByText(text)
                        ).not.toBeInTheDocument();
                    });
                });
            }
        );

        it("Should show all rows when search is cleared", async () => {
            await renderPage();
            await waitForGroupsLoad();

            const searchInput = await typeInSearch("Engineering");

            await waitFor(() => {
                expect(
                    screen.queryByText("Administrators")
                ).not.toBeInTheDocument();
            });

            await userEvent.clear(searchInput);

            for (const group of groupsResponse) {
                await verifyTextExists(group.display_name);
            }
        });

        it("Should show no rows when search matches nothing", async () => {
            await renderPage();
            await waitForGroupsLoad();

            await typeInSearch("no-match");

            await waitFor(() => {
                for (const group of groupsResponse) {
                    expect(
                        screen.queryByText(group.display_name)
                    ).not.toBeInTheDocument();
                }
            });
        });
    });

    describe("Dialogs", () => {
        it("Should open and close the NewGroup dialog", async () => {
            await renderPage();
            await waitForGroupsLoad();

            const button = await screen.findByRole("button", {
                name: /New Group/i
            });
            await userEvent.click(button);

            const dialog = await screen.findByRole("dialog");
            expect(dialog).toBeInTheDocument();
            expect(
                within(dialog).getAllByRole("heading", { name: "New Group" })
                    .length
            ).toBeGreaterThan(0);

            await closeDialog(dialog);
        });

        it("Should navigate to the group edit page on row click", async () => {
            await renderPage();
            await waitForGroupsLoad();

            const groupName = await screen.findByText(
                groupsResponse[0].display_name
            );
            await userEvent.click(groupName);

            await waitFor(() => {
                expect(mockNavigate).toHaveBeenCalledWith(
                    `/admin/groups/${groupsResponse[0].id}`
                );
            });
        });
    });
});
