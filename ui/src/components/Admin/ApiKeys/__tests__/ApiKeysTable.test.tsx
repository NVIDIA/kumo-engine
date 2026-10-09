// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { getColumnHeaders, getSkeletonElement } from "tests/testHelpers";

import { mockAdminApiKeysList } from "hooks/apiKeys/__tests__/MockFunctions/AdminApiKeys.mock";

import { ApiKeysTable, ApiKeysTableProps } from "../ApiKeysTable";

const onDeleteClick = vi.fn();

const defaultProps: ApiKeysTableProps = {
    data: mockAdminApiKeysList,
    searchStr: "",
    onDeleteClick,
    isLoading: false,
    showUserColumn: false
};

describe("ApiKeysTable", () => {
    const renderApiKeysTable = async (
        props: ApiKeysTableProps = defaultProps
    ) => {
        await act(async () => {
            render(<ApiKeysTable {...props} />);
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should show a loading skeleton when isLoading is true", async () => {
        await renderApiKeysTable({ ...defaultProps, isLoading: true });

        expect(getSkeletonElement()).toBeInTheDocument();
    });

    it("Should render the table with column headers and all key names from data", async () => {
        await renderApiKeysTable();

        const headers = getColumnHeaders();
        expect(headers).toContain("Name");
        expect(headers).toContain("Created");
        expect(headers).toContain("Value");

        expect(
            screen.getByText(mockAdminApiKeysList[0].name)
        ).toBeInTheDocument();
        expect(
            screen.getByText(mockAdminApiKeysList[1].name)
        ).toBeInTheDocument();
    });

    it("Should render the table when data is empty", async () => {
        await renderApiKeysTable({ ...defaultProps, data: [] });

        expect(document.querySelector("table")).toBeInTheDocument();
    });

    describe("Column visibility", () => {
        it.each([
            { showUserColumn: false, visibleInHeaders: false },
            { showUserColumn: true, visibleInHeaders: true }
        ])(
            "Should show User column in headers: $visibleInHeaders when showUserColumn is $showUserColumn",
            async ({ showUserColumn, visibleInHeaders }) => {
                await renderApiKeysTable({ ...defaultProps, showUserColumn });
                expect(getColumnHeaders().includes("User")).toBe(
                    visibleInHeaders
                );
            }
        );

        it.each(mockAdminApiKeysList)(
            "Should display owner '$owner_display_name' for '$name' when showUserColumn is true",
            async ({ owner_display_name }) => {
                await renderApiKeysTable({
                    ...defaultProps,
                    showUserColumn: true
                });
                expect(
                    screen.getByText(owner_display_name)
                ).toBeInTheDocument();
            }
        );
    });

    describe("Search filtering", () => {
        it("Should show a row when searchStr matches the key name", async () => {
            await renderApiKeysTable({
                ...defaultProps,
                searchStr: mockAdminApiKeysList[0].name
            });

            expect(
                screen.getByText(mockAdminApiKeysList[0].name)
            ).toBeInTheDocument();
        });

        it("Should hide rows when searchStr matches no key names", async () => {
            await renderApiKeysTable({
                ...defaultProps,
                searchStr: "nonexistent-key-xyz"
            });

            expect(
                screen.queryByText(mockAdminApiKeysList[0].name)
            ).not.toBeInTheDocument();
            expect(
                screen.queryByText(mockAdminApiKeysList[1].name)
            ).not.toBeInTheDocument();
            expect(screen.getByText("No data found.")).toBeInTheDocument();
        });

        it("Should filter case-insensitively", async () => {
            const upperName = mockAdminApiKeysList[0].name.toUpperCase();

            await renderApiKeysTable({ ...defaultProps, searchStr: upperName });
            expect(
                screen.getByText(mockAdminApiKeysList[0].name)
            ).toBeInTheDocument();
        });

        it("Should show only the matching row on a partial name match", async () => {
            await renderApiKeysTable({ ...defaultProps, searchStr: "One" });
            expect(
                screen.getByText(mockAdminApiKeysList[0].name)
            ).toBeInTheDocument();
            expect(
                screen.queryByText(mockAdminApiKeysList[1].name)
            ).not.toBeInTheDocument();
        });
    });

    describe("Delete interaction", () => {
        it.each([
            { index: 0, key: mockAdminApiKeysList[0] },
            { index: 1, key: mockAdminApiKeysList[1] }
        ])(
            "Should call `onDeleteClick` with key id when delete button[$index] is clicked",
            async ({ index, key }) => {
                await renderApiKeysTable();

                const deleteButtons = screen.getAllByRole("button", {
                    name: /delete/i
                });
                await userEvent.click(deleteButtons[index]);

                await waitFor(() => {
                    expect(onDeleteClick).toHaveBeenCalledWith(key.id);
                    expect(onDeleteClick).toHaveBeenCalledTimes(1);
                });
            }
        );
    });
});
