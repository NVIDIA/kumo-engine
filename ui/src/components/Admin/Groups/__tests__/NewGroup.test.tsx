// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import NewGroup from "../NewGroup";

import { setCreateGroupResponseType } from "./MockFunctions/NewGroup.mock";

describe("NewGroup", () => {
    const queryClient = createQueryClient();
    const onClose = vi.fn();

    const renderDialog = () =>
        render(
            <MemoryRouter>
                <QueryClientProvider client={queryClient}>
                    <NewGroup open onClose={onClose} />
                </QueryClientProvider>
            </MemoryRouter>
        );

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
        setCreateGroupResponseType(testResponseType.ALL);
    });

    it("renders only native group metadata fields", () => {
        renderDialog();

        expect(
            screen.getAllByRole("heading", { name: "New Group" }).length
        ).toBeGreaterThan(0);
        expect(screen.getByText("Group name")).toBeVisible();
        expect(screen.getByText(/Description/)).toBeVisible();
        expect(
            screen.queryByText(/identity provider group/i)
        ).not.toBeInTheDocument();
        expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    });

    it("requires a non-blank native group name", async () => {
        renderDialog();
        const create = screen.getByRole("button", { name: "Create" });
        expect(create).toBeDisabled();

        await userEvent.type(screen.getByPlaceholderText("Enter name"), "   ");
        expect(create).toBeDisabled();

        await userEvent.type(screen.getByPlaceholderText("Enter name"), "Team");
        expect(create).toBeEnabled();
    });

    it("creates one native group without external-provider fields", async () => {
        renderDialog();
        await userEvent.type(
            screen.getByPlaceholderText("Enter name"),
            " Team "
        );
        await userEvent.type(
            screen.getByPlaceholderText("Enter description"),
            " Data owners "
        );
        await userEvent.click(screen.getByRole("button", { name: "Create" }));

        await waitFor(() =>
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                "v0/groups",
                { display_name: "Team", description: "Data owners" },
                { headers: { "Content-Type": "application/json" } }
            )
        );
        await waitFor(() => expect(onClose).toHaveBeenCalled());
    });

    it("omits an empty optional description", async () => {
        renderDialog();
        await userEvent.type(screen.getByPlaceholderText("Enter name"), "Team");
        await userEvent.click(screen.getByRole("button", { name: "Create" }));

        await waitFor(() =>
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                "v0/groups",
                { display_name: "Team" },
                { headers: { "Content-Type": "application/json" } }
            )
        );
    });

    it("previews a derived ID and submits an explicit ID", async () => {
        renderDialog();
        await userEvent.type(
            screen.getByPlaceholderText("Enter name"),
            "Risk Team"
        );
        expect(screen.getByText("Preview: risk-team")).toBeVisible();
        await userEvent.type(
            screen.getByRole("textbox", { name: /ID/i }),
            "risk-owners"
        );
        await userEvent.click(screen.getByRole("button", { name: "Create" }));

        await waitFor(() =>
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                "v0/groups",
                {
                    display_name: "Risk Team",
                    slug: "risk-owners",
                    description: undefined
                },
                { headers: { "Content-Type": "application/json" } }
            )
        );
    });

    it("closes without creating when cancelled", async () => {
        renderDialog();
        await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
        expect(onClose).toHaveBeenCalled();
        expect(mockAxiosInstance.post).not.toHaveBeenCalled();
    });

    it("surfaces create failures", async () => {
        setCreateGroupResponseType(testResponseType.ERROR);
        renderDialog();
        await userEvent.type(screen.getByPlaceholderText("Enter name"), "Team");
        await userEvent.click(screen.getByRole("button", { name: "Create" }));

        expect(await screen.findByText("Group creation failed!")).toBeVisible();
        expect(onClose).not.toHaveBeenCalled();
    });

    it("preserves an explicit ID when the server reports a slug conflict", async () => {
        setCreateGroupResponseType("conflict");
        renderDialog();
        await userEvent.type(
            screen.getByPlaceholderText("Enter name"),
            "Risk Team"
        );
        const slugInput = screen.getByRole("textbox", { name: /ID/i });
        await userEvent.type(slugInput, "risk-owners");
        await userEvent.click(screen.getByRole("button", { name: "Create" }));

        expect(
            await screen.findByText(
                "Group slug 'risk-owners' is already in use."
            )
        ).toBeVisible();
        expect(mockAxiosInstance.post).toHaveBeenCalledTimes(1);
        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
            "v0/groups",
            {
                display_name: "Risk Team",
                slug: "risk-owners",
                description: undefined
            },
            { headers: { "Content-Type": "application/json" } }
        );
        expect(slugInput).toHaveValue("risk-owners");
        expect(onClose).not.toHaveBeenCalled();
    });
});
