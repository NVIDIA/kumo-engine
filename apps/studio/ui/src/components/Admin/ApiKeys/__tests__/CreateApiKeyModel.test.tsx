// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { APIKeyType } from "gen/v0/types";

import { CreateAPIKey, CreateApiKeyModelProps } from "../CreateApiKeyModel";

const toggleCreate = vi.fn();
const createKeys = vi.fn();

const defaultProps: CreateApiKeyModelProps = {
    open: true,
    toggleCreate,
    createKeys,
    keyType: APIKeyType.USER,
    isAdmin: false
};

describe("CreateAPIKey", () => {
    const renderCreateAPIKey = async (
        props: CreateApiKeyModelProps = defaultProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <CreateAPIKey {...props} />
                </MemoryRouter>
            );
        });
    };

    const typeKeyName = async (name: string) => {
        await userEvent.type(
            screen.getByRole("textbox", { name: /api key name/i }),
            name
        );
    };

    const clickCreate = async () => {
        await userEvent.click(
            screen.getByRole("button", { name: /^create$/i })
        );
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render the dialog when open is true", async () => {
        await renderCreateAPIKey();

        expect(screen.getByRole("dialog")).toBeInTheDocument();
        expect(screen.getByText("Create API Key")).toBeInTheDocument();
        expect(
            screen.getByRole("textbox", { name: /api key name/i })
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /cancel/i })
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /^create$/i })
        ).toBeInTheDocument();
    });

    it("Should not render the dialog when open is false", async () => {
        await renderCreateAPIKey({ ...defaultProps, open: false });

        expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    describe("Dialog title", () => {
        it("Non-admin sees the unqualified 'Create API Key' title", async () => {
            await renderCreateAPIKey({
                ...defaultProps,
                isAdmin: false,
                keyType: APIKeyType.USER
            });

            expect(screen.getByText("Create API Key")).toBeInTheDocument();
        });

        it("Admin on the Admin tab sees 'Create Admin API Key'", async () => {
            await renderCreateAPIKey({
                ...defaultProps,
                isAdmin: true,
                keyType: APIKeyType.ADMIN
            });

            expect(
                screen.getByText("Create Admin API Key")
            ).toBeInTheDocument();
        });

        it("Admin on the User tab sees 'Create User API Key'", async () => {
            await renderCreateAPIKey({
                ...defaultProps,
                isAdmin: true,
                keyType: APIKeyType.USER
            });

            expect(screen.getByText("Create User API Key")).toBeInTheDocument();
        });
    });

    describe("Name input", () => {
        it("Should disable Create when name is empty", async () => {
            await renderCreateAPIKey();

            expect(
                screen.getByRole("button", { name: /^create$/i })
            ).toBeDisabled();
        });

        it("Should enable Create after entering a name", async () => {
            await renderCreateAPIKey();

            await typeKeyName("My Key");

            expect(
                screen.getByRole("button", { name: /^create$/i })
            ).toBeEnabled();
        });

        it("Should disable Create when input contains only whitespace", async () => {
            await renderCreateAPIKey();

            await typeKeyName("   ");

            expect(
                screen.getByRole("button", { name: /^create$/i })
            ).toBeDisabled();
        });

        it("Should clear the inline error when the name is changed", async () => {
            createKeys.mockResolvedValue({
                error: "API key name already exists",
                statusCode: 422
            });
            await renderCreateAPIKey();
            await typeKeyName("Duplicate");
            await clickCreate();

            await waitFor(() => {
                expect(
                    screen.getByText("API key name already exists")
                ).toBeInTheDocument();
            });

            await userEvent.clear(
                screen.getByRole("textbox", { name: /api key name/i })
            );
            await typeKeyName("New Name");

            expect(
                screen.queryByText("API key name already exists")
            ).not.toBeInTheDocument();
        });
    });

    it("Should call toggleCreate when Cancel is clicked", async () => {
        await renderCreateAPIKey();

        await userEvent.click(screen.getByRole("button", { name: /cancel/i }));

        expect(toggleCreate).toHaveBeenCalledTimes(1);
    });
});
