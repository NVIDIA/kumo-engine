// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import * as directConnectSession from "hooks/auth/directConnectAuth";

import * as directConnectAuth from "utils/directConnectAuth";

import DirectConnectLogin from "../DirectConnectLogin";

vi.mock("hooks/auth/directConnectAuth", () => ({
    clearDirectConnectLoggedOut: vi.fn(),
    completeDirectConnectPasswordReset: vi.fn(),
    initDirectConnect: vi.fn(),
    isDirectConnectLoggedOut: vi.fn(),
    loginDirectConnect: vi.fn(),
    loginDirectConnectWithPassword: vi.fn()
}));

vi.mock("utils/directConnectAuth", async () => {
    const actual = await vi.importActual<
        typeof import("utils/directConnectAuth")
    >("utils/directConnectAuth");
    return {
        ...actual,
        isDirectConnectPasswordLogin: vi.fn()
    };
});

const mockAssign = vi.fn();

describe("DirectConnectLogin", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        vi.mocked(directConnectSession.initDirectConnect).mockResolvedValue(
            false
        );
        vi.mocked(
            directConnectSession.isDirectConnectLoggedOut
        ).mockReturnValue(false);
        vi.mocked(
            directConnectAuth.isDirectConnectPasswordLogin
        ).mockReturnValue(true);
        Object.defineProperty(window, "location", {
            value: {
                assign: mockAssign,
                origin: "http://localhost:3000",
                pathname: "/login",
                search: ""
            },
            writable: true
        });
    });

    it("Should redirect to OIDC login outside password mode", async () => {
        vi.mocked(
            directConnectAuth.isDirectConnectPasswordLogin
        ).mockReturnValue(false);

        render(React.createElement(DirectConnectLogin));

        await waitFor(() => {
            expect(directConnectSession.loginDirectConnect).toHaveBeenCalled();
        });
        expect(screen.getByText("Redirecting to login...")).toBeInTheDocument();
    });

    it("Should render signed out page outside password mode", () => {
        vi.mocked(
            directConnectAuth.isDirectConnectPasswordLogin
        ).mockReturnValue(false);
        vi.mocked(
            directConnectSession.isDirectConnectLoggedOut
        ).mockReturnValue(true);

        render(React.createElement(DirectConnectLogin));

        screen.getByRole("heading", {
            name: "You've signed out"
        });

        expect(
            screen.getByText(
                "Your Kumo session has ended. You may still be signed in to your organization's identity provider."
            )
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: "Sign in again" })
        ).toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Sign in again" }));

        expect(directConnectSession.loginDirectConnect).toHaveBeenCalledTimes(
            1
        );
    });

    it("Should render signed out page in password mode and return to login", () => {
        vi.mocked(
            directConnectSession.isDirectConnectLoggedOut
        ).mockReturnValue(true);

        render(React.createElement(DirectConnectLogin));

        expect(
            screen.getByRole("heading", {
                name: "You've signed out"
            })
        ).toBeInTheDocument();
        expect(
            screen.getByText("Your Kumo session has ended.")
        ).toBeInTheDocument();
        expect(
            screen.queryByText(
                /You may still be signed in to your organization's identity provider/
            )
        ).not.toBeInTheDocument();
        const emailInput = screen.queryByLabelText("Email Address");

        expect(emailInput).not.toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Sign in again" }));

        expect(
            directConnectSession.clearDirectConnectLoggedOut
        ).toHaveBeenCalledTimes(1);
        expect(mockAssign).toHaveBeenCalledWith("/login");
        expect(directConnectSession.loginDirectConnect).not.toHaveBeenCalled();
    });

    it("Should render password form in password mode", async () => {
        render(React.createElement(DirectConnectLogin));

        const emailInput = await screen.findByLabelText("Email Address");
        const passwordInput = screen.getByLabelText("Password");

        expect(emailInput).toBeInTheDocument();
        expect(emailInput).toHaveAttribute("placeholder", "you@example.com");
        expect(emailInput).toBeRequired();
        expect(passwordInput).toHaveAttribute(
            "placeholder",
            "Enter your password"
        );
        expect(passwordInput).not.toHaveAttribute("minLength");
        expect(passwordInput).toBeRequired();
        expect(
            screen.queryByText("We'll never share your email with anyone else.")
        ).not.toBeInTheDocument();
        expect(
            screen.queryByText(
                "Your password must be at least 12 characters long."
            )
        ).not.toBeInTheDocument();
        expect(directConnectSession.loginDirectConnect).not.toHaveBeenCalled();
    });

    it("Should require password reset for temporary passwords", async () => {
        vi.mocked(
            directConnectSession.loginDirectConnectWithPassword
        ).mockResolvedValue({
            status: "password_reset_required",
            reset_token: "reset-token"
        });

        render(React.createElement(DirectConnectLogin));
        fireEvent.change(await screen.findByLabelText("Email Address"), {
            target: { value: "alice@example.com" }
        });
        fireEvent.change(screen.getByLabelText("Password"), {
            target: { value: "SeedPass12345" }
        });
        fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

        await waitFor(() => {
            expect(
                directConnectSession.loginDirectConnectWithPassword
            ).toHaveBeenCalledWith("alice@example.com", "SeedPass12345");
        });
        expect(
            await screen.findByLabelText("New password")
        ).toBeInTheDocument();
    });

    it("Should block weak password reset before submitting", async () => {
        vi.mocked(
            directConnectSession.loginDirectConnectWithPassword
        ).mockResolvedValue({
            status: "password_reset_required",
            reset_token: "reset-token"
        });

        render(React.createElement(DirectConnectLogin));
        fireEvent.change(await screen.findByLabelText("Email Address"), {
            target: { value: "alice@example.com" }
        });
        fireEvent.change(screen.getByLabelText("Password"), {
            target: { value: "SeedPass12345" }
        });
        fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

        fireEvent.change(await screen.findByLabelText("New password"), {
            target: { value: "short1" }
        });

        expect(
            await screen.findByText("Password must be at least 12 characters")
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: "Set password" })
        ).toBeDisabled();
        expect(
            directConnectSession.completeDirectConnectPasswordReset
        ).not.toHaveBeenCalled();
    });

    it("Should show backend reset policy errors", async () => {
        vi.mocked(
            directConnectSession.loginDirectConnectWithPassword
        ).mockResolvedValue({
            status: "password_reset_required",
            reset_token: "reset-token"
        });
        vi.mocked(
            directConnectSession.completeDirectConnectPasswordReset
        ).mockRejectedValue(
            new Error(
                JSON.stringify({
                    detail: "New password must differ from the temporary password"
                })
            )
        );

        render(React.createElement(DirectConnectLogin));
        fireEvent.change(await screen.findByLabelText("Email Address"), {
            target: { value: "alice@example.com" }
        });
        fireEvent.change(screen.getByLabelText("Password"), {
            target: { value: "SeedPass12345" }
        });
        fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

        fireEvent.change(await screen.findByLabelText("New password"), {
            target: { value: "FinalPass67890" }
        });
        fireEvent.click(screen.getByRole("button", { name: "Set password" }));

        expect(
            await screen.findByText(
                "New password must differ from the temporary password"
            )
        ).toBeInTheDocument();
    });

    it("Should complete password reset and navigate", async () => {
        vi.mocked(
            directConnectSession.loginDirectConnectWithPassword
        ).mockResolvedValue({
            status: "password_reset_required",
            reset_token: "reset-token"
        });
        vi.mocked(
            directConnectSession.completeDirectConnectPasswordReset
        ).mockResolvedValue({ status: "ok" });

        render(React.createElement(DirectConnectLogin));
        fireEvent.change(await screen.findByLabelText("Email Address"), {
            target: { value: "alice@example.com" }
        });
        fireEvent.change(screen.getByLabelText("Password"), {
            target: { value: "SeedPass12345" }
        });
        fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

        fireEvent.change(await screen.findByLabelText("New password"), {
            target: { value: "FinalPass67890" }
        });
        fireEvent.click(screen.getByRole("button", { name: "Set password" }));

        await waitFor(() => {
            expect(
                directConnectSession.completeDirectConnectPasswordReset
            ).toHaveBeenCalledWith("reset-token", "FinalPass67890");
            expect(mockAssign).toHaveBeenCalledWith("/home");
        });
    });
});
