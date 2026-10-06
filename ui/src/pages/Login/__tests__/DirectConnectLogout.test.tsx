// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { logoutDirectConnect } from "hooks/auth/directConnectAuth";

import DirectConnectLogout from "../DirectConnectLogout";

vi.mock("hooks/auth/directConnectAuth", () => ({
    logoutDirectConnect: vi.fn()
}));

describe("DirectConnectLogout", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("shows a retry when backend logout fails", async () => {
        vi.mocked(logoutDirectConnect).mockResolvedValue(false);

        render(React.createElement(DirectConnectLogout));

        expect(await screen.findByText("Sign out failed")).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: "Try again" })
        ).toBeInTheDocument();
    });

    it("retries logout", async () => {
        vi.mocked(logoutDirectConnect)
            .mockResolvedValueOnce(false)
            .mockResolvedValueOnce(true);

        render(React.createElement(DirectConnectLogout));
        fireEvent.click(
            await screen.findByRole("button", { name: "Try again" })
        );

        await waitFor(() => {
            expect(logoutDirectConnect).toHaveBeenCalledTimes(2);
        });
    });
});
