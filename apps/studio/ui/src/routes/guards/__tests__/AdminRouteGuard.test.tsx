// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter, Route, Routes } from "react-router";

import { render, screen } from "@testing-library/react";
import { vi } from "vitest";

import AdminRouteGuard from "../AdminRouteGuard";

const mockUseAdminInfo = vi.fn();
vi.mock("hooks/auth/admin", () => ({
    useAdminInfo: () => mockUseAdminInfo()
}));

vi.mock("react-router", async () => {
    const actual = await vi.importActual("react-router");
    return {
        ...actual,
        Navigate: ({ to, replace }: { to: string; replace?: boolean }) => (
            <div>
                Redirect to {to}
                {replace ? " (replace)" : ""}
            </div>
        )
    };
});

describe("AdminRouteGuard", () => {
    const renderAdminRouteGuard = () =>
        render(
            <MemoryRouter initialEntries={["/admin/members"]}>
                <Routes>
                    <Route element={<AdminRouteGuard />}>
                        <Route
                            path="/admin/members"
                            element={<div>AdminChild</div>}
                        />
                    </Route>
                </Routes>
            </MemoryRouter>
        );

    it("Should render the admin child routes for an admin", () => {
        mockUseAdminInfo.mockReturnValue({ isAdmin: true, isLoading: false });
        renderAdminRouteGuard();

        expect(screen.getByText("AdminChild")).toBeInTheDocument();
    });

    it("Should redirect a non-admin to home instead of rendering", () => {
        mockUseAdminInfo.mockReturnValue({ isAdmin: false, isLoading: false });
        renderAdminRouteGuard();

        expect(screen.queryByText("AdminChild")).not.toBeInTheDocument();
        expect(
            screen.getByText("Redirect to /home (replace)")
        ).toBeInTheDocument();
    });

    it("Should render nothing while the admin status is loading", () => {
        mockUseAdminInfo.mockReturnValue({ isAdmin: false, isLoading: true });
        renderAdminRouteGuard();

        expect(screen.queryByText("AdminChild")).not.toBeInTheDocument();
        expect(screen.queryByText(/Redirect to/)).not.toBeInTheDocument();
    });
});
