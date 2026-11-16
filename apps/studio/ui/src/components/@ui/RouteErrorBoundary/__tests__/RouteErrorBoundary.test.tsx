// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, render, screen } from "@testing-library/react";
import { mockIsRouteErrorResponse, mockRouteError } from "tests/mockRouter";

import RouteErrorBoundary from "../RouteErrorBoundary";

import "tests/mockRouter";

describe("RouteErrorBoundary", () => {
    const renderRouteErrorBoundary = async () => {
        return await act(async () => {
            return render(<RouteErrorBoundary />);
        });
    };

    beforeEach(() => {
        mockIsRouteErrorResponse(false);
    });

    describe("Route error responses", () => {
        beforeEach(() => {
            mockIsRouteErrorResponse(true);
        });

        it("Should show 'Page not found' for 404 errors", async () => {
            mockRouteError({ status: 404, statusText: "Not Found" });

            await renderRouteErrorBoundary();

            expect(screen.getByText(/Page not found/i)).toBeInTheDocument();
            expect(screen.getByText(/Error 404/i)).toBeInTheDocument();
        });

        it("Should show statusText for non-404 route errors", async () => {
            mockRouteError({
                status: 500,
                statusText: "Internal Server Error"
            });

            await renderRouteErrorBoundary();

            expect(
                screen.getByText(/Internal Server Error/i)
            ).toBeInTheDocument();
            expect(screen.getByText(/Error 500/i)).toBeInTheDocument();
        });

        it("Should fall back to data.message when statusText is absent", async () => {
            mockRouteError({
                status: 403,
                statusText: "",
                data: { message: "Access denied" }
            });

            await renderRouteErrorBoundary();

            expect(screen.getByText(/Access denied/i)).toBeInTheDocument();
        });
    });

    describe("JavaScript errors", () => {
        it("Should show message for Error instances", async () => {
            mockRouteError(new Error("Something broke"));

            await renderRouteErrorBoundary();

            expect(screen.getByText(/Something broke/i)).toBeInTheDocument();
        });

        it("Should show string errors directly", async () => {
            mockRouteError("Connection failed");

            await renderRouteErrorBoundary();

            expect(screen.getByText(/Connection failed/i)).toBeInTheDocument();
        });

        it("Should show default message for unknown error types", async () => {
            mockRouteError({ unknownError: true });

            await renderRouteErrorBoundary();

            expect(
                screen.getByText(/An unexpected error occurred/i)
            ).toBeInTheDocument();
        });
    });

    it("Should always show Go Back button", async () => {
        mockRouteError("error");

        await renderRouteErrorBoundary();

        expect(
            screen.getByRole("button", { name: /Go Back/i })
        ).toBeInTheDocument();
    });
});
