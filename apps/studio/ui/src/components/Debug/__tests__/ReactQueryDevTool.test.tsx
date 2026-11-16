// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockDevToolMode } from "tests/testHelpers";

import ReactQueryDevTool from "../ReactQueryDevTool";

vi.mock("utils/storage", () => ({
    getReactQueryDevToolMode: vi.fn()
}));

describe("ReactQueryDevTool", () => {
    const renderReactQueryDevTool = async () => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <ReactQueryDevTool />
                </MemoryRouter>
            );
        });
    };

    const getToggleButton = () => screen.getByRole("button");

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should not render when mode is DISABLED", async () => {
        await mockDevToolMode("DISABLED");

        await renderReactQueryDevTool();

        expect(
            screen.queryByText(/react query devtools/i)
        ).not.toBeInTheDocument();
    });

    it("Should render button when mode is ENABLED", async () => {
        await mockDevToolMode("ENABLED");

        await renderReactQueryDevTool();

        expect(
            screen.getByText("Show react query devtools")
        ).toBeInTheDocument();
    });

    it("Should toggle button text when clicked", async () => {
        await mockDevToolMode("ENABLED");

        await renderReactQueryDevTool();

        const button = getToggleButton();
        expect(button).toHaveTextContent("Show react query devtools");

        await userEvent.click(button);
        expect(button).toHaveTextContent("Close react query devtools");

        await userEvent.click(button);
        expect(button).toHaveTextContent("Show react query devtools");
    });
});
