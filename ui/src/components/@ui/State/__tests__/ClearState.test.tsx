// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen } from "@testing-library/react";

import ClearState from "../ClearState";

describe("ClearState", () => {
    const renderClearState = async () => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <ClearState />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.useFakeTimers();
        Object.defineProperty(window, "location", {
            configurable: true,
            value: {
                href: "",
                origin: "http://localhost",
                assign: (url: string) => {
                    window.location.href = url;
                }
            }
        });
    });

    afterEach(() => {
        vi.clearAllTimers();
        vi.useRealTimers();
    });

    it("Should render Bouncingball component", async () => {
        await renderClearState();

        expect(screen.getByTestId("bouncingball-loading")).toBeInTheDocument();
    });

    it("Should clear localStorage on mount", async () => {
        localStorage.setItem("testKey", "testValue");
        await renderClearState();

        expect(localStorage.getItem("testKey")).toBeNull();
    });

    it("Should redirect after 1 second", async () => {
        await renderClearState();

        act(() => {
            vi.advanceTimersByTime(1000);
        });

        expect(window.location.href).toBe(window.location.origin);
    });
});
