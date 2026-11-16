// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import DebugConsoleFooter, {
    DebugConsoleFooterProps
} from "../DebugConsoleFooter";

const handleCopyAll = vi.fn();
const handleClose = vi.fn();
const defaultProps: DebugConsoleFooterProps = {
    message: "Test message",
    handleCopyAll,
    loading: false,
    handleClose
};

describe("DebugConsoleFooter", () => {
    const renderDebugConsoleFooter = async (props: DebugConsoleFooterProps) => {
        await act(async () => {
            render(<DebugConsoleFooter {...props} />);
        });
    };

    const getCopyAllButton = () =>
        screen.getByRole("button", { name: /copy all/i });
    const getCloseButton = () => screen.getByRole("button", { name: /close/i });

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render message, Copy All and Close buttons", async () => {
        await renderDebugConsoleFooter(defaultProps);

        expect(screen.getByText("Test message")).toBeInTheDocument();
        expect(getCopyAllButton()).toBeInTheDocument();
        expect(getCloseButton()).toBeInTheDocument();
    });

    it("Should call handleCopyAll when Copy All button is clicked", async () => {
        await renderDebugConsoleFooter({ ...defaultProps, handleCopyAll });

        const copyButton = getCopyAllButton();
        await userEvent.click(copyButton);

        expect(handleCopyAll).toHaveBeenCalledTimes(1);
    });

    it("Should call handleClose when Close button is clicked", async () => {
        await renderDebugConsoleFooter({ ...defaultProps, handleClose });

        const closeButton = getCloseButton();
        await userEvent.click(closeButton);

        expect(handleClose).toHaveBeenCalledTimes(1);
    });

    it.each([
        { loading: true, expectedDisabled: true },
        { loading: false, expectedDisabled: false }
    ])(
        "Should $expectedDisabled Copy All button when loading is $loading",
        async ({ loading, expectedDisabled }) => {
            await renderDebugConsoleFooter({
                ...defaultProps,
                loading
            });

            const copyButton = getCopyAllButton();
            if (expectedDisabled) {
                expect(copyButton).toBeDisabled();
            } else {
                expect(copyButton).not.toBeDisabled();
            }
        }
    );

    it("Should display custom message", async () => {
        await renderDebugConsoleFooter({
            ...defaultProps,
            message: "Custom test message"
        });

        expect(screen.getByText("Custom test message")).toBeInTheDocument();
    });
});
