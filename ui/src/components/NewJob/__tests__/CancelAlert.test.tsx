// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import CancelAlert, { CancelAlertProps } from "../CancelAlert";

const handleClose = vi.fn();
const handleConfirm = vi.fn();

const defaultProps: CancelAlertProps = {
    title: "Discard Changes",
    description: "Are you sure you want to discard your changes?",
    show: true,
    handleClose,
    handleConfirm
};

describe("job cancellation alert", () => {
    const renderCancelAlert = async (props: CancelAlertProps) => {
        await act(async () => {
            render(<CancelAlert {...props} />);
        });
    };

    const clickButton = async (name: string) => {
        await userEvent.click(screen.getByRole("button", { name }));
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it.each([
        {
            show: false,
            shouldExist: false
        },
        {
            show: true,
            shouldExist: true
        }
    ])(
        "Should $shouldExist render dialog content when show is $show",
        async ({ show, shouldExist }) => {
            await renderCancelAlert({ ...defaultProps, show });
            await waitFor(() => {
                if (shouldExist) {
                    expect(
                        screen.getByText("Discard Changes")
                    ).toBeInTheDocument();
                    expect(
                        screen.getByText(
                            "Are you sure you want to discard your changes?"
                        )
                    ).toBeInTheDocument();
                } else {
                    expect(
                        screen.queryByText("Discard Changes")
                    ).not.toBeInTheDocument();
                    expect(
                        screen.queryByText(
                            "Are you sure you want to discard your changes?"
                        )
                    ).not.toBeInTheDocument();
                }
            });
        }
    );

    it("Should render Cancel and Discard buttons", async () => {
        await renderCancelAlert(defaultProps);

        expect(
            screen.getByRole("button", { name: "Cancel" })
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: "Discard" })
        ).toBeInTheDocument();
    });

    it.each([
        {
            buttonName: "Cancel",
            expectedFn: handleClose,
            notExpectedFn: handleConfirm
        },
        {
            buttonName: "Discard",
            expectedFn: handleConfirm,
            notExpectedFn: handleClose
        }
    ])(
        "Should call correct handler when $buttonName button is clicked",
        async ({ buttonName, expectedFn, notExpectedFn }) => {
            await renderCancelAlert(defaultProps);

            await clickButton(buttonName);

            expect(expectedFn).toHaveBeenCalledTimes(1);
            expect(notExpectedFn).not.toHaveBeenCalled();
        }
    );

    it("Should use fallback title and description when undefined values are passed", async () => {
        await renderCancelAlert({
            ...defaultProps,
            title: undefined as unknown as string,
            description: undefined as unknown as string
        });

        expect(screen.getByText("Error")).toBeInTheDocument();
        expect(screen.getByText("Something went wrong!!")).toBeInTheDocument();
    });

    it("Should allow Discard click when handleConfirm is undefined", async () => {
        await renderCancelAlert({
            ...defaultProps,
            handleConfirm: undefined as unknown as () => void
        });

        await clickButton("Discard");

        expect(handleClose).not.toHaveBeenCalled();
    });
});
