// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRef, RefObject } from "react";
import { MemoryRouter } from "react-router";

import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
    expectNoAlert,
    expectNoDialog,
    getAlert,
    getDialog,
    verifyTextExists
} from "tests/testHelpers";

import { DialogRef } from "types/common";

import Validation, { ValidationProps } from "../Validation";

const defaultProps = {
    pageLoading: false,
    banner: null,
    alert: null,
    ctaMsgs: null,
    dialogRef: createRef() as RefObject<DialogRef<object>>,
    hideWarnings: vi.fn(),
    onDialogConfirm: vi.fn(),
    onDialogCancel: vi.fn(),
    onDialogSkip: vi.fn(),
    clearErrors: vi.fn()
};

describe("Validation", () => {
    const renderValidation = async (props: ValidationProps<object>) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <Validation {...props} />
                </MemoryRouter>
            );
        });
    };

    const toggleDialogAndVerify = (
        dialogRef: RefObject<DialogRef<object>>,
        title: string,
        description: string,
        value?: object,
        showYesNo?: boolean
    ) => {
        act(() => {
            dialogRef.current?.toggleDialog(
                title,
                description,
                value,
                showYesNo
            );
        });

        const dialog = getDialog();
        expect(dialog).toHaveTextContent(title);
        expect(dialog).toHaveTextContent(description);
    };

    const clickButtonAndVerifyCallback = async (
        buttonName: string | RegExp,
        callback: ReturnType<typeof vi.fn>,
        expectedValue?: object
    ) => {
        const button = screen.getByRole("button", { name: buttonName });
        await userEvent.click(button);

        if (expectedValue !== undefined) {
            expect(callback).toHaveBeenCalledWith(expectedValue);
        } else {
            expect(callback).toHaveBeenCalled();
        }
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render empty dom when all the props are set empty", async () => {
        await renderValidation(defaultProps);

        expect(
            screen.queryByTestId("animated-loading")
        ).not.toBeInTheDocument();
        expectNoAlert();
        expectNoDialog();
    });

    it("Should display animated-loading when the prop `pageLoading` is set as true ", async () => {
        await renderValidation({
            ...defaultProps,
            pageLoading: true
        });

        expect(screen.getByTestId("animated-loading")).toBeInTheDocument();
    });

    it("Should render banner title and warnings", async () => {
        await renderValidation({
            ...defaultProps,
            banner: {
                title: "Important Banner",
                warnings: ["Something went wrong"],
                isWarning: true
            }
        });

        const alert = getAlert();
        expect(alert).toHaveTextContent("Important Banner");
        expect(alert).toHaveTextContent("Something went wrong");
    });

    it("Should render alert when provided", async () => {
        await renderValidation({
            ...defaultProps,
            alert: {
                title: "Critical Alert",
                description: "This needs attention"
            }
        });

        const dialog = getDialog();
        expect(dialog).toHaveTextContent("Critical Alert");
        expect(dialog).toHaveTextContent("This needs attention");
    });

    it("Should render CTA errors when `showCTAErrors` is true", async () => {
        await renderValidation({
            ...defaultProps,
            ctaMsgs: {
                errors: [
                    {
                        id: "err-1",
                        description: "Critical failure",
                        is_dismissible: true,
                        is_error: true,
                        title: "Error",
                        cta: null
                    }
                ],
                warnings: null,
                info_items: null,
                showCTAErrors: true,
                showCTAWarnings: false,
                warnings_fetched: true
            }
        });

        getAlert();
        await verifyTextExists("Critical failure");
    });

    it("Should render CTA warnings when `showCTAWarnings` is true", async () => {
        await renderValidation({
            ...defaultProps,
            ctaMsgs: {
                errors: null,
                warnings: [
                    {
                        id: "warn-1",
                        description: "This is a warning",
                        is_dismissible: false,
                        is_error: false,
                        title: "Warning",
                        cta: null
                    }
                ],
                info_items: null,
                showCTAErrors: false,
                showCTAWarnings: true,
                warnings_fetched: true
            }
        });

        getAlert();
        await verifyTextExists("This is a warning");
    });

    it("Should call `clearErrors` when close button is clicked", async () => {
        const clearErrors = vi.fn();
        const onDialogCancel = vi.fn();
        await renderValidation({
            ...defaultProps,
            alert: {
                title: "Critical Alert",
                description: "This needs attention"
            },
            onDialogCancel,
            clearErrors
        });

        const dialog = getDialog();
        expect(dialog).toHaveTextContent("Critical Alert");
        expect(dialog).toHaveTextContent("This needs attention");

        await clickButtonAndVerifyCallback("Cancel", clearErrors);
    });

    it.each([
        {
            description:
                "Should call `onDialogSkip` when skip button is clicked",
            callbackKey: "onDialogSkip",
            buttonName: /cancel/i,
            showYesNo: true,
            testValue: undefined
        },
        {
            description:
                "Should call `onDialogConfirm` when continue button is clicked",
            callbackKey: "onDialogConfirm",
            buttonName: /continue/i,
            showYesNo: false,
            testValue: undefined
        },
        {
            description:
                "Should call `onDialogConfirm` when confirm button is clicked with skip enabled",
            callbackKey: "onDialogConfirm",
            buttonName: /continue/i,
            showYesNo: true,
            testValue: undefined
        }
    ])(
        "$description",
        async ({ callbackKey, buttonName, showYesNo, testValue }) => {
            const callback = vi.fn();
            const dialogRef = createRef() as RefObject<DialogRef<object>>;

            await renderValidation({
                ...defaultProps,
                dialogRef,
                [callbackKey]: callback
            });

            toggleDialogAndVerify(
                dialogRef,
                "Test Title",
                "Test Description",
                testValue,
                showYesNo
            );
            await clickButtonAndVerifyCallback(buttonName, callback, testValue);
        }
    );

    it("Should toggle dialog visibility when toggleDialog is called", async () => {
        const dialogRef = createRef() as RefObject<DialogRef<object>>;

        await renderValidation({
            ...defaultProps,
            dialogRef
        });

        expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

        toggleDialogAndVerify(dialogRef, "Test Title", "Test Description");

        act(() => {
            dialogRef.current?.toggleDialog();
        });

        expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("Should render error banner with title and warnings when isWarning is false", async () => {
        await renderValidation({
            ...defaultProps,
            banner: {
                title: "Critical Error",
                warnings: ["Error 1", "Error 2"],
                isWarning: false
            }
        });

        const alert = getAlert();
        await verifyTextExists("Critical Error");
        await verifyTextExists("Error 1");
        await verifyTextExists("Error 2");
        expect(alert).toHaveAttribute("id", "cta-errors-2");
    });

    it("Should render warning banner without title when title is empty", async () => {
        await renderValidation({
            ...defaultProps,
            banner: {
                title: "",
                warnings: ["Something went wrong"],
                isWarning: true
            }
        });

        const alert = getAlert();
        await verifyTextExists("Warning");
        await verifyTextExists("Something went wrong");
        expect(alert).toHaveAttribute("id", "cta-warnings-2");
    });

    it("Should not render banner when warnings array is empty", async () => {
        await renderValidation({
            ...defaultProps,
            banner: {
                title: "No Warnings",
                warnings: [],
                isWarning: true
            }
        });

        expectNoAlert();
    });

    it("Should not render CTA warnings when `showCTAWarnings` is false", async () => {
        await renderValidation({
            ...defaultProps,
            ctaMsgs: {
                warnings: [
                    {
                        id: "warn-1",
                        description: "Should not be shown",
                        is_dismissible: false,
                        is_error: false,
                        title: "Warning",
                        cta: null
                    }
                ],
                errors: null,
                info_items: null,
                showCTAErrors: false,
                showCTAWarnings: false,
                warnings_fetched: true
            }
        });

        expectNoAlert();
    });

    it("Should use default no-op functions for dialog actions if not provided", async () => {
        const dialogRef = createRef() as RefObject<DialogRef<object>>;

        await renderValidation({
            ...defaultProps,
            dialogRef,
            onDialogConfirm: undefined,
            onDialogCancel: undefined,
            onDialogSkip: undefined
        });

        act(() => {
            dialogRef.current?.toggleDialog(
                "Title",
                "Description",
                { test: "data" },
                true
            );
        });

        const skipButton = screen.getByRole("button", { name: /cancel/i });
        const continueButton = screen.getByRole("button", {
            name: /continue/i
        });

        // Click all the buttons to invoke the defaults
        await userEvent.click(skipButton);
        await userEvent.click(continueButton);
    });
});
