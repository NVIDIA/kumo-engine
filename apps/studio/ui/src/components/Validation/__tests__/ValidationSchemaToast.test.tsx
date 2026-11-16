// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import {
    act,
    fireEvent,
    render,
    screen,
    waitFor
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
    triggerValidationSchemaToast,
    ValidationSchemaToast,
    ValidationSchemaToastProps
} from "../ValidationSchemaToast";

const defaultProps: ValidationSchemaToastProps = {
    method: "GET",
    url: "https://api.example.com/resource",
    apiResponse: '{"message": "Invalid"}',
    validationErrors: '{"field": "is required"}',
    apiPayload: '{"data": "test"}',
    onClose: vi.fn()
};

describe("ValidationSchemaToast", () => {
    const renderValidationSchemaToast = async (
        props: ValidationSchemaToastProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <ValidationSchemaToast {...props} />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render the title and call `onClose` when close button is clicked", async () => {
        const onClose = vi.fn();
        await renderValidationSchemaToast({
            ...defaultProps,
            onClose
        });

        expect(
            screen.getByText("API response validation failed")
        ).toBeInTheDocument();

        const buttons = screen.getAllByRole("button");
        const closeBtn = buttons[0];
        await userEvent.click(closeBtn);
        expect(onClose).toHaveBeenCalled();
    });

    it("Should display truncated URL if it exceeds 32 characters", async () => {
        const longUrl =
            "https://api.example.com/very/long/path/that/needs/truncation";

        await renderValidationSchemaToast({
            ...defaultProps,
            url: longUrl
        });

        const truncatedText = screen.getByText((text) =>
            text.startsWith("GET: https://api.example.com/very")
        );
        expect(truncatedText).toBeInTheDocument();
    });

    it("Should render collapsible items", async () => {
        await renderValidationSchemaToast(defaultProps);

        expect(screen.getByText("Show API Response")).toBeInTheDocument();
        expect(screen.getByText("Show Validation Errors")).toBeInTheDocument();
        expect(screen.getByText("Show API Payload")).toBeInTheDocument();
    });

    it("Should expand collapsible and show content on click", async () => {
        await renderValidationSchemaToast(defaultProps);

        const toggle = screen.getByText("Show API Response");
        fireEvent.click(toggle);

        await waitFor(() => {
            expect(
                screen.getByText(defaultProps.apiResponse)
            ).toBeInTheDocument();
        });
    });

    it("Should not render API Payload section when apiPayload is not provided", async () => {
        const propsWithoutPayload = {
            ...defaultProps,
            apiPayload: undefined
        };

        await renderValidationSchemaToast(propsWithoutPayload);

        expect(screen.queryByText("Show API Payload")).not.toBeInTheDocument();
    });

    it("Should render tooltip content for the main error message", async () => {
        await renderValidationSchemaToast(defaultProps);

        const errorTitle = screen.getByText("API response validation failed");
        expect(errorTitle).toBeInTheDocument();
    });
});

describe("triggerValidationSchemaToast", () => {
    it("Should trigger a toast with the validation schema toast component", () => {
        expect(() => {
            triggerValidationSchemaToast({
                method: "POST",
                url: "https://api.example.com/test",
                apiResponse: '{"error": "test"}',
                validationErrors: '{"field": "error"}',
                apiPayload: '{"payload": "data"}'
            });
        }).not.toThrow();
    });

    it("Should trigger a toast without apiPayload", () => {
        expect(() => {
            triggerValidationSchemaToast({
                method: "GET",
                url: "https://api.example.com/endpoint",
                apiResponse: '{"status": "error"}',
                validationErrors: '{"message": "validation failed"}'
            });
        }).not.toThrow();
    });
});
