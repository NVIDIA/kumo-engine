// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createQueryClient } from "tests/testUtil";

import { ValidationResponse } from "hooks/validation/validatedName";

import { InputState } from "types/inputs";

import ValidatedNameInput, { ValidatedNameInputProps } from "../NameInput";

const defaultProps: ValidatedNameInputProps = {
    readOnly: false,
    value: "",
    setName: vi.fn(),
    nameInputState: InputState.None,
    setNameInputState: vi.fn(),
    nameValidationResponse: { isValid: true, errMsg: "" },
    setNameValidationResponse: vi.fn(),
    connectorId: "",
    updateLocalFileTableName: vi.fn(),
    fileConnectors: []
};

describe("NameInput", () => {
    const queryClient = createQueryClient();
    const renderNameInput = async (props: ValidatedNameInputProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <ValidatedNameInput {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    it("Should render NameInput without error messages for a valid `nameValidationResponse`", async () => {
        await renderNameInput(defaultProps);

        expect(screen.getByText("Table name")).toBeInTheDocument();
        expect(screen.queryByText("Required field")).not.toBeInTheDocument();
        expect(screen.queryByText("Missing")).not.toBeInTheDocument();
    });

    it("Should call `setName` when new Table name is entered", async () => {
        const setName = vi.fn();
        const nameInput = "transaction_table";
        await renderNameInput({
            ...defaultProps,
            setName
        });

        const input = screen.getByPlaceholderText("Use default");
        expect(input).toBeEnabled();
        await userEvent.type(input, nameInput);

        expect(setName).toHaveBeenCalledWith(nameInput);
    });

    it("Should not render any empty error container when `nameValidationResponse` is empty", async () => {
        await renderNameInput({
            ...defaultProps,
            nameValidationResponse: {} as ValidationResponse
        });

        const errorElement = document.querySelector(
            '[data-slot="description"]'
        );

        if (errorElement) {
            expect(errorElement?.textContent).not.toBe("");
        } else {
            expect(errorElement).toBeNull();
        }
    });

    it("Should not allow editing when `readOnly` is true", async () => {
        await renderNameInput({
            ...defaultProps,
            readOnly: true,
            value: "readonly_table"
        });

        const input = screen.getByPlaceholderText("Use default");
        expect(input).toBeDisabled();
    });

    it("Should displays error message for `Invalid` name", async () => {
        const value = "03052000";
        await renderNameInput({
            ...defaultProps,
            value,
            setName: vi.fn(),
            nameInputState: InputState.InValid,
            nameValidationResponse: {
                isValid: false,
                errMsg: `Table identifier may only contain alphanumeric characters and underscores, and must begin with a letter. Got: "${value}"`
            }
        });

        const errorText = screen.getByText(
            `Table identifier may only contain alphanumeric characters and underscores, and must begin with a letter. Got: "${value}"`
        );
        expect(errorText).toBeInTheDocument();
    });

    it("Should call `updateLocalFileTableName` if connector is in a file connector", async () => {
        const updateLocalFileTableName = vi.fn();
        await renderNameInput({
            ...defaultProps,
            connectorId: "excel_connector",
            updateLocalFileTableName,
            fileConnectors: ["connector-project-upload", "excel_connector"]
        });

        const input = screen.getByPlaceholderText("Use default");
        await userEvent.type(input, "test_name");

        expect(updateLocalFileTableName).toHaveBeenCalled();
    });

    it("Should not call `updateLocalFileTableName` if connector is not in a file connector", async () => {
        const updateLocalFileTableName = vi.fn();
        await renderNameInput({
            ...defaultProps,
            connectorId: "connector",
            updateLocalFileTableName,
            fileConnectors: ["connector-project-upload", "excel_connector"]
        });

        const input = screen.getByPlaceholderText("Use default");
        await userEvent.type(input, "test_name");

        expect(updateLocalFileTableName).not.toHaveBeenCalled();
    });

    it("Should display a loading spinner when `nameInputState` is Validating", async () => {
        await renderNameInput({
            ...defaultProps,
            value: "test_table",
            nameInputState: InputState.Validating
        });

        const spinner = screen.getByRole("status", { name: /Loading/i });
        expect(spinner).toBeInTheDocument();
        expect(spinner).toHaveClass("animate-spin");
    });

    it("Should handle empty string input", async () => {
        const setName = vi.fn();
        await renderNameInput({
            ...defaultProps,
            value: "",
            setName
        });

        const input = screen.getByPlaceholderText("Use default");
        await userEvent.type(input, "new_table");

        expect(setName).toHaveBeenCalled();
    });

    it("Should not show error for empty/whitespace-only input", async () => {
        await renderNameInput({
            ...defaultProps,
            value: "   ",
            nameValidationResponse: {
                isValid: false,
                errMsg: "Some error"
            }
        });

        expect(screen.queryByText("Some error")).not.toBeInTheDocument();
    });

    it("Should handle special characters in table name", async () => {
        const setName = vi.fn();
        await renderNameInput({
            ...defaultProps,
            setName
        });

        const input = screen.getByPlaceholderText("Use default");
        await userEvent.type(input, "table@#$%");

        expect(setName).toHaveBeenCalledWith("table@#$%");
    });

    it("Should handle very long table names", async () => {
        const setName = vi.fn();
        const longName = "a".repeat(200);
        await renderNameInput({
            ...defaultProps,
            setName
        });

        const input = screen.getByPlaceholderText("Use default");
        await userEvent.type(input, longName);

        expect(setName).toHaveBeenCalledWith(longName);
    });

    it("Should not call `setName` for file connectors when typing", async () => {
        const setName = vi.fn();
        const updateLocalFileTableName = vi.fn();

        await renderNameInput({
            ...defaultProps,
            connectorId: "connector-project-upload",
            fileConnectors: ["connector-project-upload"],
            setName,
            updateLocalFileTableName
        });

        const input = screen.getByPlaceholderText("Use default");
        await userEvent.type(input, "test");

        expect(updateLocalFileTableName).toHaveBeenCalled();
        expect(setName).not.toHaveBeenCalled();
    });

    it("Should handle Valid input state correctly", async () => {
        await renderNameInput({
            ...defaultProps,
            value: "valid_table_name",
            nameInputState: InputState.Valid,
            nameValidationResponse: {
                isValid: true,
                errMsg: ""
            }
        });

        const input = screen.getByPlaceholderText("Use default");
        expect(input).toHaveValue("valid_table_name");
        expect(screen.queryByText("Required field")).not.toBeInTheDocument();
    });

    it("Should handle multiple rapid changes to input", async () => {
        const setName = vi.fn();
        await renderNameInput({
            ...defaultProps,
            setName
        });

        const input = screen.getByPlaceholderText("Use default");
        await userEvent.type(input, "abc");
        await userEvent.clear(input);
        await userEvent.type(input, "xyz");

        expect(setName).toHaveBeenCalled();
        expect(setName.mock.calls.length).toBeGreaterThan(1);
    });
});
