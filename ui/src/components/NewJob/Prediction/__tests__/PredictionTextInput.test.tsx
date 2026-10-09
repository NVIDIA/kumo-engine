// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import PredictionTextInput, {
    PredictionTextInputProps
} from "../Settings/PredictionTextInput";

const onChange = vi.fn();
const defaultProps: PredictionTextInputProps = {
    id: "",
    title: "",
    placeholder: "",
    value: "",
    errorMsg: "",
    isInvalid: false,
    onChange
};

describe("PredictionTextInput", () => {
    const renderPredictionTextInput = async (
        props: PredictionTextInputProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <PredictionTextInput {...props} />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render `PredictionTextInput` with all valid props", async () => {
        await renderPredictionTextInput({
            id: "parallel-workers-number",
            title: "Number of parallel workers",
            placeholder: "Number of parallel workers",
            value: "1",
            isInvalid: false,
            onChange: vi.fn()
        });

        const input = screen.getByPlaceholderText("Number of parallel workers");
        expect(input).toHaveAttribute("id", "parallel-workers-number");
        expect(
            screen.getByText("Number of parallel workers")
        ).toBeInTheDocument();
        expect(
            screen.getByPlaceholderText("Number of parallel workers")
        ).toBeInTheDocument();
        expect(screen.getByDisplayValue("1")).toBeInTheDocument();
    });

    it("Should render `PredictionTextInput` when all the props are set empty", async () => {
        await renderPredictionTextInput(defaultProps);

        expect(screen.getByPlaceholderText("")).toBeInTheDocument();
        expect(screen.getByDisplayValue("")).toBeInTheDocument();
    });

    it("Should render `PredictionTextInput` when `isInvalid` is set as true", async () => {
        await renderPredictionTextInput({
            ...defaultProps,
            errorMsg: "This field is required",
            isInvalid: true
        });

        const input = screen.getByPlaceholderText("");
        expect(input).toHaveAttribute("aria-invalid", "true");
        expect(screen.getByText("This field is required")).toBeInTheDocument();
    });

    it("Should handle change of the value", async () => {
        await renderPredictionTextInput({
            ...defaultProps,
            placeholder: "Number of parallel workers"
        });

        const input = screen.getByPlaceholderText("Number of parallel workers");
        expect(input).toBeInTheDocument();
        expect(screen.getByDisplayValue("")).toBeInTheDocument();

        await userEvent.click(input);
        const inputValue = "4";
        await userEvent.type(input, inputValue);

        expect(onChange).toHaveBeenCalled();
        expect(onChange).toHaveBeenCalledWith(inputValue);
    });

    it("Should select all text when Ctrl+A is pressed", async () => {
        await renderPredictionTextInput({
            ...defaultProps,
            placeholder: "Number of parallel workers",
            value: "test"
        });

        const input = screen.getByRole("textbox") as HTMLInputElement;
        await userEvent.type(input, "{Control>}a{/Control}");
        expect(input.selectionStart).toBe(0);
        expect(input.selectionEnd).toBe(4);
    });

    it("Should handle very long input values", async () => {
        await renderPredictionTextInput({
            ...defaultProps,
            placeholder: "Number of parallel workers"
        });

        const longText = "a".repeat(1000);
        const input = screen.getByRole("textbox");
        await userEvent.type(input, longText);
        expect(onChange).toHaveBeenCalledTimes(1000);
    });

    it("Should handle special characters in input", async () => {
        await renderPredictionTextInput({
            ...defaultProps,
            placeholder: "Number of parallel workers",
            isInvalid: true
        });

        const input = screen.getByRole("textbox");
        await userEvent.type(input, "!@#$%^&*()");
        expect(onChange).toHaveBeenCalledWith("!");
        expect(onChange).toHaveBeenLastCalledWith(")");
        const error = screen.getByPlaceholderText("Number of parallel workers");
        expect(error).toHaveAttribute("aria-invalid", "true");
    });

    it("Should be accessible via keyboard navigation", async () => {
        await renderPredictionTextInput({
            ...defaultProps,
            placeholder: "Number of parallel workers"
        });

        const input = screen.getByRole("textbox");
        await userEvent.tab();
        expect(input).toHaveFocus();
    });

    it("Should handle paste events", async () => {
        await renderPredictionTextInput({
            ...defaultProps,
            placeholder: "Number of parallel workers"
        });

        const input = screen.getByRole("textbox");
        await userEvent.click(input);
        await userEvent.paste("pasted value");
        expect(onChange).toHaveBeenCalledWith("pasted value");
    });

    it("Should handle rapid input changes", async () => {
        await renderPredictionTextInput({
            ...defaultProps,
            placeholder: "Number of parallel workers"
        });

        const input = screen.getByRole("textbox");
        await userEvent.type(input, "quick typing", { delay: 0 });
        expect(onChange).toHaveBeenCalledTimes("quick typing".length);
    });
});
