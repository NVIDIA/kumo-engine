// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { TaskType } from "types/predictiveQueries";

import PredictionFilter, {
    PredictionFilterProps
} from "../Settings/PredictionFilter";

const onChange = vi.fn();
const defaultProps: PredictionFilterProps = {
    title: "Prediction Entity Filter",
    taskType: TaskType.NONE,
    value: "FINANCE_ACCOUNT.FREQUENCY = 'POPLATEK MESICNE'",
    isPQLANG_V2: true,
    isInvalid: false,
    onChange
};

describe("PredictionFilter", () => {
    const renderPredictionFilter = async (props: PredictionFilterProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <PredictionFilter {...props} />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render `PredictionFilter` when all the props are set empty", async () => {
        await renderPredictionFilter({
            ...defaultProps,
            title: "",
            value: "",
            isPQLANG_V2: false
        });

        const textArea = screen.getByTestId("prediction-filter");
        expect(textArea.className).toMatch(
            /border-destructive|ring-destructive/
        );
    });

    it("Should render `PredictionFilter` with all valid props", async () => {
        await renderPredictionFilter({
            ...defaultProps,
            taskType: TaskType.BINARY_CLASSIFICATION
        });

        expect(
            screen.getByText("Prediction Entity Filter")
        ).toBeInTheDocument();
        const input = screen.getByTestId("prediction-filter");
        expect(input).toHaveAttribute("id", "prediction-filter");
        expect(
            screen.getByDisplayValue(
                "FINANCE_ACCOUNT.FREQUENCY = 'POPLATEK MESICNE'"
            )
        ).toBeInTheDocument();
    });

    it("Should render `PredictionFilter` for various task type", async () => {
        await renderPredictionFilter({
            ...defaultProps,
            taskType: TaskType.REGRESSION
        });

        expect(
            screen.getByText("Prediction Entity Filter")
        ).toBeInTheDocument();
        expect(
            screen.getByDisplayValue(
                "FINANCE_ACCOUNT.FREQUENCY = 'POPLATEK MESICNE'"
            )
        ).toBeInTheDocument();
    });

    it("Should render `PredictionFilter` when `isInvalid` is set as true", async () => {
        await renderPredictionFilter({
            ...defaultProps,
            isInvalid: true
        });

        const textArea = screen.getByTestId("prediction-filter");
        expect(textArea.className).toMatch(
            /border-destructive|ring-destructive/
        );
    });

    it("Should handle change of the value", async () => {
        await renderPredictionFilter({
            ...defaultProps,
            isPQLANG_V2: false,
            value: ""
        });

        const input = screen.getByTestId("prediction-filter");
        expect(input).toBeInTheDocument();
        expect(screen.getByDisplayValue("")).toBeInTheDocument();

        await userEvent.click(input);
        const inputValue = "FINANCE_ACCOUNT.FREQUENCY = 'POPLATEK MESICNE'";
        await userEvent.type(input, inputValue);

        expect(onChange).toHaveBeenCalled();
        expect(onChange).toHaveBeenCalledTimes(46);
    });

    it("Should select all text when Ctrl+A is pressed", async () => {
        await renderPredictionFilter({
            ...defaultProps,
            isPQLANG_V2: false
        });

        const input = screen.getByRole("textbox") as HTMLInputElement;
        await userEvent.type(input, "{Control>}a{/Control}");
        expect(input.selectionStart).toBe(0);
        expect(input.selectionEnd).toBe(46);
    });

    it("Should handle very long input values", async () => {
        await renderPredictionFilter({
            ...defaultProps,
            isPQLANG_V2: false
        });

        const longText = "a".repeat(1000);
        const input = screen.getByTestId("prediction-filter");
        await userEvent.type(input, longText);
        expect(onChange).toHaveBeenCalledTimes(1000);
    });

    it("Should handle special characters in input", async () => {
        await renderPredictionFilter({
            ...defaultProps,
            title: "",
            value: "",
            isPQLANG_V2: false,
            isInvalid: true
        });

        const input = screen.getByTestId("prediction-filter");
        await userEvent.type(input, "!@#$%^&*()");
        expect(onChange).toHaveBeenCalledWith("!");
        expect(onChange).toHaveBeenLastCalledWith(")");
    });

    it("Should be accessible via keyboard navigation", async () => {
        await renderPredictionFilter({
            ...defaultProps,
            isPQLANG_V2: false
        });

        const input = screen.getByTestId("prediction-filter");
        await userEvent.tab();
        expect(input).toHaveFocus();
    });

    it("Should handle paste events", async () => {
        await renderPredictionFilter({
            ...defaultProps,
            title: "",
            value: "",
            isPQLANG_V2: false
        });

        const input = screen.getByTestId("prediction-filter");
        await userEvent.click(input);
        await userEvent.paste("pasted value");
        expect(onChange).toHaveBeenCalledWith("pasted value");
    });

    it("Should handle rapid input changes", async () => {
        await renderPredictionFilter({
            ...defaultProps,
            taskType: TaskType.NONE,
            isPQLANG_V2: false
        });

        const input = screen.getByTestId("prediction-filter");
        await userEvent.type(input, "quick typing", { delay: 0 });
        expect(onChange).toHaveBeenCalledTimes("quick typing".length);
    });
});
