// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import PredictionExplanations, {
    PredictionExplanationsProps
} from "../Settings/PredictionExplanations";

const onChange = vi.fn();
const defaultProps: PredictionExplanationsProps = {
    isToggled: false,
    disabled: false,
    onChange
};

describe("PredictionExplanations", () => {
    const renderPredictionExplanations = async (
        props: PredictionExplanationsProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <PredictionExplanations {...props} />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render `PredictionExplanations` when `isToggled` and `disabled` are set as false", async () => {
        await renderPredictionExplanations({
            isToggled: false,
            disabled: false,
            onChange: vi.fn()
        });

        expect(screen.getByText("Enable Explanations")).toBeInTheDocument();

        const toggle = screen.getByRole("switch");
        expect(toggle).not.toBeChecked();
    });

    it("Should render `PredictionExplanations` when `isToggled` is set as true", async () => {
        await renderPredictionExplanations({
            ...defaultProps,
            isToggled: true
        });

        expect(screen.getByText("Enable Explanations")).toBeInTheDocument();

        const toggle = screen.getByRole("switch");
        expect(toggle).toBeChecked();
    });

    it("Should render `PredictionExplanations` when `disabled` is set as true", async () => {
        await renderPredictionExplanations({
            ...defaultProps,
            disabled: true
        });

        expect(screen.getByText("Enable Explanations")).toBeInTheDocument();

        const toggle = screen.getByRole("switch");
        expect(toggle).toBeDisabled();
    });

    it("Should handle change of the toggle", async () => {
        const onChange = vi.fn();
        await renderPredictionExplanations({
            isToggled: false,
            disabled: false,
            onChange
        });

        expect(screen.getByText("Enable Explanations")).toBeInTheDocument();

        const toggle = screen.getByRole("switch");
        expect(toggle).not.toBeChecked();
        await userEvent.click(toggle);
        expect(onChange).toHaveBeenCalledTimes(1);
    });
});
