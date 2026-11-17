// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import PredictionAnchorTime, {
    PredictionAnchorTimeProps
} from "../Settings/PredictionAnchorTime";

const onChange = vi.fn();
const defaultProps: PredictionAnchorTimeProps = {
    onChange
};

describe("PredictionAnchorTime", () => {
    const renderPredictionAnchorTime = async (
        props: PredictionAnchorTimeProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <PredictionAnchorTime {...props} />
                </MemoryRouter>
            );
        });
    };

    const openDatePicker = async () => {
        const dateInput = screen.getByRole("button");
        await userEvent.click(dateInput);
        const datePicker = await screen.findByRole("dialog");
        return datePicker;
    };

    const getTimeInputs = async () => {
        const timeInputs = await screen.findAllByRole("spinbutton");
        const [hoursInput, minutesInput, secondsInput] = timeInputs;
        const amPmButton = screen.getByRole("button", { name: /AM|PM/i });
        return { hoursInput, minutesInput, secondsInput, amPmButton };
    };

    it("Should render PredictionAnchorTime with header", async () => {
        await renderPredictionAnchorTime(defaultProps);

        const header = screen.getByText("Prediction Anchor Time");
        expect(header).toBeInTheDocument();
    });

    it("Should display current date and default time when no value provided", async () => {
        const currentDate = new Date();
        const currentYear = currentDate.getFullYear();
        const currentDay = currentDate.getDate();
        const currentMonth = currentDate.toLocaleString("default", {
            month: "long"
        });

        await renderPredictionAnchorTime(defaultProps);

        await openDatePicker();

        // Verify date defaults
        const comboboxes = await screen.findAllByRole("combobox");
        const [monthButton, yearButton] = comboboxes;

        expect(monthButton).toHaveTextContent(currentMonth);
        expect(yearButton).toHaveTextContent(currentYear.toString());

        const selectedDayButton = document.querySelector(
            'button[name="day"][tabindex="0"]'
        ) as HTMLButtonElement;

        expect(selectedDayButton).toBeInTheDocument();
        expect(selectedDayButton).toHaveTextContent(currentDay.toString());

        // Verify time defaults to 12:00:00 AM
        const { hoursInput, minutesInput, secondsInput, amPmButton } =
            await getTimeInputs();
        expect(hoursInput).toHaveValue(12);
        expect(minutesInput).toHaveValue(0);
        expect(secondsInput).toHaveValue(0);
        expect(amPmButton).toHaveTextContent("AM");
    });

    describe("Date and time changes", () => {
        beforeEach(() => {
            window.HTMLElement.prototype.hasPointerCapture = vi.fn();
            window.HTMLElement.prototype.releasePointerCapture = vi.fn();
        });

        it("Should handle changes in Month, Year, and Date", async () => {
            await renderPredictionAnchorTime(defaultProps);

            await openDatePicker();

            const comboboxes = await screen.findAllByRole("combobox");
            const [monthButton] = comboboxes;

            // Change month to May
            await userEvent.click(monthButton);
            const monthDropdown = await screen.findByRole("listbox");
            const mayOption = await within(monthDropdown).findByText("May", {
                selector: '[role="option"] span'
            });
            await userEvent.click(mayOption);

            const updatedComboboxes = await screen.findAllByRole("combobox");
            expect(updatedComboboxes[0]).toHaveTextContent("May");

            // Change year to 2000
            await userEvent.click(updatedComboboxes[1]);
            const yearDropdown = await screen.findByRole("listbox");
            const yearOption = await within(yearDropdown).findByText("2000", {
                selector: '[role="option"] span'
            });
            await userEvent.click(yearOption);

            const finalComboboxes = await screen.findAllByRole("combobox");
            expect(finalComboboxes[1]).toHaveTextContent("2000");

            // Select day 15
            const dayToSelect = await screen.findByText("15");
            await userEvent.click(dayToSelect);

            expect(onChange).toHaveBeenCalledWith(expect.any(Date));
        });

        it("Should handle changes in Date and Time", async () => {
            await renderPredictionAnchorTime(defaultProps);

            await openDatePicker();

            // Select date
            const days = await screen.findAllByText("26", {
                selector: 'button[name="day"]'
            });
            await userEvent.click(days[0]);

            // Change time to 11:30:45 PM
            const { hoursInput, minutesInput, secondsInput, amPmButton } =
                await getTimeInputs();

            await userEvent.click(hoursInput);

            await userEvent.clear(hoursInput);
            await userEvent.type(hoursInput, "11");
            expect(hoursInput).toHaveValue(11);

            await userEvent.clear(minutesInput);
            await userEvent.type(minutesInput, "30");
            expect(minutesInput).toHaveValue(30);

            await userEvent.clear(secondsInput);
            await userEvent.type(secondsInput, "45");
            expect(secondsInput).toHaveValue(45);

            if (amPmButton.textContent !== "PM") {
                await userEvent.click(amPmButton);
            }

            await waitFor(() => {
                expect(onChange).toHaveBeenCalled();
            });
        });

        it("Should validate time inputs and prevent invalid values", async () => {
            await renderPredictionAnchorTime(defaultProps);

            await openDatePicker();

            const { hoursInput } = await getTimeInputs();

            await userEvent.click(hoursInput);
            await userEvent.keyboard("[Backspace][Backspace]24");
            expect(hoursInput).toHaveValue(0);
        });

        it("Should toggle between AM and PM", async () => {
            await renderPredictionAnchorTime(defaultProps);

            await openDatePicker();

            const { amPmButton } = await getTimeInputs();
            const initialPeriod = amPmButton.textContent;

            await userEvent.click(amPmButton);
            expect(amPmButton.textContent).not.toBe(initialPeriod);

            await userEvent.click(amPmButton);
            expect(amPmButton.textContent).toBe(initialPeriod);
        });
    });

    it("Should display tooltip with documentation link when hovering over info icon", async () => {
        await renderPredictionAnchorTime(defaultProps);

        const tooltipTrigger = document.querySelector(
            '[data-slot="tooltip-trigger"]'
        );
        expect(tooltipTrigger).toBeInTheDocument();

        await userEvent.hover(tooltipTrigger!);
        const tooltip = await screen.findByRole("tooltip");

        expect(tooltip).toHaveTextContent(
            "Anchor time represents a specific point in time from which you want to predict the future. To learn more, visit the"
        );

        const docLink = within(tooltip).getByTestId(
            "testid-prediction-anchor-link"
        );
        expect(docLink).toHaveTextContent("docs");
        expect(docLink).toHaveAttribute(
            "href",
            "https://docs.nvidia.com/sdgm/kumo-engine/1.0.0/kumo-studio-guide-coming-soon/preliminary-documentation/train-model/predictive-query/what-is-anchor-time-and-why-is-it-important"
        );

        const linkWrapper = docLink!.closest("span");
        expect(linkWrapper).toHaveClass(
            "transition-colors",
            "text-text-primary",
            "underline"
        );
    });
});
