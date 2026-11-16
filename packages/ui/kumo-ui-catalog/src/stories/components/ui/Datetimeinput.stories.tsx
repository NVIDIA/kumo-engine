// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DateTimeInput } from "@/components/ui/datetimeinput";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { expect, fn, userEvent, waitFor, within } from "storybook/test";

const meta: Meta<typeof DateTimeInput> = {
    title: "Components/UI/DateTimeInput",
    component: DateTimeInput,
    tags: ["autodocs"],
    args: {
        onDateTimeChange: fn()
    },
    argTypes: {
        placeholder: {
            control: "text",
            description: "Placeholder text for the DateTimeInput"
        }
    }
};

export default meta;
type Story = StoryObj<typeof DateTimeInput>;

function SyncsFromDefaultDateStory() {
    const [date, setDate] = useState<Date | undefined>(
        new Date("2026-04-02T10:30:00")
    );

    return (
        <div className="flex flex-col gap-2">
            <button
                type="button"
                onClick={() => setDate(new Date("2026-05-10T08:15:00"))}>
                Set New Date
            </button>
            <DateTimeInput
                placeholder="Select Date-Time"
                defaultDate={date}
                onDateTimeChange={setDate}
            />
        </div>
    );
}

// Date-time input
export const Date_Time_Input: Story = {
    render: () => <DateTimeInput placeholder="Select Date-Time"></DateTimeInput>
};

export const OpensCalendarOnClick: Story = {
    render: () => <DateTimeInput placeholder="Select Date-Time" />,
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getByRole("button");

        await expect(trigger).toBeVisible();
        await expect(trigger).toHaveTextContent("Select Date-Time");

        await userEvent.click(trigger);
        await waitFor(() => {
            expect(document.querySelector('[role="dialog"]')).not.toBeNull();
        });

        await userEvent.click(trigger);
        await waitFor(() => {
            expect(document.querySelector('[role="dialog"]')).toBeNull();
        });
    }
};

export const SelectsDate: Story = {
    args: {
        placeholder: "Select Date-Time"
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getByRole("button");

        await expect(trigger).toHaveTextContent("Select Date-Time");

        await userEvent.click(trigger);

        await waitFor(() => {
            expect(document.querySelector('[role="dialog"]')).not.toBeNull();
        });

        const dialog = within(
            document.querySelector('[role="dialog"]') as HTMLElement
        );
        const dayButtons = dialog.getAllByRole("gridcell");
        const enabledDay = dayButtons.find(
            (btn) => !btn.hasAttribute("disabled") && btn.textContent?.trim()
        );
        if (enabledDay) {
            await userEvent.click(enabledDay);
        }

        await waitFor(() => {
            expect(args.onDateTimeChange).toHaveBeenCalledTimes(1);
        });
        await waitFor(() => {
            expect(trigger).not.toHaveTextContent("Select Date-Time");
        });

        await userEvent.click(trigger);
        await waitFor(() => {
            expect(document.querySelector('[role="dialog"]')).toBeNull();
        });
    }
};

export const ClosesOnTriggerClickWhenOpen: Story = {
    render: () => <DateTimeInput placeholder="Pick a date" />,
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getByRole("button");

        await userEvent.click(trigger);
        await waitFor(() => {
            expect(document.querySelector('[role="dialog"]')).not.toBeNull();
        });

        await userEvent.click(trigger);
        await waitFor(() => {
            expect(document.querySelector('[role="dialog"]')).toBeNull();
        });
    }
};

export const SyncsFromDefaultDate: Story = {
    render: () => <SyncsFromDefaultDateStory />,
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getAllByRole("button")[1];

        await expect(trigger).toHaveTextContent("2026-04-02 10:30:00");

        await userEvent.click(canvas.getByText("Set New Date"));

        await waitFor(() => {
            expect(trigger).toHaveTextContent("2026-05-10 08:15:00");
        });
    }
};
