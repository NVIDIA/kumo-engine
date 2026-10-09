// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// stories/Calendar.stories.tsx

import { Calendar } from "@/components/ui/calendar";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";

const meta: Meta<typeof Calendar> = {
    title: "UI/Calendar",
    component: Calendar,
    tags: ["autodocs"],
    parameters: {
        layout: "centered"
    },
    argTypes: {
        mode: {
            control: "text"
        },
        setHMS: {
            description: "Handler to set time in the Calendar date object",
            table: {
                type: { summary: "SetStateAction<Date | undefined>" }
            },
            action: "setHMS", // logs calls in the Actions tab
            control: false
        }
    }
};

export default meta;

type Story = StoryObj<typeof Calendar>;

export const Default: Story = {
    render: () => <Calendar mode="single" />,
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await expect(canvas.getByLabelText("Select month")).toBeVisible();
        await expect(canvas.getByLabelText("Select year")).toBeVisible();

        const nextButton = canvas.getByLabelText("Go to next month");
        await userEvent.click(nextButton);

        const prevButton = canvas.getByLabelText("Go to previous month");
        await userEvent.click(prevButton);

        const day15 = canvas.getByText("15");
        await expect(day15).toBeVisible();
        await userEvent.click(day15);
    }
};

export const Default_With_Time_Inputs: Story & {
    setHMS?: React.Dispatch<React.SetStateAction<Date | undefined>>;
} = {
    args: {
        mode: "single",
        setHMS: () => {}
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const hoursInput = canvas.getByLabelText("Hours");
        const minutesInput = canvas.getByLabelText("Minutes");
        const secondsInput = canvas.getByLabelText("Seconds");
        await expect(hoursInput).toBeVisible();
        await expect(minutesInput).toBeVisible();
        await expect(secondsInput).toBeVisible();

        await userEvent.clear(hoursInput);
        await userEvent.type(hoursInput, "5");
        await expect(hoursInput).toHaveValue(5);

        await userEvent.clear(minutesInput);
        await userEvent.type(minutesInput, "30");
        await expect(minutesInput).toHaveValue(30);

        const ampmButton = canvas.getByRole("button", { name: /AM|PM/ });
        const initialText = ampmButton.textContent;
        await userEvent.click(ampmButton);
        const expectedText = initialText === "AM" ? "PM" : "AM";
        await expect(ampmButton).toHaveTextContent(expectedText);
    }
};
