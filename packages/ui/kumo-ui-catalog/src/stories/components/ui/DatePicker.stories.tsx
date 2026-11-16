// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DatePicker } from "@/components/ui/date-picker";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, fn, userEvent, waitFor, within } from "storybook/test";

const meta: Meta<typeof DatePicker> = {
    title: "Components/UI/DatePicker",
    component: DatePicker,
    tags: ["autodocs"],
    args: {
        onDateChange: fn()
    },
    argTypes: {
        placeholder: {
            control: "text",
            description: "Placeholder text shown before a date is selected"
        },
        format: {
            control: "text",
            description: "date-fns format string for the displayed date"
        }
    }
};

export default meta;
type Story = StoryObj<typeof DatePicker>;

export const Default: Story = {
    render: (args) => <DatePicker {...args} />
};

export const CustomPlaceholder: Story = {
    args: {
        placeholder: "Choose a date"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByRole("button")).toHaveTextContent(
            "Choose a date"
        );
    }
};

export const OpensCalendarOnClick: Story = {
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

export const WithPlaceholderDate: Story = {
    args: {
        placeholderDate: new Date("2024-01-15T10:30:00")
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByRole("button")).toHaveTextContent(
            "2024-01-15 10:30:00"
        );
    }
};
