// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";
import { Label } from "@/components/ui/label";
import { TextArea } from "@/components/ui/text-area";

const meta: Meta<typeof TextArea> = {
    title: "UI/TextArea",
    component: TextArea,
    tags: ["autodocs"],
    argTypes: {
        disabled: {
            control: "boolean",
            description: "Whether the input is disabled"
        },
        placeholder: {
            control: "text",
            description: "Placeholder text for the input"
        }
    }
};

export default meta;
type Story = StoryObj<typeof TextArea>;

// Basic Input
export const Basic: Story = {
    args: {
        placeholder: "Enter your name"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const textarea = canvas.getByPlaceholderText("Enter your name");

        await expect(textarea).toHaveValue("");

        await userEvent.type(textarea, "Hello World");
        await expect(textarea).toHaveValue("Hello World");
    }
};

// Input with Label
export const WithLabel: Story = {
    render: () => (
        <div className="grid w-full max-w-lg items-center gap-1.5">
            <Label htmlFor="name">Name</Label>
            <TextArea id="name" placeholder="Enter your name" />
        </div>
    )
};

// Disabled Input
export const Disabled: Story = {
    args: {
        disabled: true,
        placeholder: "This input is disabled",
        value: "Disabled value"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const textarea = canvas.getByDisplayValue("Disabled value");

        await expect(textarea).toBeDisabled();
    }
};
