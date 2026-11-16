// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Checkbox } from "@/components/ui/checkbox";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, fn, userEvent, within } from "storybook/test";

const meta = {
    title: "UI/Checkbox",
    component: Checkbox,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    args: {
        onCheckedChange: fn()
    },
    argTypes: {
        checked: {
            control: "boolean",
            description: "Whether the checkbox is checked"
        },
        disabled: {
            control: "boolean",
            description: "Whether the checkbox is disabled"
        },
        className: {
            control: { type: "text" },
            description: "Additional CSS classes"
        }
    }
} satisfies Meta<typeof Checkbox>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: {
        "aria-label": "Default checkbox"
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const checkbox = canvas.getByRole("checkbox", {
            name: /default checkbox/i
        });
        await expect(checkbox).toBeVisible();
        await expect(checkbox).not.toBeChecked();

        await userEvent.click(checkbox);
        await expect(args.onCheckedChange).toHaveBeenCalledTimes(1);
    }
};

export const Checked: Story = {
    args: {
        checked: true,
        "aria-label": "Checked checkbox"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const checkbox = canvas.getByRole("checkbox", {
            name: /checked checkbox/i
        });
        await expect(checkbox).toBeChecked();
    }
};

export const Unchecked: Story = {
    args: {
        checked: false,
        "aria-label": "Unchecked checkbox"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const checkbox = canvas.getByRole("checkbox", {
            name: /checked checkbox/i
        });
        await expect(checkbox).not.toBeChecked();
    }
};

export const Disabled: Story = {
    args: {
        disabled: true,
        "aria-label": "Disabled checkbox"
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const checkbox = canvas.getByRole("checkbox", {
            name: /disabled checkbox/i
        });
        await expect(checkbox).toBeDisabled();
        await expect(args.onCheckedChange).not.toHaveBeenCalled();
    }
};

export const DisabledChecked: Story = {
    args: {
        checked: true,
        disabled: true,
        "aria-label": "Disabled checked checkbox"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const checkbox = canvas.getByRole("checkbox", {
            name: /disabled checked checkbox/i
        });
        await expect(checkbox).toBeChecked();
        await expect(checkbox).toBeDisabled();
    }
};

export const WithLabel: Story = {
    render: () => (
        <div className="flex items-center gap-2">
            <Checkbox id="terms" />
            <label
                htmlFor="terms"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                Accept terms and conditions
            </label>
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const checkbox = canvas.getByRole("checkbox");
        await expect(checkbox).not.toBeChecked();
        const label = canvas.getByText(/accept terms and conditions/i);
        await userEvent.click(label);
        await expect(checkbox).toBeChecked();
    }
};

export const WithLabelChecked: Story = {
    render: () => (
        <div className="flex items-center gap-2">
            <Checkbox id="terms-checked" defaultChecked />
            <label
                htmlFor="terms-checked"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                I agree to the terms
            </label>
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const checkbox = canvas.getByRole("checkbox");
        await expect(checkbox).toBeChecked();
        await userEvent.click(canvas.getByText(/i agree to the terms/i));
        await expect(checkbox).not.toBeChecked();
    }
};

export const WithLabelDisabled: Story = {
    render: () => (
        <div className="flex items-center gap-2">
            <Checkbox id="terms-disabled" disabled />
            <label
                htmlFor="terms-disabled"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                Disabled option
            </label>
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const checkbox = canvas.getByRole("checkbox");
        await expect(checkbox).toBeDisabled();
        await expect(checkbox).not.toBeChecked();
    }
};
