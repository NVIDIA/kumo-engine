// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// src/components/ui/label.stories.tsx
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";

const meta: Meta<typeof Label> = {
    title: "UI/Label",
    component: Label,
    tags: ["autodocs"],
    argTypes: {
        /*disabled: {
            control: "boolean",
            description: "Whether the label is disabled"
        },*/
        htmlFor: {
            control: "text",
            description: "The ID of the associated form control"
        }
    }
};

export default meta;
type Story = StoryObj<typeof Label>;

// Basic Label
export const Basic: Story = {
    args: {
        children: "Name"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Name")).toHaveAttribute(
            "data-slot",
            "label"
        );
    }
};

// Label with Input
export const WithInput: Story = {
    render: () => (
        <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="email">Email</Label>
            <Input type="email" id="email" placeholder="Email address" />
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(canvas.getByText("Email"));
        await expect(
            canvas.getByPlaceholderText("Email address")
        ).toHaveFocus();
    }
};

// Required Label
export const Required: Story = {
    render: () => (
        <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="required-input">
                Username <span className="text-red-500">*</span>
            </Label>
            <Input
                id="required-input"
                placeholder="Enter your username"
                required
            />
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const input = canvas.getByPlaceholderText("Enter your username");

        await expect(input).toBeRequired();

        await userEvent.click(canvas.getByText(/username/i));
        await expect(input).toHaveFocus();
    }
};

// Label with Description
export const WithDescription: Story = {
    render: () => (
        <div className="grid w-full max-w-sm items-center gap-1.5">
            <div className="space-y-1">
                <Label htmlFor="description-input">Display name</Label>
                <p className="text-muted-foreground text-xs">
                    This is your public display name. It can be your real name
                    or a pseudonym.
                </p>
            </div>
            <Input
                id="description-input"
                placeholder="Enter your display name"
            />
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(canvas.getByText("Display name"));
        await expect(
            canvas.getByPlaceholderText("Enter your display name")
        ).toHaveFocus();
    }
};

// Disabled Label
export const Disabled: Story = {
    render: () => (
        <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="disabled-input" /*disabled*/>Disabled field</Label>
            <Input id="disabled-input" disabled placeholder="Disabled input" />
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await expect(
            canvas.getByPlaceholderText("Disabled input")
        ).toBeDisabled();
    }
};

// With Checkbox
export const WithCheckbox: Story = {
    render: () => (
        <div className="flex items-center space-x-2">
            <Checkbox id="terms" />
            <Label htmlFor="terms">Accept terms and conditions</Label>
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const checkbox = canvas.getByRole("checkbox");

        await expect(checkbox).not.toBeChecked();

        await userEvent.click(canvas.getByText(/accept terms and conditions/i));
        await expect(checkbox).toBeChecked();

        await userEvent.click(canvas.getByText(/accept terms and conditions/i));
        await expect(checkbox).not.toBeChecked();
    }
};

// With Switch
export const WithSwitch: Story = {
    render: () => (
        <div className="flex items-center space-x-2">
            <Switch id="airplane-mode" />
            <Label htmlFor="airplane-mode">Airplane Mode</Label>
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const switchControl = canvas.getByRole("switch");

        await expect(switchControl).not.toBeChecked();

        await userEvent.click(canvas.getByText("Airplane Mode"));
        await expect(switchControl).toBeChecked();

        await userEvent.click(canvas.getByText("Airplane Mode"));
        await expect(switchControl).not.toBeChecked();
    }
};

// Different Sizes
export const Sizes: Story = {
    render: () => (
        <div className="grid w-full max-w-sm gap-5">
            <div className="space-y-1">
                <Label htmlFor="small-input" className="text-xs">
                    Small Label
                </Label>
                <Input
                    id="small-input"
                    className="h-8 text-xs"
                    placeholder="Small input"
                />
            </div>

            <div className="space-y-1">
                <Label htmlFor="default-input">Default Label</Label>
                <Input id="default-input" placeholder="Default input" />
            </div>

            <div className="space-y-1">
                <Label htmlFor="large-input" className="text-lg">
                    Large Label
                </Label>
                <Input
                    id="large-input"
                    className="h-12 text-lg"
                    placeholder="Large input"
                />
            </div>
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const smallLabel = canvas.getByText("Small Label");
        await userEvent.click(smallLabel);
        await expect(canvas.getByPlaceholderText("Small input")).toHaveFocus();

        const largeLabel = canvas.getByText("Large Label");
        await userEvent.click(largeLabel);
        await expect(canvas.getByPlaceholderText("Large input")).toHaveFocus();
    }
};

// Label Positions
export const Positions: Story = {
    render: () => (
        <div className="grid w-full max-w-sm gap-6">
            {/* Top positioned label (default) */}
            <div className="space-y-1.5">
                <Label htmlFor="top-input">Top positioned label</Label>
                <Input id="top-input" placeholder="Input with label on top" />
            </div>

            {/* Left positioned label */}
            <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="left-input" className="text-right">
                    Left label
                </Label>
                <Input
                    id="left-input"
                    className="col-span-3"
                    placeholder="Input with label on left"
                />
            </div>

            {/* Right positioned label */}
            <div className="grid grid-cols-4 items-center gap-4">
                <Input
                    id="right-input"
                    className="col-span-3"
                    placeholder="Input with label on right"
                />
                <Label htmlFor="right-input">Right label</Label>
            </div>

            {/* Bottom positioned label */}
            <div className="flex flex-col-reverse space-y-1.5">
                <Label htmlFor="bottom-input">Bottom positioned label</Label>
                <Input
                    id="bottom-input"
                    placeholder="Input with label on bottom"
                />
            </div>
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(canvas.getByText("Top positioned label"));
        await expect(
            canvas.getByPlaceholderText("Input with label on top")
        ).toHaveFocus();

        await userEvent.click(canvas.getByText("Left label"));
        await expect(
            canvas.getByPlaceholderText("Input with label on left")
        ).toHaveFocus();

        await userEvent.click(canvas.getByText("Right label"));
        await expect(
            canvas.getByPlaceholderText("Input with label on right")
        ).toHaveFocus();

        await userEvent.click(canvas.getByText("Bottom positioned label"));
        await expect(
            canvas.getByPlaceholderText("Input with label on bottom")
        ).toHaveFocus();
    }
};
