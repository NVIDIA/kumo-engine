// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// src/components/ui/switch.stories.tsx
import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { expect, userEvent, within } from "storybook/test";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

const meta: Meta<typeof Switch> = {
    title: "UI/Switch",
    component: Switch,
    tags: ["autodocs"],
    argTypes: {
        checked: {
            control: "boolean",
            description: "Whether the switch is checked"
        },
        disabled: {
            control: "boolean",
            description: "Whether the switch is disabled"
        },
        defaultChecked: {
            control: "boolean",
            description: "The default checked state when initially rendered"
        },
        onCheckedChange: {
            action: "checked changed",
            description: "Function called when the checked state changes"
        }
    }
};

export default meta;
type Story = StoryObj<typeof Switch>;

// Basic Switch
export const Basic: Story = {
    args: {
        "aria-label": "Toggle switch"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const toggle = canvas.getByRole("switch");

        await expect(toggle).toHaveAttribute("data-state", "unchecked");

        await userEvent.click(toggle);
        await expect(toggle).toHaveAttribute("data-state", "checked");

        await userEvent.click(toggle);
        await expect(toggle).toHaveAttribute("data-state", "unchecked");
    }
};

// Default On
export const DefaultOn: Story = {
    args: {
        defaultChecked: true,
        "aria-label": "Toggle switch"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const toggle = canvas.getByRole("switch");
        await expect(toggle).toHaveAttribute("data-state", "checked");
        await userEvent.click(toggle);
        await expect(toggle).toHaveAttribute("data-state", "unchecked");
    }
};

// Disabled
export const Disabled: Story = {
    args: {
        disabled: true,
        "aria-label": "Toggle switch"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const toggle = canvas.getByRole("switch");

        await expect(toggle).toBeDisabled();
        await expect(toggle).toHaveAttribute("data-state", "unchecked");
    }
};

// Disabled Checked
export const DisabledChecked: Story = {
    args: {
        disabled: true,
        defaultChecked: true,
        "aria-label": "Toggle switch"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const toggle = canvas.getByRole("switch");
        await expect(toggle).toBeDisabled();
        await expect(toggle).toHaveAttribute("data-state", "checked");
    }
};

// With Label
export const WithLabel: Story = {
    render: () => (
        <div className="flex items-center space-x-2">
            <Switch id="airplane-mode" />
            <Label htmlFor="airplane-mode">Airplane Mode</Label>
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const toggle = canvas.getByRole("switch");
        await expect(toggle).toHaveAttribute("data-state", "unchecked");
        await userEvent.click(canvas.getByText("Airplane Mode"));
        await expect(toggle).toHaveAttribute("data-state", "checked");
    }
};

// With Label Left
export const WithLabelLeft: Story = {
    render: () => (
        <div className="flex items-center space-x-2">
            <Label htmlFor="wifi-mode">WiFi</Label>
            <Switch id="wifi-mode" />
        </div>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const toggle = canvas.getByRole("switch");
        await expect(toggle).toHaveAttribute("data-state", "unchecked");
        await userEvent.click(canvas.getByText("WiFi"));
        await expect(toggle).toHaveAttribute("data-state", "checked");
    }
};

// Controlled Switch
export const Controlled: Story = {
    render: function ControlledSwitch() {
        const [checked, setChecked] = useState(false);

        return (
            <div className="flex flex-col space-y-4">
                <div className="flex items-center space-x-2">
                    <Switch
                        checked={checked}
                        onCheckedChange={setChecked}
                        id="controlled-switch"
                    />
                    <Label htmlFor="controlled-switch">
                        {checked ? "On" : "Off"}
                    </Label>
                </div>
                <p className="text-muted-foreground text-sm">
                    The switch is {checked ? "on" : "off"}.
                </p>
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const toggle = canvas.getByRole("switch");

        await expect(canvas.getByText("Off")).toBeVisible();
        await expect(canvas.getByText("The switch is off.")).toBeVisible();

        await userEvent.click(toggle);

        await expect(canvas.getByText("On")).toBeVisible();
        await expect(canvas.getByText("The switch is on.")).toBeVisible();

        await userEvent.click(toggle);

        await expect(canvas.getByText("Off")).toBeVisible();
        await expect(canvas.getByText("The switch is off.")).toBeVisible();
    }
};

// Switch Sizes
export const Sizes: Story = {
    render: () => (
        <div className="flex flex-col space-y-4">
            <div className="flex items-center space-x-2">
                <Switch className="h-[18px] w-[34px]" id="small-switch" />
                <Label htmlFor="small-switch" className="text-sm">
                    Small Switch
                </Label>
            </div>

            <div className="flex items-center space-x-2">
                <Switch id="default-switch" />
                <Label htmlFor="default-switch">Default Switch</Label>
            </div>

            <div className="flex items-center space-x-2">
                <Switch className="h-[30px] w-[56px]" id="large-switch" />
                <Label htmlFor="large-switch" className="text-lg">
                    Large Switch
                </Label>
            </div>
        </div>
    )
};

// Variants
export const Variants: Story = {
    render: () => (
        <div className="flex flex-col space-y-4">
            <div className="flex items-center space-x-2">
                <Switch id="default-variant" defaultChecked />
                <Label htmlFor="default-variant">Default (Green)</Label>
            </div>

            <div className="flex items-center space-x-2">
                <Switch id="primary-variant" variant="primary" defaultChecked />
                <Label htmlFor="primary-variant">Primary</Label>
            </div>
        </div>
    )
};

// Custom Colors
export const CustomColors: Story = {
    render: () => (
        <div className="flex flex-col space-y-4">
            <div className="flex items-center space-x-2">
                <Switch
                    className="data-[state=checked]:bg-green-500"
                    id="success-switch"
                    defaultChecked
                />
                <Label htmlFor="success-switch">Success</Label>
            </div>

            <div className="flex items-center space-x-2">
                <Switch
                    className="data-[state=checked]:bg-blue-500"
                    id="info-switch"
                    defaultChecked
                />
                <Label htmlFor="info-switch">Info</Label>
            </div>

            <div className="flex items-center space-x-2">
                <Switch
                    className="data-[state=checked]:bg-yellow-500"
                    id="warning-switch"
                    defaultChecked
                />
                <Label htmlFor="warning-switch">Warning</Label>
            </div>

            <div className="flex items-center space-x-2">
                <Switch
                    className="data-[state=checked]:bg-red-500"
                    id="danger-switch"
                    defaultChecked
                />
                <Label htmlFor="danger-switch">Danger</Label>
            </div>

            <div className="flex items-center space-x-2">
                <Switch
                    className="data-[state=checked]:bg-purple-500"
                    id="purple-switch"
                    defaultChecked
                />
                <Label htmlFor="purple-switch">Purple</Label>
            </div>
        </div>
    )
};
