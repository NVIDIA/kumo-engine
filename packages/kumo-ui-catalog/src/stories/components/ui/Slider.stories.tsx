// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// src/components/ui/slider.stories.tsx
import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { expect, userEvent, within } from "storybook/test";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Volume2, VolumeX } from "lucide-react";

const meta: Meta<typeof Slider> = {
    title: "UI/Slider",
    component: Slider,
    tags: ["autodocs"],
    argTypes: {
        defaultValue: {
            control: "object",
            description: "The initial value of the slider"
        },
        value: {
            control: "object",
            description: "The controlled value of the slider"
        },
        min: {
            control: { type: "number" },
            description: "The minimum value for the range"
        },
        max: {
            control: { type: "number" },
            description: "The maximum value for the range"
        },
        step: {
            control: { type: "number" },
            description: "The stepping interval"
        },
        disabled: {
            control: "boolean",
            description: "Whether the slider is disabled"
        },
        orientation: {
            control: { type: "select", options: ["horizontal", "vertical"] },
            description: "The orientation of the slider"
        }
    }
};

export default meta;
type Story = StoryObj<typeof Slider>;

// Basic Slider
export const Basic: Story = {
    args: {
        defaultValue: [0.5],
        max: 1,
        step: 0.01
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const slider = canvas.getByRole("slider");

        await expect(slider).toHaveAttribute("aria-valuenow", "0.5");

        // Use arrow keys to change value
        slider.focus();
        await userEvent.keyboard("{ArrowRight}");
        await expect(slider).toHaveAttribute("aria-valuenow", "0.51");

        await userEvent.keyboard("{ArrowLeft}");
        await expect(slider).toHaveAttribute("aria-valuenow", "0.5");
    }
};

// Default Value
export const DefaultValue: Story = {
    args: {
        defaultValue: [75],
        max: 100,
        step: 1
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const slider = canvas.getByRole("slider");
        await expect(slider).toHaveAttribute("aria-valuenow", "75");
    }
};

// Min and Max Values
export const MinMax: Story = {
    args: {
        defaultValue: [50],
        min: 20,
        max: 80,
        step: 1
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const slider = canvas.getByRole("slider");
        await expect(slider).toHaveAttribute("aria-valuemin", "20");
        await expect(slider).toHaveAttribute("aria-valuemax", "80");
        await expect(slider).toHaveAttribute("aria-valuenow", "50");
    }
};

// Multiple Values (Range)
export const Range: Story = {
    args: {
        defaultValue: [25, 75],
        max: 100,
        step: 1
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const slider = canvas.getByRole("slider");
        await expect(slider).toHaveAttribute("aria-valuenow", "25");
    }
};

// Disabled Slider
export const Disabled: Story = {
    args: {
        defaultValue: [50],
        max: 1,
        step: 1,
        disabled: true
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const slider = canvas.getByRole("slider");

        await expect(slider).toHaveAttribute("data-disabled");
    }
};

// Step Sizes
export const Steps: Story = {
    render: () => (
        <div className="w-full max-w-md space-y-8">
            <div className="space-y-2">
                <Label>Fine Control (Step 1)</Label>
                <Slider defaultValue={[50]} max={100} step={1} />
            </div>

            <div className="space-y-2">
                <Label>Medium Steps (Step 5)</Label>
                <Slider defaultValue={[50]} max={100} step={5} />
            </div>

            <div className="space-y-2">
                <Label>Coarse Control (Step 10)</Label>
                <Slider defaultValue={[50]} max={100} step={10} />
            </div>

            <div className="space-y-2">
                <Label>Large Steps (Step 25)</Label>
                <Slider defaultValue={[50]} max={100} step={25} />
            </div>
        </div>
    )
};

// With Value Label
export const WithValueLabel: Story = {
    render: function ValueLabelSlider() {
        const [value, setValue] = useState([50]);

        return (
            <div className="w-full max-w-md space-y-4">
                <div className="flex items-center justify-between">
                    <Label>Slider Value</Label>
                    <span className="border-input text-muted-foreground w-12 rounded-md border px-2 py-0.5 text-center text-sm">
                        {value[0]}
                    </span>
                </div>
                <Slider
                    value={value}
                    onValueChange={setValue}
                    max={100}
                    step={1}
                />
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const slider = canvas.getByRole("slider");

        await expect(canvas.getAllByText("50")[0]).toBeVisible();

        slider.focus();
        await userEvent.keyboard("{ArrowRight}");
        await expect(canvas.getAllByText("51")[0]).toBeVisible();

        await userEvent.keyboard("{ArrowLeft}{ArrowLeft}");
        await expect(canvas.getAllByText("49")[0]).toBeVisible();
    }
};

// Volume Control Example
export const VolumeControl: Story = {
    render: function VolumeSlider() {
        const [volume, setVolume] = useState([50]);

        return (
            <div className="w-full max-w-md space-y-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                        {volume[0] === 0 ? (
                            <VolumeX className="text-muted-foreground h-5 w-5" />
                        ) : (
                            <Volume2 className="text-primary h-5 w-5" />
                        )}
                        <Label>Volume</Label>
                    </div>
                    <span className="border-input text-muted-foreground w-12 rounded-md border px-2 py-0.5 text-center text-sm">
                        {volume[0]}%
                    </span>
                </div>
                <Slider
                    value={volume}
                    onValueChange={setVolume}
                    max={100}
                    step={1}
                    className="cursor-pointer"
                />
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const slider = canvas.getByRole("slider");
        await expect(slider).toHaveAttribute("aria-valuenow", "50");
        slider.focus();
        await userEvent.keyboard("{ArrowRight}");
        await expect(slider).toHaveAttribute("aria-valuenow", "51");
    }
};
