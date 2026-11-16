// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { ProgressBar } from "@/animations/progress-bar";

const meta: Meta<typeof ProgressBar> = {
    title: "UI/ProgressBar",
    component: ProgressBar,
    tags: ["autodocs"],
    argTypes: {
        progress: {
            control: { type: "number", min: 0, max: 100, step: 1 },
            description: "Progress percentage (0-100)",
            defaultValue: 50
        },
        color: {
            control: "color",
            description: "Color used for progress fill stripes",
            defaultValue: "#3b82f6"
        },
        className: {
            control: "text",
            description: "Additional Tailwind classes for outer container"
        }
    }
};

export default meta;
type Story = StoryObj<typeof ProgressBar>;

// Default story
export const Default: Story = {
    args: {
        progress: 50,
        color: "#3b82f6" // Tailwind blue-500
    }
};

// Success color
export const Success: Story = {
    args: {
        progress: 75,
        color: "#22c55e", // Tailwind green-500
        className: "rounded-md border border-gray-200"
    }
};

// Danger color
export const Danger: Story = {
    args: {
        progress: 30,
        color: "#ef4444", // Tailwind red-500
        className: "rounded-md border border-gray-200"
    }
};
