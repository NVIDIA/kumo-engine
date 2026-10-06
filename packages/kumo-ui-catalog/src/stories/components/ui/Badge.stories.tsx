// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge } from "@/components/ui/badge";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";

// Define metadata for Badge component
const meta = {
    title: "UI/Badge",
    component: Badge,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {
        variant: {
            control: "select",
            options: [
                "default",
                "done",
                "queued",
                "running",
                "cancelled",
                "failed",
                "pink"
            ]
        },
        size: {
            control: "select",
            options: ["sm", "md", "lg"]
        },
        className: {
            control: { type: "text" },
            defaultValue: ""
        }
    }
} satisfies Meta<typeof Badge>;

export default meta;
type Story = StoryObj<typeof meta>;

// Basic Variants
export const BadgeDefault: Story = {
    args: {
        children: "Default Badge",
        variant: "default",
        size: "sm"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Default Badge")).toBeVisible();
    }
};

export const BadgeDone: Story = {
    args: {
        children: "Done",
        variant: "done",
        size: "sm"
    }
};

export const BadgeQueued: Story = {
    args: {
        children: "Queued",
        variant: "queued",
        size: "sm"
    }
};

export const BadgeRunning: Story = {
    args: {
        children: "Running",
        variant: "running",
        size: "sm"
    }
};

export const BadgeCancelled: Story = {
    args: {
        children: "Cancelled",
        variant: "cancelled",
        size: "sm"
    }
};

export const BadgeFalied: Story = {
    args: {
        children: "Failed",
        variant: "failed",
        size: "sm"
    }
};

export const BadgePink: Story = {
    args: {
        children: "mock-tag",
        variant: "pink",
        size: "sm"
    }
};

export const AsChild: Story = {
    render: () => (
        <Badge asChild variant="done" size="sm">
            <a href="#">Link Badge</a>
        </Badge>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const link = canvas.getByRole("link", { name: /link badge/i });
        await expect(link).toBeVisible();
        await expect(link).toHaveAttribute("data-slot", "badge");
    }
};
