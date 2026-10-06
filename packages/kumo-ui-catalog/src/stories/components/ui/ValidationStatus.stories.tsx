// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ValidationStatus } from "@/components/ui/validation-status";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";

const meta: Meta<typeof ValidationStatus> = {
    title: "UI/ValidationStatus",
    component: ValidationStatus,
    tags: ["autodocs"],
    parameters: { layout: "centered" }
};

export default meta;

type Story = StoryObj<typeof ValidationStatus>;

export const Empty: Story = {
    args: { state: "none" },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const pill = canvas.getByTestId("validation-status");
        await expect(pill).toHaveAttribute("data-state", "none");
        await expect(canvas.getByText("Empty Query")).toBeVisible();
    }
};

export const CustomEmptyText: Story = {
    args: { state: "none", emptyText: "Type a query to validate" }
};

export const Valid: Story = {
    args: { state: "valid" },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Looks good!")).toBeVisible();
    }
};

export const Validating: Story = {
    args: { state: "validating" }
};

export const Warning: Story = {
    args: { state: "warning" }
};

export const Invalid: Story = {
    args: { state: "invalid" },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Issues found")).toBeVisible();
    }
};

export const WithHoverContent: Story = {
    args: {
        state: "invalid",
        hoverContent: (
            <div className="text-title-primary bg-red-50 p-3 text-sm">
                <strong>Schema mismatch:</strong> column &quot;id&quot; not
                found.
            </div>
        )
    }
};
