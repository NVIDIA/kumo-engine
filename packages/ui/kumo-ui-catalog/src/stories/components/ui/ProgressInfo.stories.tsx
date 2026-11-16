// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ProgressInfo } from "@/components/ui/progress-info";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";

const meta: Meta<typeof ProgressInfo> = {
    title: "UI/ProgressInfo",
    component: ProgressInfo,
    tags: ["autodocs"],
    parameters: { layout: "centered" }
};

export default meta;

type Story = StoryObj<typeof ProgressInfo>;

export const Default: Story = {
    args: {
        info: [
            { row1: "Rows processed", row2: "1,234 / 10,000" },
            { row1: "Elapsed", row2: "00:01:23" }
        ]
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const container = canvas.getByTestId("progress-info-container");
        await expect(container).toBeVisible();
        await expect(canvas.getByText("Rows processed")).toBeVisible();
        await expect(canvas.getByText("1,234 / 10,000")).toBeVisible();
        await expect(canvas.getByText("Elapsed")).toBeVisible();
    }
};

export const SingleEntry: Story = {
    args: {
        info: [{ row1: "Status", row2: "Running" }]
    }
};

export const EmptyInfo: Story = {
    args: { info: [] },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const container = canvas.getByTestId("progress-info-container");
        // Container renders even when empty
        await expect(container).toBeVisible();
        await expect(container.children.length).toBe(0);
    }
};

export const RichNodes: Story = {
    args: {
        info: [
            {
                row1: <strong>Bold label</strong>,
                row2: (
                    <span style={{ color: "var(--kumo-pink-600)" }}>
                        Styled value
                    </span>
                )
            }
        ]
    }
};
