// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within, waitFor } from "storybook/test";
import { WarningsPopover } from "@/components/blocks/warnings-popover";

const meta = {
    title: "Components/Blocks/WarningsPopover",
    component: WarningsPopover,
    tags: ["autodocs"],
    parameters: { layout: "centered" },
    args: {
        warnings: [
            {
                description:
                    "Feature 'age' has a high missing rate (42%). Consider imputation."
            }
        ]
    }
} satisfies Meta<typeof WarningsPopover>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const MultipleWarnings: Story = {
    args: {
        warnings: [
            {
                description:
                    "Feature 'age' has a high missing rate (42%). Consider imputation."
            },
            {
                description:
                    "Feature 'income' distribution has shifted significantly from training data."
            },
            {
                description:
                    "Low prediction confidence detected for 15% of samples."
            }
        ]
    }
};

export const SingleWarning: Story = {
    args: {
        warnings: [
            {
                description:
                    "Model accuracy has dropped below the recommended threshold."
            }
        ]
    }
};

export const ManyWarnings: Story = {
    args: {
        warnings: Array.from({ length: 10 }, (_, i) => ({
            description: `Warning ${i + 1}: This is a sample warning message for testing overflow behavior.`
        }))
    }
};

export const OpenPopover: Story = {
    args: {
        warnings: [
            { description: "Data drift detected in feature 'revenue'." },
            { description: "Missing values exceed 30% for feature 'region'." }
        ]
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(canvas.getByRole("button"));
        await waitFor(() => {
            const body = within(document.body);
            expect(body.getByText("2 warnings detected")).toBeVisible();
        });
    }
};
