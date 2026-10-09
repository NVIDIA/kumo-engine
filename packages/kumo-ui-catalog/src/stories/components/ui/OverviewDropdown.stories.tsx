// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { OverviewDropdown } from "@/components/ui/overview-dropdown";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, fn, screen, userEvent, waitFor, within } from "storybook/test";

const meta: Meta<typeof OverviewDropdown> = {
    title: "UI/OverviewDropdown",
    component: OverviewDropdown,
    tags: ["autodocs"],
    args: {
        id: "sample-overview-dropdown",
        actionItems: [
            {
                content: "Edit",
                disabled: false,
                onSelect: fn(),
                tooltipText: ""
            },
            {
                content: "Duplicate",
                disabled: false,
                onSelect: fn(),
                tooltipText: ""
            },
            {
                content: "Delete",
                disabled: false,
                onSelect: fn(),
                tooltipText: ""
            }
        ]
    }
};

export default meta;

type Story = StoryObj<typeof OverviewDropdown>;

export const Default: Story = {
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getByTestId("more-options-dropdown");

        await userEvent.click(trigger);
        await waitFor(() => expect(screen.getByRole("menu")).toBeVisible());

        const editItem = screen.getByText("Edit");
        await userEvent.click(editItem);
        await expect(args.actionItems[0].onSelect).toHaveBeenCalledTimes(1);
    }
};

export const WithDisabledItem: Story = {
    args: {
        actionItems: [
            {
                content: "Open",
                disabled: false,
                onSelect: fn(),
                tooltipText: ""
            },
            {
                content: "Archive",
                disabled: true,
                onSelect: fn(),
                tooltipText: "Not available for this row"
            },
            {
                content: "Delete",
                disabled: false,
                onSelect: fn(),
                tooltipText: ""
            }
        ]
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(canvas.getByTestId("more-options-dropdown"));
        await waitFor(() => expect(screen.getByRole("menu")).toBeVisible());

        const archive = screen.getByText("Archive");
        await expect(archive.closest('[role="menuitem"]')).toHaveClass(
            "pointer-events-none"
        );
    }
};

export const WithIconLabels: Story = {
    args: {
        actionItems: [
            {
                content: (
                    <>
                        <span aria-hidden="true">✏️</span> Rename
                    </>
                ),
                disabled: false,
                onSelect: fn(),
                tooltipText: ""
            },
            {
                content: (
                    <>
                        <span aria-hidden="true">🗑</span> Delete
                    </>
                ),
                disabled: false,
                onSelect: fn(),
                tooltipText: ""
            }
        ]
    }
};

export const WithCustomButton: Story = {
    args: {
        buttonClassName:
            "w-12 h-12 rounded-full bg-kumo-pink-100 hover:bg-kumo-pink-200 border-0"
    }
};

export const EmptyActions: Story = {
    args: {
        actionItems: []
    }
};
