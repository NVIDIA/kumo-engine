// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, screen, waitFor, within } from "storybook/test";
import {
    Tooltip,
    TooltipTrigger,
    TooltipContent
} from "@/components/ui/tooltip";

const meta: Meta<typeof Tooltip> = {
    title: "UI/Tooltip",
    component: Tooltip,
    tags: ["autodocs"],
    parameters: {
        layout: "centered"
    }
};

export default meta;
type Story = StoryObj<typeof Tooltip>;

export const Default: Story = {
    render: () => (
        <Tooltip>
            <TooltipTrigger>
                <div className="cursor-pointer text-sm text-slate-700">
                    {"Hover me"}
                </div>
            </TooltipTrigger>
            <TooltipContent>{"Tooltip message here"}</TooltipContent>
        </Tooltip>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getByText("Hover me");

        expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

        await userEvent.hover(trigger);

        await waitFor(() => {
            expect(screen.getByRole("tooltip")).toBeVisible();
        });
        expect(screen.getByRole("tooltip")).toHaveTextContent(
            "Tooltip message here"
        );

        await userEvent.unhover(trigger);

        await waitFor(() => {
            expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
        });
    }
};

export const LightTooltip: Story = {
    render: () => (
        <Tooltip>
            <TooltipTrigger>
                <div className="cursor-pointer text-sm text-slate-700">
                    {"Hover me"}
                </div>
            </TooltipTrigger>
            <TooltipContent variant="light">
                {"Tooltip message here"}
            </TooltipContent>
        </Tooltip>
    )
};
