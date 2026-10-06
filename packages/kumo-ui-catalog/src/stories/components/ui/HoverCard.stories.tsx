// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Meta, StoryObj } from "@storybook/react-vite";
import { Button } from "@/components/ui/button";
import {
    HoverCard,
    HoverCardContent,
    HoverCardTrigger
} from "@/components/ui/hover-card";
import { expect, userEvent, waitFor, within } from "storybook/test";

const meta: Meta<typeof HoverCard> = {
    title: "UI/HoverCard",
    component: HoverCard,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {}
};

export default meta;
type Story = StoryObj<typeof HoverCard>;

export const HoverCardWithoutArrow: Story = {
    render: () => (
        <HoverCard>
            <HoverCardTrigger>
                <Button variant="light-outline">Hover over me</Button>
            </HoverCardTrigger>

            <HoverCardContent arrow={false}>
                <div className="text-Primary flex-1 justify-start whitespace-pre-line font-['Source_Code_Pro'] text-sm font-semibold leading-tight">
                    {
                        "PREDICT LIST_DISTINCT(transactions.sales_channel_id, 0, 90\nCLASSIFY TOP 2 \nFOR EACH customers.customer_id"
                    }
                </div>
            </HoverCardContent>
        </HoverCard>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getByRole("button", { name: "Hover over me" });
        await userEvent.hover(trigger);
        const body = within(document.body);
        await expect(body.getByText(/PREDICT LIST_DISTINCT/i)).toBeVisible();

        await userEvent.unhover(trigger);
        await waitFor(() => {
            expect(
                body.queryByText(/PREDICT LIST_DISTINCT/i)
            ).not.toBeInTheDocument();
        });
    }
};

export const HoverCardWithArrow: Story = {
    render: () => (
        <HoverCard>
            <HoverCardTrigger>
                <Button variant="light-outline">Hover over me</Button>
            </HoverCardTrigger>
            <HoverCardContent arrow={true}>
                <div className="flex-1 justify-start whitespace-pre-line font-['Source_Code_Pro'] text-sm font-semibold leading-tight text-gray-800">
                    {
                        "PREDICT LIST_DISTINCT(transactions.sales_channel_id, 0, 90\nCLASSIFY TOP 2 \nFOR EACH customers.customer_id"
                    }
                </div>
            </HoverCardContent>
        </HoverCard>
    )
};
