// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryCard } from "@/components/ui/query-card";
import { BookSpark, CalendarBasic } from "@/icons";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";
import { Code, Network } from "lucide-react";

const meta: Meta<typeof QueryCard> = {
    title: "UI/QueryCard",
    component: QueryCard,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {}
};

export default meta;
type Story = StoryObj<typeof QueryCard>;

export const QueryCarddefault: Story = {
    render: () => (
        <QueryCard
            id={"query-id"}
            content={
                "PREDICT LIST_DISTINCT(transactions.sales_channel_id, 0, 90\nCLASSIFY TOP 2 \nFOR EACH customers.customer_id"
            }
            copyEnabled={true}
        />
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(
            canvas.getByRole("button", { name: /copy/i })
        ).toBeVisible();
    }
};

export const QueryCardWithTitle: Story = {
    render: () => (
        <QueryCard
            id={"query-id"}
            title={"Predictive query"}
            titleIcon={<Code size={16} color="#6E84A3" />}
            content={
                "PREDICT LIST_DISTINCT(transactions.sales_channel_id, 0, 90\nCLASSIFY TOP 2 \nFOR EACH customers.customer_id"
            }
            copyEnabled={true}
        />
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Predictive query")).toBeVisible();
        await userEvent.click(canvas.getByRole("button", { name: /copy/i }));
    }
};

const propertyList = [
    {
        icon: <Network size={16} color="#6E84A3" />,
        label: "Web"
    },
    {
        icon: <CalendarBasic className="size-4" />,
        label: "6 days ago"
    },
    {
        icon: <BookSpark className="size-4" />,
        label: "test"
    }
];

export const QueryCardWithFooter: Story = {
    render: () => (
        <QueryCard
            id={"query-id"}
            title={"Predictive query"}
            titleIcon={<Code size={20} color={"#6E84A3"} />}
            content={
                "PREDICT LIST_DISTINCT(transactions.sales_channel_id, 0, 90\nCLASSIFY TOP 2 \nFOR EACH customers.customer_id"
            }
            propertyList={propertyList}
            className={"border-b-0"}
            copyEnabled={true}
        />
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Web")).toBeVisible();
        await expect(canvas.getByText("6 days ago")).toBeVisible();
    }
};
