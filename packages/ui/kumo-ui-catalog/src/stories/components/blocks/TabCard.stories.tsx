// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";

import { TabCard } from "@/components/blocks/card/tab-card";

const meta = {
    title: "Components/Blocks/TabCard",
    component: TabCard,
    tags: ["autodocs"],
    parameters: { layout: "centered" }
} satisfies Meta<typeof TabCard>;

export default meta;
type Story = StoryObj<typeof meta>;

const tabs = [
    { value: 0, label: "Overview" },
    { value: 1, label: "Details" }
];

const renderContent = (currentTab: number) =>
    currentTab === 0 ? <div>Overview content</div> : <div>Details content</div>;

export const Default: Story = {
    args: {
        title: "Model summary",
        tabs,
        children: renderContent
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        // First tab's content is shown by default.
        await expect(canvas.getByText("Overview content")).toBeInTheDocument();

        // Switching tabs swaps the rendered content.
        await userEvent.click(canvas.getByTestId("tab-details"));
        await expect(canvas.getByText("Details content")).toBeInTheDocument();
    }
};

export const Loading: Story = {
    args: {
        title: "Model summary",
        tabs,
        loadingStatus: true,
        children: renderContent
    }
};

export const DefaultSelected: Story = {
    args: {
        title: "Model summary",
        tabs,
        defaultSelected: 1,
        children: renderContent
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Details content")).toBeInTheDocument();
    }
};
