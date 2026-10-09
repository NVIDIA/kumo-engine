// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";

const meta: Meta<typeof Tabs> = {
    title: "UI/Tabs",
    component: Tabs,
    tags: ["autodocs"]
};

export default meta;
type Story = StoryObj<typeof Tabs>;

export const Default: Story = {
    render: () => {
        return (
            <Tabs defaultValue="tab1">
                <TabsList>
                    <TabsTrigger className="pb-1" value="tab1">
                        Output table schema
                    </TabsTrigger>
                    <TabsTrigger className="pb-1" value="tab2">
                        Graph schema
                    </TabsTrigger>
                    <TabsTrigger className="pb-1" value="tab3">
                        Tab 3
                    </TabsTrigger>
                </TabsList>
                <TabsContent value="tab1">
                    Content for Output table schema
                </TabsContent>
                <TabsContent value="tab2">Content for Graph schema</TabsContent>
                <TabsContent value="tab3">Content for Tab 3</TabsContent>
            </Tabs>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const tab1 = canvas.getByRole("tab", { name: /output table schema/i });
        await expect(tab1).toHaveAttribute("data-state", "active");
        await expect(
            canvas.getByText("Content for Output table schema")
        ).toBeVisible();

        const tab2 = canvas.getByRole("tab", { name: /graph schema/i });
        await userEvent.click(tab2);
        await expect(tab2).toHaveAttribute("data-state", "active");
        await expect(tab1).toHaveAttribute("data-state", "inactive");
        await expect(
            canvas.getByText("Content for Graph schema")
        ).toBeVisible();

        const tab3 = canvas.getByRole("tab", { name: /tab 3/i });
        await userEvent.click(tab3);
        await expect(tab3).toHaveAttribute("data-state", "active");
        await expect(tab2).toHaveAttribute("data-state", "inactive");
        await expect(canvas.getByText("Content for Tab 3")).toBeVisible();
    }
};
