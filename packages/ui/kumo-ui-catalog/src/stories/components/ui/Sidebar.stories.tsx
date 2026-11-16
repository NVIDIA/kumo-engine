// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";
import {
    Sidebar,
    SidebarProvider,
    SidebarHeader,
    SidebarContent,
    SidebarGroup,
    SidebarMenu,
    SidebarMenuItem,
    SidebarMenuButton,
    SidebarInset
} from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import { Cards, NetworkNode, Power } from "@/icons";
import { Cube } from "@/icons/MaterialIcons/Cube";

const meta: Meta<typeof Sidebar> = {
    title: "UI/Sidebar",
    component: Sidebar,
    tags: ["autodocs"],
    parameters: {
        layout: "fullscreen"
    }
};

export default meta;

type Story = StoryObj<typeof Sidebar>;

export const MenuButtonVariants: Story = {
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Default (Primary)")).toBeVisible();
        await expect(canvas.getByText("Outline")).toBeVisible();
        await expect(canvas.getByText("Outline Active")).toBeVisible();
        await expect(canvas.getByText("Secondary")).toBeVisible();
        const activeBtn = canvas
            .getByText("Outline Active")
            .closest("[data-active]")!;
        await expect(activeBtn).toHaveAttribute("data-active", "true");
        await userEvent.click(canvas.getByText("Outline"));
    },
    render: () => (
        <SidebarProvider>
            <Sidebar>
                <SidebarHeader>
                    <div className="px-2 text-lg font-semibold">
                        Button Variants
                    </div>
                </SidebarHeader>
                <SidebarContent>
                    <SidebarGroup>
                        <SidebarMenu>
                            <SidebarMenuItem className="my-2">
                                <SidebarMenuButton variant="default">
                                    <Cube className="h-4 w-4" />
                                    Default (Primary)
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroup>
                    <div className="px-3">
                        <Separator className="my-0" />
                    </div>
                    <SidebarGroup>
                        <SidebarMenu>
                            <SidebarMenuItem className="my-2">
                                <SidebarMenuButton variant="outline">
                                    <Cards className="h-4 w-4" />
                                    Outline
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem className="my-2">
                                <SidebarMenuButton
                                    variant="outline"
                                    data-active={true}>
                                    <NetworkNode className="h-4 w-4" />
                                    Outline Active
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroup>
                    <div className="px-3">
                        <Separator className="my-0" />
                    </div>
                    <SidebarGroup>
                        <SidebarMenu>
                            <SidebarMenuItem className="my-2">
                                <SidebarMenuButton variant="secondary">
                                    <Power className="h-4 w-4" />
                                    Secondary
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroup>
                </SidebarContent>
            </Sidebar>
            <SidebarInset>
                <div className="p-4">
                    Demonstrating different SidebarMenuButton variants
                </div>
            </SidebarInset>
        </SidebarProvider>
    )
};
