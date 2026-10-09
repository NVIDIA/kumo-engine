// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";
import kumoLogo from "@/images/kumo_logo_small.svg";

const meta = {
    title: "UI/Avatar",
    component: Avatar,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {
        className: {
            control: { type: "text" },
            description: "Additional CSS classes for the avatar container"
        }
    }
} satisfies Meta<typeof Avatar>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithImage: Story = {
    render: (args) => (
        <Avatar {...args}>
            <AvatarImage src={kumoLogo} alt="@shadcn" />
            <AvatarFallback>CN</AvatarFallback>
        </Avatar>
    )
};

export const WithFallback: Story = {
    render: (args) => (
        <Avatar {...args}>
            <AvatarImage src="/broken-image.jpg" alt="User" />
            <AvatarFallback>JD</AvatarFallback>
        </Avatar>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("JD")).toBeVisible();
    }
};

export const FallbackOnly: Story = {
    render: (args) => (
        <Avatar {...args}>
            <AvatarFallback>AB</AvatarFallback>
        </Avatar>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("AB")).toBeVisible();
    }
};

export const CustomSize: Story = {
    render: (args) => (
        <Avatar className="size-16" {...args}>
            <AvatarImage src={kumoLogo} alt="@shadcn" />
            <AvatarFallback>CN</AvatarFallback>
        </Avatar>
    )
};

export const SmallSize: Story = {
    render: (args) => (
        <Avatar className="size-6" {...args}>
            <AvatarImage src={kumoLogo} alt="@shadcn" />
            <AvatarFallback>CN</AvatarFallback>
        </Avatar>
    )
};

export const AvatarGroup: Story = {
    render: () => (
        <div className="flex -space-x-2">
            <Avatar className="border-background border-2">
                <AvatarImage src={kumoLogo} alt="User 1" />
                <AvatarFallback>U1</AvatarFallback>
            </Avatar>
            <Avatar className="border-background border-2">
                <AvatarFallback>U2</AvatarFallback>
            </Avatar>
            <Avatar className="border-background border-2">
                <AvatarFallback>U3</AvatarFallback>
            </Avatar>
        </div>
    )
};
