// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Image } from "@/components/ui/image";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";

const meta: Meta<typeof Image> = {
    title: "UI/Image",
    component: Image,
    tags: ["autodocs"],
    args: {
        src: "https://placehold.co/64x64",
        name: "Sample placeholder image",
        width: "64"
    }
};

export default meta;

type Story = StoryObj<typeof Image>;

export const Default: Story = {
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const img = canvas.getByAltText(args.name);

        await expect(img).toBeVisible();
        await expect(img).toHaveAttribute("src", args.src);
        await expect(img).toHaveAttribute("width", "64");
    }
};

export const DefaultWidth: Story = {
    args: {
        width: undefined
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const img = canvas.getByAltText(args.name);

        // Falls back to default width "28"
        await expect(img).toHaveAttribute("width", "28");
    }
};

export const ConnectorLogo: Story = {
    args: {
        src: "https://placehold.co/120x40/png?text=Snowflake",
        name: "Snowflake connector",
        width: "120"
    }
};
