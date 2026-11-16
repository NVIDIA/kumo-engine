// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TextLink } from "@/components/ui/text-link";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, fn, userEvent, screen, waitFor, within } from "storybook/test";

const meta: Meta<typeof TextLink> = {
    title: "UI/TextLink",
    component: TextLink,
    tags: ["autodocs"],
    args: {
        id: "sample-text-link",
        title: "Learn more",
        pageLink: "https://kumo.ai/docs/",
        tooltipText: "Opens documentation page",
        newTab: true,
        disabled: false
    }
};

export default meta;

type Story = StoryObj<typeof TextLink>;

export const Default: Story = {
    args: {
        onClick: fn()
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const link = canvas.getByRole("link", { name: /learn more/i });

        await expect(link).toHaveAttribute("href", "https://kumo.ai/docs/");
        await expect(link).toHaveAttribute("target", "_blank");

        // Click the link
        await userEvent.click(link);
        await expect(args.onClick).toHaveBeenCalledTimes(1);
    }
};

export const WithTooltip: Story = {
    args: {
        tooltipText: "This will take you to an external site"
    }
};

export const OpenInNewTab: Story = {
    args: {
        newTab: true
    }
};

export const Disabled: Story = {
    args: {
        disabled: true
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const disabledText = canvas.getByText("Learn more");
        await expect(disabledText.tagName).toBe("SPAN");
        expect(canvas.queryByRole("link")).toBeNull();

        await expect(disabledText).toHaveClass("cursor-not-allowed");

        await userEvent.hover(disabledText);

        await waitFor(() => {
            expect(screen.getByRole("tooltip")).toBeVisible();
        });
        expect(screen.getByRole("tooltip")).toHaveTextContent(
            "Opens documentation page"
        );
    }
};

export const CustomStyling: Story = {
    args: {
        className: "text-red-500 underline",
        title: "Styled Link"
    }
};
