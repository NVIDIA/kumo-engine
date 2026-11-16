// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CustomAlert } from "@/components/ui/custom-alert";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, fn, screen, userEvent, waitFor } from "storybook/test";

const meta: Meta<typeof CustomAlert> = {
    title: "UI/CustomAlert",
    component: CustomAlert,
    tags: ["autodocs"],
    args: {
        title: "Error",
        description: "Something went wrong while saving your changes.",
        show: true,
        handleClose: fn()
    }
};

export default meta;

type Story = StoryObj<typeof CustomAlert>;

export const Default: Story = {
    play: async () => {
        // Dialog renders in a portal; assert against document body
        const title = await screen.findByText("Error");
        await expect(title).toBeVisible();
        await expect(
            screen.getByText("Something went wrong while saving your changes.")
        ).toBeVisible();
        const cancelBtn = screen.getByRole("button", { name: /cancel/i });
        await expect(cancelBtn).toBeVisible();
    }
};

export const Closed: Story = {
    args: { show: false },
    play: async () => {
        // No dialog content visible when show=false
        await expect(screen.queryByRole("dialog")).toBeNull();
    }
};

export const HandleCloseInvoked: Story = {
    play: async ({ args }) => {
        const cancelBtn = await screen.findByRole("button", {
            name: /cancel/i
        });
        await userEvent.click(cancelBtn);
        await waitFor(() => expect(args.handleClose).toHaveBeenCalledTimes(1));
    }
};

export const LongDescription: Story = {
    args: {
        title: "Validation failed",
        description:
            "There are several issues with the input. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
    }
};
