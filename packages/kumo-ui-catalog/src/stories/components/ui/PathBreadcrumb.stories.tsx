// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PathBreadcrumb } from "@/components/ui/path-breadcrumb";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";

const meta: Meta<typeof PathBreadcrumb> = {
    title: "UI/PathBreadcrumb",
    component: PathBreadcrumb,
    tags: ["autodocs"],
    parameters: { layout: "centered" }
};

export default meta;

type Story = StoryObj<typeof PathBreadcrumb>;

export const Default: Story = {
    args: {
        paths: ["Projects", "My Project", "Restore"]
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Projects")).toBeVisible();
        await expect(canvas.getByText("My Project")).toBeVisible();
        // Last item is the current page (aria-current="page")
        const current = canvas.getByText("Restore");
        await expect(current).toBeVisible();
        await expect(current).toHaveAttribute("aria-current", "page");
    }
};

export const SingleSegment: Story = {
    args: { paths: ["Home"] },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const current = canvas.getByText("Home");
        await expect(current).toHaveAttribute("aria-current", "page");
    }
};

export const Empty: Story = {
    args: { paths: [] },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        // Returns null when no paths
        await expect(canvas.queryByRole("navigation")).toBeNull();
    }
};

export const DeepPath: Story = {
    args: {
        paths: ["Workspace", "Team", "Project", "Job", "Run #123", "Logs"]
    }
};
