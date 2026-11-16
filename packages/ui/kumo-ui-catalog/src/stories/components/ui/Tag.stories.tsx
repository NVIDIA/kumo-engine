// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Tag } from "@/components/ui/tag";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";

const meta: Meta<typeof Tag> = {
    title: "UI/Tag",
    component: Tag,
    tags: ["autodocs"],
    parameters: {
        layout: "centered"
    },
    argTypes: {
        mode: {
            control: "select",
            options: ["display", "add", "edit"]
        },
        tagKey: { control: "text" },
        value: { control: "text" },
        onRemove: { action: "removed" },
        onSave: { action: "saved" },
        onCancel: { action: "cancelled" }
    }
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Display: Story = {
    args: {
        tagKey: "env",
        value: "production",
        mode: "display"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("env=")).toBeVisible();
        await expect(canvas.getByText("production")).toBeVisible();
    }
};

export const DisplayWithRemove: Story = {
    args: {
        tagKey: "team",
        value: "ml",
        mode: "display",
        onRemove: () => {}
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("team=")).toBeVisible();
        await expect(canvas.getByTestId("tag-remove-icon")).toBeVisible();
    }
};

export const ProtectedTag: Story = {
    args: {
        tagKey: "owner",
        value: "kumo",
        mode: "display",
        onRemove: () => {},
        protectedKeys: new Set(["owner"])
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("owner=")).toBeVisible();
        await expect(canvas.queryByTestId("tag-remove-icon")).toBeNull();
    }
};

export const AddMode: Story = {
    args: {
        mode: "add"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Add tag")).toBeVisible();
    }
};
