// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TagList } from "@/components/ui/tag-list";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";
import { useState } from "react";

const meta: Meta<typeof TagList> = {
    title: "UI/TagList",
    component: TagList,
    tags: ["autodocs"],
    parameters: {
        layout: "centered"
    },
    argTypes: {
        showHeader: { control: "boolean" },
        noWrap: { control: "boolean" },
        onTagAdd: { action: "tagAdded" },
        onTagRemove: { action: "tagRemoved" }
    }
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    render: function DefaultRender() {
        const [tags, setTags] = useState<Record<string, string>>({
            env: "production",
            team: "ml"
        });
        return (
            <TagList
                tags={tags}
                onTagAdd={(key, value) =>
                    setTags((prev) => ({ ...prev, [key]: value ?? "" }))
                }
                onTagRemove={(key) =>
                    setTags((prev) => {
                        const next = { ...prev };
                        delete next[key];
                        return next;
                    })
                }
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("env=")).toBeVisible();
        await expect(canvas.getByText("production")).toBeVisible();
        await expect(canvas.getByText("team=")).toBeVisible();
    }
};

export const Empty: Story = {
    render: function EmptyRender() {
        const [tags, setTags] = useState<Record<string, string>>({});
        return (
            <TagList
                tags={tags}
                onTagAdd={(key, value) =>
                    setTags((prev) => ({ ...prev, [key]: value ?? "" }))
                }
                onTagRemove={(key) =>
                    setTags((prev) => {
                        const next = { ...prev };
                        delete next[key];
                        return next;
                    })
                }
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Add tag")).toBeVisible();
    }
};

export const WithProtectedKeys: Story = {
    render: function ProtectedRender() {
        const [tags, setTags] = useState<Record<string, string>>({
            owner: "kumo",
            region: "us-east-1"
        });
        return (
            <TagList
                tags={tags}
                protectedKeys={new Set(["owner"])}
                onTagAdd={(key, value) =>
                    setTags((prev) => ({ ...prev, [key]: value ?? "" }))
                }
                onTagRemove={(key) =>
                    setTags((prev) => {
                        const next = { ...prev };
                        delete next[key];
                        return next;
                    })
                }
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("owner=")).toBeVisible();
        // region tag has a remove icon, owner does not
        const removeIcons = canvas.getAllByTestId("tag-remove-icon");
        await expect(removeIcons).toHaveLength(1);
    }
};
