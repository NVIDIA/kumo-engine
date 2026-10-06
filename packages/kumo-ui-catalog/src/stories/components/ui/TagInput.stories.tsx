// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TagInput } from "@/components/ui/tag-input";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";
import { useState } from "react";

const meta: Meta<typeof TagInput> = {
    title: "UI/TagInput",
    component: TagInput,
    tags: ["autodocs"],
    parameters: {
        layout: "centered"
    },
    argTypes: {
        placeholder: { control: "text" },
        autoFocus: { control: "boolean" },
        onChange: { action: "changed" },
        onKeyDown: { action: "keyDown" },
        onBlur: { action: "blurred" }
    }
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    render: function DefaultRender() {
        const [value, setValue] = useState("");
        return (
            <TagInput
                value={value}
                onChange={(e) => setValue(e.target.value)}
                onKeyDown={() => {}}
                onBlur={() => {}}
                placeholder="key"
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByPlaceholderText("key")).toBeVisible();
    }
};
