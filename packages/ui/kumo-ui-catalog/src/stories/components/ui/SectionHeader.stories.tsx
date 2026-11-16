// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SectionHeader } from "@/components/ui/section-header";

import type { Meta, StoryObj } from "@storybook/react-vite";

const meta: Meta<typeof SectionHeader> = {
    title: "UI/SectionHeader",
    component: SectionHeader,
    tags: ["autodocs"],
    argTypes: {
        titleVariant: {
            control: { type: "select" },
            options: ["default", "secondary"],
            defaultValue: "default"
        },
        titleSize: {
            control: { type: "select" },
            options: ["xs", "sm", "md", "lg", "xl", "xxl"],
            defaultValue: "xxl"
        },
        descriptionVariant: {
            control: { type: "select" },
            options: ["default", "secondary"],
            defaultValue: "secondary"
        },
        descriptionSize: {
            control: { type: "select" },
            options: ["xs", "sm", "md", "lg"],
            defaultValue: "sm"
        },
        titleLetterSpacing: {
            control: { type: "number", min: 0, step: 0.1 },
            description: "Letter spacing in pixels for title",
            defaultValue: 0
        }
    }
};

export default meta;
type Story = StoryObj<typeof SectionHeader>;

export const Default: Story = {
    args: {
        title: "Title",
        subtitle: "Subtitle",
        description: "This is a Description"
    }
};
