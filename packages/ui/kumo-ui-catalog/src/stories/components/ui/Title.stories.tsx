// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { Title } from "@/components/ui/title";

const meta = {
    title: "UI/Title",
    component: Title,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {
        variant: {
            control: "select",
            options: [
                "page-heading",
                "primary-heading",
                "secondary-heading",
                "tertiary-heading",
                "table-title"
            ],
            description: "The visual style variant of the title"
        },
        children: {
            control: "text",
            description: "The text content of the title"
        }
    }
} satisfies Meta<typeof Title>;

export default meta;
type Story = StoryObj<typeof meta>;

export const PagePrimary: Story = {
    args: {
        variant: "page-heading",
        children: "Dashboard Overview"
    }
};

export const PrimaryHeading: Story = {
    args: {
        variant: "primary-heading",
        children: "Manage your workspace settings and preferences"
    }
};

export const SecondaryHeading: Story = {
    args: {
        variant: "secondary-heading",
        children: "Recent Activity"
    }
};

export const TertiaryHeading: Story = {
    args: {
        variant: "tertiary-heading",
        children: "Last 30 days"
    }
};

export const TableTitle: Story = {
    args: {
        variant: "section-header",
        children: "Optional field"
    }
};
