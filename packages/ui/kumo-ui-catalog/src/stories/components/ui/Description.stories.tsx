// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Description } from "@/components/ui/description";
import { Meta, StoryObj } from "@storybook/react-vite";

const meta = {
    title: "UI/Description",
    component: Description,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {
        variant: {
            control: "select",
            options: ["default", "secondary", "success", "destructive"],
            description: "The visual style variant"
        },
        size: {
            control: "select",
            options: ["xs", "sm", "md", "lg"],
            description: "The text size"
        },
        children: {
            control: "text",
            description: "The description text content"
        }
    }
} satisfies Meta<typeof Description>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: {
        children: 'Are you sure you want to restore "My Project"?',
        size: "sm"
    }
};

export const Secondary: Story = {
    args: {
        children:
            "Enter your Databricks warehouse and client id/client secret details.",
        variant: "secondary",
        size: "sm",
        className: "mb-1"
    }
};

export const Destructive: Story = {
    args: {
        children: "Private Key is required!",
        variant: "destructive",
        size: "sm",
        className: "!mb-0"
    }
};

export const ConfirmationDialog: Story = {
    args: {
        children: "This will make the project and its assets active again.",
        variant: "secondary",
        size: "sm"
    },
    decorators: [
        (Story) => (
            <div className="w-[400px]">
                <Description size="sm">
                    Are you sure you want to restore "My Project"?
                </Description>
                <Story />
            </div>
        )
    ]
};

export const HelperText: Story = {
    args: {
        children:
            "In order to connect your databricks warehouse, you need to provide the client id/client secret of a service principal account or the personal access token (PAT) of a regular account in your workspace.",
        variant: "secondary",
        size: "sm",
        className: "leading-5"
    },
    decorators: [
        (Story) => (
            <div className="w-[500px]">
                <Story />
            </div>
        )
    ]
};

export const PageSubtitle: Story = {
    args: {
        children: "Project created on Jan 15, 2024",
        variant: "secondary",
        size: "sm",
        className: "my-0"
    }
};

export const NoteText: Story = {
    args: {
        children:
            "Note: Only '.csv' or '.parquet' files are allowed with max size 1.0 GB.",
        variant: "secondary",
        size: "sm"
    }
};

export const FormInstruction: Story = {
    args: {
        children:
            "Select the columns & any preprocessing you want to apply to new table.",
        size: "sm",
        className: "m-0 p-0 leading-5"
    }
};
