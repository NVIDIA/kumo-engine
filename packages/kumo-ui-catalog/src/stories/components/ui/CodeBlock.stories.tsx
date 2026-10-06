// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeBlock } from "@/components/ui/code-block";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";

const meta: Meta<typeof CodeBlock> = {
    title: "UI/CodeBlock",
    component: CodeBlock,
    tags: ["autodocs"],
    args: {
        codeString: "console.log('Hello, Kumo');",
        language: "javascript"
    }
};

export default meta;

type Story = StoryObj<typeof CodeBlock>;

export const Default: Story = {
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("console")).toBeVisible();
        await expect(canvas.getByText("'Hello, Kumo'")).toBeVisible();
    }
};

export const Json: Story = {
    args: {
        language: "json",
        codeString: JSON.stringify(
            {
                Version: "2012-10-17",
                Statement: [
                    {
                        Sid: "KumoRead",
                        Effect: "Allow",
                        Action: ["s3:GetObject", "s3:ListBucket"],
                        Resource: ["arn:aws:s3:::your-bucket/*"]
                    }
                ]
            },
            null,
            2
        )
    }
};

export const WithLineNumbers: Story = {
    args: {
        codeString: "line1\nline2\nline3",
        showLineNumbers: true
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("1")).toBeVisible();
        await expect(canvas.getByText("2")).toBeVisible();
        await expect(canvas.getByText("3")).toBeVisible();
    }
};

export const PlainText: Story = {
    args: {
        codeString: "Some unformatted text without a known language.",
        language: "text"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const block = canvas.getByRole("code");
        await expect(block).toHaveClass("language-text");
    }
};

export const CustomClassName: Story = {
    args: {
        codeString: "print('hi')",
        language: "python",
        className: "bg-surface-grey rounded-md p-3 max-w-md text-xs"
    }
};
