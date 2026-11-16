// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { CopyToClipBoard } from "@/components/ui/copy-to-clipboard";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";

const meta = {
    title: "UI/CopyToClipboard",
    component: CopyToClipBoard,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {
        copiedText: {
            control: "text",
            description: "The text to copy to clipboard"
        },
        tooltipText: {
            control: "text",
            description: "Text shown in tooltip after copying"
        },
        size: {
            control: "select",
            options: ["xs", "sm", "default", "lg"],
            description: "Button size"
        },
        variant: {
            control: "select",
            options: ["default", "secondary", "outline", "link", "underline"],
            description: "Button variant"
        },
        iconSize: {
            control: "number",
            description: "Size of the copy icon in pixels"
        },
        title: {
            control: "text",
            description: "Optional title text next to the icon"
        }
    }
} satisfies Meta<typeof CopyToClipBoard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: {
        copiedText: "Text to copy",
        iconSize: 16
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const button = canvas.getByRole("button", {
            name: /copy to clipboard/i
        });
        await expect(button).toBeVisible();
        await userEvent.click(button);
        await expect(button).toBeVisible();
    }
};

export const WithTitle: Story = {
    args: {
        copiedText: "s3://example-bucket/transactions/",
        tooltipText: ""
    },
    render: function Render(args) {
        const [copyState, setCopyState] = useState(0);
        return (
            <div
                style={{
                    display: "flex",
                    borderRadius: 4,
                    fontFamily: "Inter",
                    fontSize: 12,
                    fontStyle: "normal",
                    fontWeight: 500,
                    lineHeight: "16px",
                    border: "1px solid #E7EBEF",
                    padding: 8,
                    color: "#6E84A3",
                    textAlign: "center",
                    width: "150px"
                }}
                onClick={() => {
                    setCopyState(1);
                    setTimeout(() => setCopyState(0), 2000);
                }}>
                <CopyToClipBoard
                    {...args}
                    title={copyState === 0 ? "Copy S3 path" : "Copied"}
                />
            </div>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Copy S3 path")).toBeVisible();
        await userEvent.click(
            canvas.getByRole("button", { name: /copy s3 path/i })
        );
        await waitFor(() => {
            expect(canvas.getByText("Copied")).toBeVisible();
        });
    }
};

export const WithTooltip: Story = {
    args: {
        copiedText: "SELECT * FROM users WHERE active = true",
        tooltipText: "Query copied!"
    }
};

export const LargeSize: Story = {
    args: {
        copiedText: "Error content to copy",
        size: "lg"
    }
};

export const InlineWithValue: Story = {
    args: {
        copiedText: "s3://bucket/path/to/data",
        iconSize: 16
    },
    render: (args) => (
        <div className="flex items-center gap-1">
            <span className="text-muted-foreground text-sm">
                s3://bucket/path/to/data
            </span>
            <CopyToClipBoard {...args} />
        </div>
    )
};

export const PageHeaderStyle: Story = {
    args: {
        copiedText: "project-abc-123",
        iconSize: 16
    },
    render: (args) => (
        <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold">Project Name</h2>
            <CopyToClipBoard {...args} />
        </div>
    )
};

export const CodeEditorStyle: Story = {
    args: {
        copiedText: "PREDICT MAX(transaction_table.Quantity)",
        tooltipText: "Query copied!",
        size: "sm",
        variant: "outline",
        iconSize: 16
    },
    render: (args) => (
        <div className="bg-muted relative w-[400px] rounded-md border p-4">
            <pre className="pr-10 font-mono text-sm">
                PREDICT MAX(transaction_table.Quantity)
            </pre>
            <div
                style={{
                    position: "absolute",
                    top: "12px",
                    right: "8px"
                }}>
                <CopyToClipBoard {...args} />
            </div>
        </div>
    )
};
