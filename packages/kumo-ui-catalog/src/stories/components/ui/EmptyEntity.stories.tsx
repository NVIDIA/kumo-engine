// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EmptyEntity } from "@/components/ui/emptyentity";
import { Power, NetworkNode, DeveloperBoard } from "@/icons";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, fn, userEvent, within } from "storybook/test";

const meta = {
    title: "UI/EmptyEntity",
    component: EmptyEntity,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {
        data: {
            description: "Configuration object for the empty entity state"
        }
    }
} satisfies Meta<typeof EmptyEntity>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Connectors: Story = {
    args: {
        data: {
            icon: <Power className="size-12" />,
            label: <label>Add a data source to begin!</label>,
            renderButtonId: "create-connector",
            renderButtonText: "Connect",
            onRenderClick: fn(),
            docsId: "Data_source",
            docsText: "Quickstart Documentation",
            onDocsClick: fn()
        }
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);

        const renderButton = canvas.getByRole("button", { name: /Connect/i });
        await expect(renderButton).toBeVisible();
        await userEvent.click(renderButton);
        await expect(args.data.onRenderClick).toHaveBeenCalledTimes(1);

        const docsButton = canvas.getByRole("button", {
            name: /Quickstart Documentation/i
        });
        await expect(docsButton).toBeVisible();
        await userEvent.click(docsButton);
        await expect(args.data.onDocsClick).toHaveBeenCalledTimes(1);
    }
};

export const Tables: Story = {
    args: {
        data: {
            icon: <NetworkNode className="size-12" />,
            label: <label>Connect a table to create a Kumo Graph</label>,
            renderButtonId: "create-table",
            renderButtonText: "Add table",
            onRenderClick: fn(),
            docsId: "Tables_doc",
            docsText: "Quickstart Documentation",
            onDocsClick: fn()
        }
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);

        const renderButton = canvas.getByRole("button", { name: /Add table/i });
        await expect(renderButton).toBeVisible();
        await userEvent.click(renderButton);
        await expect(args.data.onRenderClick).toHaveBeenCalledTimes(1);

        const docsButton = canvas.getByRole("button", {
            name: /Quickstart Documentation/i
        });
        await expect(docsButton).toBeVisible();
        await userEvent.click(docsButton);
        await expect(args.data.onDocsClick).toHaveBeenCalledTimes(1);
    }
};

export const Graphs: Story = {
    args: {
        data: {
            icon: <DeveloperBoard className="size-12" />,
            label: <label>Connect a graph to create a Training Job</label>,
            renderButtonId: "create-graph",
            renderButtonText: "Add Graph",
            onRenderClick: fn(),
            docsId: "Graph_doc",
            docsText: "Quickstart Documentation",
            onDocsClick: fn()
        }
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);

        const renderButton = canvas.getByRole("button", {
            name: /Add Graph/i
        });
        await expect(renderButton).toBeVisible();
        await userEvent.click(renderButton);
        await expect(args.data.onRenderClick).toHaveBeenCalledTimes(1);

        const docsButton = canvas.getByRole("button", {
            name: /Quickstart Documentation/i
        });
        await expect(docsButton).toBeVisible();
        await userEvent.click(docsButton);
        await expect(args.data.onDocsClick).toHaveBeenCalledTimes(1);
    }
};

export const Training: Story = {
    args: {
        data: {
            icon: <NetworkNode className="size-12" />,
            label: (
                <label className="text-center">
                    No trained model yet.
                    <br /> Let's start by training your first model!!
                </label>
            ),
            renderButtonId: "create-training",
            renderButtonText: "Train a model",
            onRenderClick: fn(),
            docsId: "Training_doc",
            docsText: "Quickstart Documentation",
            onDocsClick: fn()
        }
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);

        const renderButton = canvas.getByRole("button", {
            name: /Train a model/i
        });
        await expect(renderButton).toBeVisible();
        await userEvent.click(renderButton);
        await expect(args.data.onRenderClick).toHaveBeenCalledTimes(1);

        const docsButton = canvas.getByRole("button", {
            name: /Quickstart Documentation/i
        });
        await expect(docsButton).toBeVisible();
        await userEvent.click(docsButton);
        await expect(args.data.onDocsClick).toHaveBeenCalledTimes(1);
    }
};

export const Predictions: Story = {
    args: {
        data: {
            icon: <NetworkNode className="size-12" />,
            label: (
                <label className="text-center">
                    No predictions yet.
                    <br /> Let's start by making your first prediction!!
                </label>
            ),
            renderButtonId: "create-prediction",
            renderButtonText: "Make predictions",
            onRenderClick: fn(),
            docsId: "Predictions_doc",
            docsText: "Quickstart Documentation",
            onDocsClick: fn()
        }
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);

        const renderButton = canvas.getByRole("button", {
            name: /Make predictions/i
        });
        await expect(renderButton).toBeVisible();
        await userEvent.click(renderButton);
        await expect(args.data.onRenderClick).toHaveBeenCalledTimes(1);

        const docsButton = canvas.getByRole("button", {
            name: /Quickstart Documentation/i
        });
        await expect(docsButton).toBeVisible();
        await userEvent.click(docsButton);
        await expect(args.data.onDocsClick).toHaveBeenCalledTimes(1);
    }
};
