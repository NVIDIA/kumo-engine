// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Toaster } from "@/components/ui/sonner";
import { Button } from "@/components/ui/button";
import { Meta, StoryObj } from "@storybook/react-vite";
import { toast } from "sonner";
import { expect, userEvent, waitFor, within } from "storybook/test";

const meta: Meta = {
    title: "UI/Sonner",
    component: Toaster,
    parameters: {
        layout: "centered",
        docs: {
            story: {
                inline: false,
                iframeHeight: 200
            }
        }
    },
    tags: ["autodocs"],
    decorators: [
        (Story) => (
            <div>
                <Story />
                <Toaster />
            </div>
        )
    ]
};

export default meta;
type Story = StoryObj;

export const Default: Story = {
    render: () => (
        <Button
            variant="light-outline"
            onClick={() => toast("This is a toast message")}>
            Show Toast
        </Button>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(canvas.getByText("Show Toast"));

        await waitFor(() => {
            const toastEl = document.querySelector("[data-sonner-toast]");
            expect(toastEl).toBeTruthy();
            expect(toastEl).toHaveTextContent("This is a toast message");
        });
    }
};

export const AllVariants: Story = {
    render: () => (
        <div className="flex flex-wrap gap-2">
            <Button
                variant="light-outline"
                onClick={() => toast("Default toast message")}>
                Default
            </Button>
            <Button
                variant="light-outline"
                onClick={() => toast.success("Success message")}>
                Success
            </Button>
            <Button
                variant="light-outline"
                onClick={() => toast.error("Error message")}>
                Error
            </Button>
            <Button
                variant="light-outline"
                onClick={() => toast.warning("Warning message")}>
                Warning
            </Button>
            <Button
                variant="light-outline"
                onClick={() => toast.info("Info message")}>
                Info
            </Button>
        </div>
    )
};

export const FormSubmitSuccess: Story = {
    render: () => (
        <Button
            onClick={() =>
                toast.success("Settings saved", {
                    description:
                        "Your preferences have been updated successfully."
                })
            }>
            Save Settings
        </Button>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(canvas.getByText("Save Settings"));
        await waitFor(() => {
            const toastEl = document.querySelector("[data-sonner-toast]");
            expect(toastEl).toBeTruthy();
            expect(toastEl).toHaveTextContent("Settings saved");
            expect(toastEl).toHaveTextContent(
                "Your preferences have been updated successfully."
            );
        });
    }
};

export const CopyToClipboard: Story = {
    render: () => (
        <Button
            variant="light-outline"
            onClick={() => {
                navigator.clipboard.writeText("s3://example-bucket/data/");
                toast.success("Copied to clipboard");
            }}>
            Copy S3 Path
        </Button>
    )
};

export const DeleteWithUndo: Story = {
    render: () => (
        <Button
            variant="light-outline"
            onClick={() =>
                toast("Model deleted", {
                    description: "my-prediction-model was removed.",
                    action: {
                        label: "Undo",
                        onClick: () => toast.success("Model restored")
                    }
                })
            }>
            Delete Model
        </Button>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(canvas.getByText("Delete Model"));

        await waitFor(() => {
            const toastEl = document.querySelector("[data-sonner-toast]");
            expect(toastEl).toBeTruthy();
            expect(toastEl).toHaveTextContent("Model deleted");
            expect(toastEl).toHaveTextContent(
                "my-prediction-model was removed."
            );
        });

        const undoBtn = document.querySelector("[data-sonner-toast] button")!;
        await userEvent.click(undoBtn);

        await waitFor(() => {
            const toasts = document.querySelectorAll("[data-sonner-toast]");
            const restoredToast = Array.from(toasts).find((t) =>
                t.textContent?.includes("Model restored")
            );
            expect(restoredToast).toBeTruthy();
        });
    }
};
