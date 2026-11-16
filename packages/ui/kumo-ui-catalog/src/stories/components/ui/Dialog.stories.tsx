// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";

import { Button } from "@/components/ui/button";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle
} from "@/components/ui/dialog";
import { useState } from "react";

const meta: Meta<typeof Dialog> = {
    title: "UI/Dialog",
    component: Dialog,
    tags: ["autodocs"]
};

export default meta;
type Story = StoryObj<typeof Dialog>;

// Create a real component
function TestDialog() {
    const [open, setOpen] = useState<boolean>(false);

    return (
        <>
            <Button variant="light-outline" onClick={() => setOpen(true)}>
                Open Dialog
            </Button>

            <Dialog open={open} onOpenChange={setOpen}>
                <DialogContent>
                    <DialogHeader className="mt-2">
                        <DialogTitle>Dialog Title</DialogTitle>
                    </DialogHeader>
                    <DialogDescription>
                        Are you sure you want to perform this operation. This
                        action cannot bee undone.
                    </DialogDescription>

                    <DialogFooter>
                        <Button
                            variant="light-outline"
                            onClick={() => setOpen(false)}>
                            Cancel
                        </Button>
                        <Button onClick={() => setOpen(false)}>Confirm</Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </>
    );
}

function TestDialogWithClose() {
    const [open, setOpen] = useState<boolean>(false);

    return (
        <>
            <Button variant="light-outline" onClick={() => setOpen(true)}>
                Open Dialog With Close
            </Button>

            <Dialog open={open} onOpenChange={setOpen}>
                <DialogContent enableClose>
                    <DialogHeader className="mt-2">
                        <DialogTitle>Dialog With Close Button</DialogTitle>
                    </DialogHeader>
                    <DialogDescription>
                        This dialog has an X close button.
                    </DialogDescription>
                    <DialogFooter>
                        <Button onClick={() => setOpen(false)}>Confirm</Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </>
    );
}

export const WithCloseButton: Story = {
    render: () => <TestDialogWithClose />,
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", { name: /open dialog with close/i })
        );

        await waitFor(() => {
            expect(document.querySelector('[role="dialog"]')).not.toBeNull();
        });

        await waitFor(() => {
            const dialog = within(
                document.querySelector('[role="dialog"]') as HTMLElement
            );
            expect(dialog.getByText("Dialog With Close Button")).toBeVisible();
        });

        const dialogEl = document.querySelector(
            '[data-slot="dialog-content"]'
        ) as HTMLElement;
        const closeBtn = dialogEl.querySelector("button") as HTMLElement;
        await userEvent.click(closeBtn);

        await waitFor(() => {
            expect(document.querySelector('[role="dialog"]')).toBeNull();
        });
    }
};

export const Default: Story = {
    render: () => <TestDialog />,
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const openButton = canvas.getByRole("button", { name: /open dialog/i });
        await userEvent.click(openButton);
        await waitFor(() => {
            const dialog = document.querySelector('[role="dialog"]');
            expect(dialog).not.toBeNull();
            const dialogScope = within(dialog as HTMLElement);
            expect(dialogScope.getByText("Dialog Title")).toBeVisible();
        });

        const dialog = within(
            document.querySelector('[role="dialog"]') as HTMLElement
        );
        const cancelButton = dialog.getByRole("button", { name: /cancel/i });
        await userEvent.click(cancelButton);
    }
};
