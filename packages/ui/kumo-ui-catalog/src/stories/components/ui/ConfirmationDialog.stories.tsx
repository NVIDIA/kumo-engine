// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from "@/components/ui/button";
import {
    ConfirmationDialog,
    DialogRef
} from "@/components/ui/confirmation-dialog";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { useRef } from "react";
import { expect, fn, screen, userEvent, waitFor } from "storybook/test";

const meta: Meta<typeof ConfirmationDialog> = {
    title: "UI/ConfirmationDialog",
    component: ConfirmationDialog,
    tags: ["autodocs"]
};

export default meta;

// Imperative dialog: rendered with a trigger button that calls toggleDialog
// via ref. The trigger button is part of the story scaffolding, not the
// component under test.
const ConfirmationHarness: React.FC<{
    title?: string;
    description?: string;
    enableSkip?: boolean;
    onConfirm: (value?: object) => void;
    onClose: () => void;
    onSkip?: (value?: object) => void;
}> = ({
    title = "Delete project",
    description = "This action cannot be undone.",
    enableSkip = false,
    onConfirm,
    onClose,
    onSkip
}) => {
    const ref = useRef<DialogRef<object>>(null);
    return (
        <>
            <Button
                id="open-dialog"
                onClick={() =>
                    ref.current?.toggleDialog(
                        title,
                        description,
                        { id: "demo-1" },
                        enableSkip
                    )
                }>
                Open dialog
            </Button>
            <ConfirmationDialog
                ref={ref}
                handleClose={() => {
                    onClose();
                    ref.current?.toggleDialog();
                }}
                handleConfirm={(value) => {
                    onConfirm(value);
                    ref.current?.toggleDialog();
                }}
                handleSkip={
                    onSkip
                        ? (value) => {
                              onSkip(value);
                              ref.current?.toggleDialog();
                          }
                        : undefined
                }
            />
        </>
    );
};

type HarnessArgs = {
    onConfirm: ReturnType<typeof fn>;
    onClose: ReturnType<typeof fn>;
    onSkip?: ReturnType<typeof fn>;
};

export const Default: StoryObj<HarnessArgs> = {
    args: { onConfirm: fn(), onClose: fn() },
    render: (args) => (
        <ConfirmationHarness
            onConfirm={args.onConfirm}
            onClose={args.onClose}
        />
    ),
    play: async ({ canvasElement }) => {
        const trigger = canvasElement.querySelector(
            "#open-dialog"
        ) as HTMLButtonElement;
        await userEvent.click(trigger);

        const title = await screen.findByText("Delete project");
        await expect(title).toBeVisible();
        await expect(
            screen.getByText("This action cannot be undone.")
        ).toBeVisible();
    }
};

export const ConfirmAction: StoryObj<HarnessArgs> = {
    args: { onConfirm: fn(), onClose: fn() },
    render: (args) => (
        <ConfirmationHarness
            onConfirm={args.onConfirm}
            onClose={args.onClose}
        />
    ),
    play: async ({ canvasElement, args }) => {
        const trigger = canvasElement.querySelector(
            "#open-dialog"
        ) as HTMLButtonElement;
        await userEvent.click(trigger);

        const confirmBtn = await screen.findByRole("button", { name: /ok/i });
        await userEvent.click(confirmBtn);
        await waitFor(() => expect(args.onConfirm).toHaveBeenCalled());
    }
};

export const CancelAction: StoryObj<HarnessArgs> = {
    args: { onConfirm: fn(), onClose: fn() },
    render: (args) => (
        <ConfirmationHarness
            onConfirm={args.onConfirm}
            onClose={args.onClose}
        />
    ),
    play: async ({ canvasElement, args }) => {
        const trigger = canvasElement.querySelector(
            "#open-dialog"
        ) as HTMLButtonElement;
        await userEvent.click(trigger);

        const cancelBtn = await screen.findByRole("button", {
            name: /cancel/i
        });
        await userEvent.click(cancelBtn);
        await waitFor(() => expect(args.onClose).toHaveBeenCalled());
    }
};

export const WithSkipButton: StoryObj<HarnessArgs> = {
    args: { onConfirm: fn(), onClose: fn(), onSkip: fn() },
    render: (args) => (
        <ConfirmationHarness
            enableSkip
            onConfirm={args.onConfirm}
            onClose={args.onClose}
            onSkip={args.onSkip}
        />
    ),
    play: async ({ canvasElement }) => {
        const trigger = canvasElement.querySelector(
            "#open-dialog"
        ) as HTMLButtonElement;
        await userEvent.click(trigger);
        // With skip enabled the cancel button becomes a skip button
        await expect(
            await screen.findByRole("button", { name: /cancel/i })
        ).toBeVisible();
        await expect(screen.getByRole("button", { name: /ok/i })).toBeVisible();
    }
};
