// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

import {
    createFileUploadItems,
    Upload,
    UploadContent,
    UploadDescription,
    UploadInputElement,
    UploadRoot,
    UploadTrigger,
    type FileUploadItem
} from "@/components/ui/upload";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, fn, userEvent, within } from "storybook/test";

const meta = {
    title: "UI/Upload",
    component: Upload,
    parameters: {
        layout: "padded"
    },
    tags: ["autodocs"],
    args: {
        onValueChange: fn(),
        onFileRemove: fn()
    },
    argTypes: {
        accept: {
            control: { type: "text" },
            description:
                "Comma-separated list of MIME types or file extensions the picker should allow"
        },
        multiple: {
            control: "boolean",
            description: "Whether more than one file can be selected"
        },
        disabled: {
            control: "boolean",
            description:
                "Disables file selection, drag-and-drop, and item actions"
        },
        kind: {
            control: { type: "radio" },
            options: ["flat", "floating"],
            description: "Bordered shell (flat) or borderless (floating)"
        },
        listKind: {
            control: { type: "radio" },
            options: ["card", "media"],
            description:
                "Detailed rows for documents (card) or thumbnail tiles for images (media)"
        },
        status: {
            control: { type: "radio" },
            options: [undefined, "error", "success"],
            description: "Validation state applied to the trigger"
        }
    }
} satisfies Meta<typeof Upload>;

export default meta;
type Story = StoryObj<typeof meta>;

const csvFile = (name = "transactions.csv", size = 2_400_000) => {
    const file = new File(["id,amount\n1,10\n"], name, { type: "text/csv" });
    Object.defineProperty(file, "size", { value: size });
    return file;
};

const asItem = (
    file: File,
    overrides: Partial<FileUploadItem> = {}
): FileUploadItem => ({
    ...createFileUploadItems([file])[0],
    ...overrides
});

export const Default: Story = {
    args: {
        accept: ".csv,.parquet",
        children: "Only '.CSV' or '.Parquet' files are allowed, up to 1 GB"
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText(/choose file/i)).toBeVisible();

        const input =
            canvasElement.querySelector<HTMLInputElement>('input[type="file"]');
        await expect(input).not.toBeNull();
        await userEvent.upload(input as HTMLInputElement, csvFile());

        await expect(args.onValueChange).toHaveBeenCalled();
        await expect(canvas.getByText("transactions.csv")).toBeVisible();
    }
};

export const Multiple: Story = {
    args: {
        multiple: true,
        accept: ".csv,.parquet",
        children: "Select as many files as you need"
    }
};

/** Compact tiles, for images and videos rather than documents. */
export const MediaList: Story = {
    args: {
        multiple: true,
        listKind: "media",
        accept: "image/*",
        value: [
            asItem(csvFile("cover.png", 180_000), { status: "success" }),
            asItem(csvFile("hero.png", 240_000), { status: "uploading" })
        ],
        children: "PNG or JPG, up to 10 MB each"
    },
    play: async ({ canvasElement }) => {
        const tiles = canvasElement.querySelectorAll(
            '[data-slot="upload-item"]'
        );
        await expect(tiles).toHaveLength(2);
        await expect(tiles[0]).toHaveClass(/h-25/);
    }
};

export const Floating: Story = {
    args: {
        kind: "floating",
        children: "Borderless variant, for use inside an existing card"
    }
};

export const Disabled: Story = {
    args: {
        disabled: true,
        children: "Uploading is unavailable while a file is already selected"
    },
    play: async ({ canvasElement }) => {
        const input =
            canvasElement.querySelector<HTMLInputElement>('input[type="file"]');
        await expect(input).toBeDisabled();
    }
};

export const Uploading: Story = {
    args: {
        value: asItem(csvFile(), {
            status: "uploading",
            uploadedBytes: 900_000
        })
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("900 KB of 2.4 MB")).toBeVisible();
    }
};

export const Success: Story = {
    args: {
        status: "success",
        value: asItem(csvFile(), { status: "success" })
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("2.4 MB")).toBeVisible();
    }
};

export const WithError: Story = {
    args: {
        status: "error",
        onFileRetry: fn(),
        value: asItem(csvFile("too-big.csv", 4_000_000_000), {
            status: "error",
            errorMessage:
                "File size is too large. Kumo supports a maximum of 1 GB"
        })
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText(/file size is too large/i)).toBeVisible();

        await userEvent.click(canvas.getByRole("button", { name: /retry/i }));
        await expect(args.onFileRetry).toHaveBeenCalledTimes(1);
    }
};

/**
 * The compound form, for when the trigger and the file list need to sit in
 * different places or carry extra chrome.
 */
export const Composed: StoryObj = {
    render: () => {
        // eslint-disable-next-line react-hooks/rules-of-hooks
        const [files, setFiles] = useState<FileUploadItem[]>([]);
        return (
            <UploadRoot multiple value={files} onValueChange={setFiles}>
                <UploadTrigger slotAnchor="Browse your machine">
                    <UploadInputElement accept=".csv,.parquet" />
                    <UploadDescription>
                        CSV or Parquet, up to 1 GB each
                    </UploadDescription>
                </UploadTrigger>
                <UploadContent />
            </UploadRoot>
        );
    }
};

/**
 * A controlled single-file upload. `value` is `null` rather than `undefined`
 * while nothing is selected, which keeps the upload controlled: `undefined`
 * means "manage your own state", so clearing the selection with it would hand
 * control back and resurface the file that was just removed.
 */
export const ControlledSingle: StoryObj = {
    render: () => {
        // eslint-disable-next-line react-hooks/rules-of-hooks
        const [file, setFile] = useState<FileUploadItem | null>(null);
        return (
            <Upload
                accept=".json"
                value={file}
                onValueChange={(next) => setFile(next ?? null)}
            />
        );
    },
    play: async ({ canvasElement, step }) => {
        const canvas = within(canvasElement);
        const input =
            canvasElement.querySelector<HTMLInputElement>('input[type="file"]');
        if (!input) throw new Error("file input not found");

        await step("selecting a file shows it", async () => {
            await userEvent.upload(
                input,
                new File(["{}"], "key.json", { type: "application/json" })
            );
            await expect(await canvas.findByText("key.json")).toBeVisible();
        });

        await step("one click removes it", async () => {
            await userEvent.click(
                canvas.getByRole("button", { name: /Remove key.json/ })
            );
            await expect(
                canvas.queryByText("key.json")
            ).not.toBeInTheDocument();
        });
    }
};

/**
 * `kind`, `slotAnchor` and `slotHeaderText` all have defaults, so a minimal
 * upload sets none of them.
 */
export const Minimal: StoryObj = {
    render: () => <Upload accept=".json" />
};
