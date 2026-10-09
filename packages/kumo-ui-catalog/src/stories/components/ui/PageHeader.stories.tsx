// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/ui/page-header";
import { CheckCircle } from "@/icons";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";

const meta: Meta<typeof PageHeader> = {
    title: "UI/PageHeader",
    component: PageHeader,
    tags: ["autodocs"],
    parameters: { layout: "padded" }
};

export default meta;

type Story = StoryObj<typeof PageHeader>;

export const Default: Story = {
    args: {
        preTitle: "Training Jobs",
        title: "training-job-2025-05-26",
        titleCopyText: "training-job-2025-05-26",
        subTitles: "Started 2 minutes ago",
        screenshotTestId: "page-header"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Training Jobs")).toBeVisible();
        await expect(canvas.getByText("training-job-2025-05-26")).toBeVisible();
    }
};

export const WithIconAndButton: Story = {
    args: {
        preTitle: "Status",
        icon: <CheckCircle className="size-5 fill-emerald-600" />,
        title: "Job completed",
        disableCopy: true,
        headerButton: <Button variant="outline">Re-run</Button>
    }
};

export const LoadingState: Story = {
    args: {
        preTitle: "Training Jobs",
        title: "anything",
        disableCopy: true,
        loadingStatus: true
    }
};

export const Closable: Story = {
    args: {
        preTitle: "Drawer",
        title: "Item details",
        disableCopy: true,
        onCloseDialog: () => {}
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByTestId("close-icon")).toBeVisible();
    }
};

export const NoPreTitle: Story = {
    args: {
        hidePreTitle: true,
        title: "Compact header",
        disableCopy: true
    }
};
