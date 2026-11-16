// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AlertCtaList } from "@/components/ui/alert-cta-list";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";

const meta: Meta<typeof AlertCtaList> = {
    title: "UI/AlertCtaList",
    component: AlertCtaList,
    tags: ["autodocs"],
    parameters: { layout: "padded" }
};

export default meta;

type Story = StoryObj<typeof AlertCtaList>;

const sampleWarnings = [
    {
        title: "Missing primary key",
        description: "Table 'orders' has no declared primary key.",
        cta: { name: "Open table", url: "/tables/orders" }
    },
    {
        title: "Stale data",
        description: "Last refresh was 7 days ago.",
        cta: { name: "Refresh", url: "/graphs/refresh" }
    }
];

const fiveErrors = Array.from({ length: 5 }, (_, i) => ({
    title: `Issue ${i + 1}`,
    description: `Description for issue number ${i + 1}.`,
    cta: { name: "Fix", url: `/fix/${i + 1}` }
}));

export const SingleWarning: Story = {
    args: {
        data: [sampleWarnings[0]],
        level: "warning"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Warning")).toBeVisible();
        await expect(canvas.getByText("Missing primary key:")).toBeVisible();
    }
};

export const MultipleErrors: Story = {
    args: {
        data: sampleWarnings,
        level: "error"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Errors")).toBeVisible();
    }
};

export const InfoLevel: Story = {
    args: {
        data: [{ description: "Heads up — your trial ends Friday." }],
        level: "info",
        title: "Reminder"
    }
};

export const ShowMoreToggle: Story = {
    args: {
        data: fiveErrors,
        level: "error",
        previewCount: 3
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Show More")).toBeVisible();
        await userEvent.click(canvas.getByText("Show More"));
        await expect(canvas.getByText("Show Less")).toBeVisible();
    }
};

export const DismissibleWithCtaClick: Story = {
    args: {
        data: sampleWarnings,
        level: "warning",
        onClose: () => {},
        onCtaClick: () => {}
    }
};

export const FilteredCta: Story = {
    args: {
        data: sampleWarnings,
        level: "warning",
        // Pretend caller is currently on /tables/orders — hide that CTA
        shouldShowCta: (url) => url !== "/tables/orders"
    }
};

export const HiddenTitle: Story = {
    args: {
        data: sampleWarnings,
        level: "error",
        hideTitle: true
    }
};
