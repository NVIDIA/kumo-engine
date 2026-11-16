// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within, waitFor } from "storybook/test";
import { DropdownCard } from "@/components/blocks/card/dropdown-card";

const options = [
    { value: "week", label: "Last 7 days" },
    { value: "month", label: "Last 30 days" },
    { value: "quarter", label: "Last 90 days" }
];

const meta = {
    title: "Components/Blocks/DropdownCard",
    component: DropdownCard,
    tags: ["autodocs"],
    parameters: { layout: "centered" },
    args: {
        title: "Performance Overview",
        placeholder: "Select period",
        options,
        children: (selected) => (
            <p className="text-title-secondary text-sm">
                Showing data for: <strong>{selected?.label}</strong>
            </p>
        )
    }
} satisfies Meta<typeof DropdownCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithChartContent: Story = {
    args: {
        title: "Revenue Trend",
        placeholder: "Select range",
        options,
        children: (selected) => (
            <div className="text-title-secondary bg-surface-grey flex h-32 items-center justify-center rounded text-sm">
                Chart content for "{selected?.label}"
            </div>
        )
    }
};

export const NoTitle: Story = {
    args: {
        title: "",
        placeholder: "Select period",
        options,
        children: (selected) => (
            <p className="text-title-secondary text-sm">
                Selected: <strong>{selected?.label}</strong>
            </p>
        )
    }
};

export const SelectOption: Story = {
    args: {
        title: "Performance Overview",
        placeholder: "Select period",
        options,
        children: (selected) => (
            <p className="text-title-secondary text-sm">
                Showing data for: <strong>{selected?.label}</strong>
            </p>
        )
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await userEvent.click(
            canvas.getByRole("button", { name: /last 7 days/i })
        );
        await waitFor(() => {
            const body = within(document.body);
            expect(body.getByText("Last 30 days")).toBeVisible();
        });

        await userEvent.click(within(document.body).getByText("Last 30 days"));
        await waitFor(() => {
            expect(
                canvas.getByText("Last 30 days", { selector: "strong" })
            ).toBeInTheDocument();
        });
    }
};

export const ManyOptions: Story = {
    args: {
        title: "Feature Distribution",
        placeholder: "Select feature",
        options: [
            { value: "f1", label: "Feature A" },
            { value: "f2", label: "Feature B" },
            { value: "f3", label: "Feature C" },
            { value: "f4", label: "Feature D" },
            { value: "f5", label: "Feature E" }
        ],
        children: (selected) => (
            <p className="text-title-secondary text-sm">
                Showing: <strong>{selected?.label}</strong>
            </p>
        )
    }
};
