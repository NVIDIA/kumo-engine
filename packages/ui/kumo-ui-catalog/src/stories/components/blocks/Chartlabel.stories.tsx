// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChartLabel } from "@/components/blocks/chart/chart-label";
import { Meta, StoryObj } from "@storybook/react-vite";

const meta: Meta<typeof ChartLabel> = {
    title: "Blocks/ChartLabel",
    component: ChartLabel,
    tags: ["autodocs"]
};

export default meta;
type Story = StoryObj<typeof ChartLabel>;

export const Basic: Story = {
    render: function DefaultFunc() {
        return (
            <>
                <ChartLabel
                    iconClassName={"bg-red-600"}
                    labelClassName="text-slate-400"
                    text={"Label Text 1"}
                />
                <ChartLabel
                    iconClassName={"bg-teal-700"}
                    labelClassName="text-slate-400"
                    text={"Label Text 2"}
                />
            </>
        );
    }
};

// Exercises default prop fallbacks — all props omitted (iconClassName="", labelClassName="", text="")
export const Defaults: Story = {
    render: function DefaultFunc() {
        return <ChartLabel />;
    }
};

// Exercises text prop with empty string explicitly
export const EmptyText: Story = {
    args: {
        iconClassName: "bg-blue-500",
        labelClassName: "text-blue-500",
        text: ""
    }
};
