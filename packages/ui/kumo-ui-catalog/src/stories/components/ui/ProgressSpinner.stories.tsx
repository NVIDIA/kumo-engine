// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ProgressSpinner } from "@/animations/progress-spinner";
import { Meta, StoryObj } from "@storybook/react-vite";

const meta = {
    title: "UI/ProgressSpinner",
    component: ProgressSpinner,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {
        className: {
            control: "text",
            description:
                "Classes for the outer ring (size, outer border color, outer animation)"
        },
        innerClassName: {
            control: "text",
            description:
                "Classes for the inner ring (inset, inner border color, inner animation)"
        }
    }
} satisfies Meta<typeof ProgressSpinner>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Large: Story = {
    args: {
        className: "h-10 w-10 border-2",
        innerClassName: "inset-[6px] border-4"
    }
};

export const CustomColors: Story = {
    args: {
        className:
            "h-8 w-8 border-t-blue-600 border-b-blue-600 animate-[spin_3s_linear_infinite]",
        innerClassName:
            "border-t-emerald-500 border-b-emerald-500 animate-[spin_1s_linear_infinite]"
    }
};

export const SlowAndFast: Story = {
    args: {
        className: "h-12 w-12 animate-[spin_8s_linear_infinite]",
        innerClassName:
            "inset-[5px] animate-[spin_0.8s_linear_infinite] border-t-purple-600 border-b-purple-600"
    }
};

export const Group: Story = {
    render: () => (
        <div className="flex items-center gap-6">
            <ProgressSpinner />
            <ProgressSpinner
                className="h-8 w-8"
                innerClassName="inset-[4px] border-t-amber-500 border-b-amber-500"
            />
            <ProgressSpinner
                className="h-12 w-12 border-2 border-b-sky-600 border-t-sky-600"
                innerClassName="inset-[6px] border-4 border-t-rose-600 border-b-rose-600"
            />
        </div>
    )
};
