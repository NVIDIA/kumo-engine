// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Symbol } from "@/components/ui/symbol";
import { Meta, StoryObj } from "@storybook/react-vite";

const meta = {
    title: "UI/Symbol",
    component: Symbol,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {
        type: {
            control: "select",
            options: ["emdash", "dot"]
        }
    }
} satisfies Meta<typeof Symbol>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Emdash: Story = {
    args: {
        type: "emdash"
    }
};

export const Dot: Story = {
    args: {
        type: "dot"
    }
};

export const AsEmptyStatePlaceholder: Story = {
    args: {
        type: "emdash"
    },
    render: () => (
        <div className="space-y-3">
            <div className="flex w-64 items-center justify-between">
                <span className="text-sm text-slate-600">Training Rows</span>
                <span className="text-base font-medium">1,234</span>
            </div>
            <div className="flex w-64 items-center justify-between">
                <span className="text-sm text-slate-600">Date Range</span>
                <Symbol type="emdash" />
            </div>
            <div className="flex w-64 items-center justify-between">
                <span className="text-sm text-slate-600">Accuracy</span>
                <Symbol type="emdash" />
            </div>
        </div>
    )
};

export const WithIconPlaceholder: Story = {
    args: {
        type: "emdash"
    },
    render: () => (
        <div className="space-y-2">
            <div className="flex items-center space-x-4">
                <svg
                    className="size-4.5 text-slate-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor">
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7c-2 0-3 1-3 3z"
                    />
                </svg>
                <span>1,500 rows</span>
            </div>
            <div className="flex items-center space-x-4">
                <svg
                    className="size-4.5 text-slate-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor">
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                </svg>
                <Symbol type="emdash" />
            </div>
        </div>
    )
};
