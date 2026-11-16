// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Meta, StoryObj } from "@storybook/react-vite";

const meta = {
    title: "UI/Skeleton",
    component: Skeleton,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {
        className: {
            control: "text",
            description:
                "CSS classes to control size and shape (e.g., 'h-4 w-48', 'h-12 w-12 rounded-full')"
        }
    }
} satisfies Meta<typeof Skeleton>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: {
        className: "h-4 w-48"
    }
};

export const Circle: Story = {
    args: {
        className: "h-12 w-12 rounded-full"
    }
};

export const KeyPair: Story = {
    render: () => (
        <div className="space-y-4">
            <div className="flex items-center gap-2">
                <Skeleton className="h-5 w-24" />
                <Skeleton className="h-5 w-36" />
            </div>
            <div className="flex items-center gap-2">
                <Skeleton className="h-5 w-24" />
                <Skeleton className="h-5 w-36" />
            </div>
            <div className="flex items-center gap-2">
                <Skeleton className="h-5 w-24" />
                <Skeleton className="h-5 w-36" />
            </div>
        </div>
    )
};

export const PropertyCard: Story = {
    render: () => (
        <Card className="rounded-xs border-1 flex-row border-slate-200 bg-slate-50 px-0 py-0 shadow-none">
            <div className="inline-flex items-center justify-start gap-2 px-2 py-1.5">
                <Skeleton className="h-4 w-4 rounded-sm" />
                <Skeleton className="h-3 w-12 rounded-none bg-neutral-200" />
            </div>
            <div className="inline-flex items-center justify-start gap-2 border-l border-slate-200 px-2 py-1.5">
                <Skeleton className="h-4 w-4 rounded-sm" />
                <Skeleton className="h-3 w-12 rounded-none bg-neutral-200" />
            </div>
            <div className="inline-flex items-center justify-start gap-2 border-l border-slate-200 px-2 py-1.5">
                <Skeleton className="h-4 w-4 rounded-sm" />
                <Skeleton className="h-3 w-12 rounded-none bg-neutral-200" />
            </div>
        </Card>
    )
};

export const ListView: Story = {
    render: () => (
        <div className="w-60 space-y-1 rounded-sm border border-slate-100 bg-white p-2">
            {Array.from({ length: 5 }).map((_, i) => (
                <div
                    key={`skeleton-${i}`}
                    className="flex flex-col gap-1 px-4 py-2">
                    <Skeleton className="w-45 h-5 rounded-sm" />
                    <Skeleton className="w-38 h-4 rounded-sm" />
                </div>
            ))}
        </div>
    )
};

export const DataTable: Story = {
    render: () => (
        <div className="w-full overflow-hidden rounded-sm border border-slate-200">
            <div className="bg-slate-50 px-4 py-3">
                <div className="flex gap-4">
                    <Skeleton className="h-4 w-24" />
                    <Skeleton className="h-4 w-32" />
                    <Skeleton className="h-4 w-28" />
                    <Skeleton className="h-4 w-20" />
                </div>
            </div>
            <div className="divide-y divide-slate-200">
                {Array(5)
                    .fill(null)
                    .map((_, rowIndex) => (
                        <div
                            key={`skeleton-row-${rowIndex}`}
                            className="px-4 py-3">
                            <div className="flex items-center gap-4">
                                <Skeleton className="h-4 w-40 rounded-sm" />
                                <Skeleton className="h-4 w-40 rounded-sm" />
                                <Skeleton className="h-4 w-40 rounded-sm" />
                                <Skeleton className="h-4 w-40 rounded-sm" />
                            </div>
                        </div>
                    ))}
            </div>
        </div>
    )
};

export const Avatar: Story = {
    render: () => (
        <div className="flex items-center space-x-4">
            <Skeleton className="h-12 w-12 rounded-full" />
            <div className="space-y-2">
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-4 w-24" />
            </div>
        </div>
    )
};

export const TextLines: Story = {
    render: () => (
        <div className="space-y-2">
            <Skeleton className="h-4 w-80" />
            <Skeleton className="h-4 w-64" />
            <Skeleton className="h-4 w-48" />
        </div>
    )
};
