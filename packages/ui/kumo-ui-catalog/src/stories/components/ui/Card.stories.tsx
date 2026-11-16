// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Card,
    CardHeader,
    CardTitle,
    CardDescription,
    CardContent,
    CardFooter
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Meta, StoryObj } from "@storybook/react-vite";
import { DeveloperBoard, CalendarBasic, AvgTime } from "@/icons";

const meta = {
    title: "UI/Card",
    component: Card,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {
        className: {
            control: { type: "text" },
            description: "Additional CSS classes"
        }
    }
} satisfies Meta<typeof Card>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    render: () => (
        <Card className="w-[350px]">
            <CardHeader>
                <CardTitle>Card Title</CardTitle>
                <CardDescription>Card description goes here.</CardDescription>
            </CardHeader>
            <CardContent>
                <p>Card content goes here.</p>
            </CardContent>
            <CardFooter>
                <Button>Action</Button>
            </CardFooter>
        </Card>
    )
};

export const WithHeaderSeparator: Story = {
    render: () => (
        <Card className="w-[350px]">
            <CardHeader bottomSeparator>
                <CardTitle>Card Title</CardTitle>
                <CardDescription>
                    This card has a bottom separator on the header.
                </CardDescription>
            </CardHeader>
            <CardContent>
                <p>Card content goes here.</p>
            </CardContent>
            <CardFooter>
                <Button>Action</Button>
            </CardFooter>
        </Card>
    )
};

export const MetricCardStyle: Story = {
    render: () => (
        <Card className="border-1 flex w-max flex-row flex-nowrap items-center gap-0 rounded border-slate-200 bg-white px-1 py-0 shadow-none">
            <div className="inline-flex shrink-0 items-center justify-start gap-1.5 px-2 py-1.5">
                <div className="text-title-secondary justify-start pl-1 text-base font-medium leading-4">
                    Accuracy
                </div>
                <div className="text-title-primary justify-start pr-1 text-base font-medium leading-5">
                    0.95
                </div>
            </div>
            <div className="inline-flex shrink-0 items-center justify-start gap-1.5 border-l border-slate-200 px-2 py-1.5">
                <div className="text-title-secondary justify-start pl-1 text-base font-medium leading-4">
                    Precision
                </div>
                <div className="text-title-primary justify-start pr-1 text-base font-medium leading-5">
                    0.89
                </div>
            </div>
            <div className="inline-flex shrink-0 items-center justify-start gap-1.5 border-l border-slate-200 px-2 py-1.5">
                <div className="text-title-secondary justify-start pl-1 text-base font-medium leading-4">
                    Recall
                </div>
                <div className="text-title-primary justify-start pr-1 text-base font-medium leading-5">
                    0.92
                </div>
            </div>
        </Card>
    )
};

export const PropertyCardStyle: Story = {
    render: () => (
        <Card className="rounded-xs border-1 flex-row border-slate-200 bg-slate-50 px-0 py-0 shadow-none">
            <div className="inline-flex items-center justify-start gap-2 px-2 py-1.5">
                <CalendarBasic className="size-4 text-slate-500" />
                <div className="justify-start text-sm font-medium leading-tight text-slate-500">
                    Jan 15, 2025
                </div>
            </div>
            <div className="inline-flex items-center justify-start gap-2 border-l border-slate-200 px-2 py-1.5">
                <AvgTime className="size-4 text-slate-500" />
                <div className="justify-start text-sm font-medium leading-tight text-slate-500">
                    2h 30m
                </div>
            </div>
        </Card>
    )
};

export const StatusCardQueued: Story = {
    render: () => (
        <Card className="inline-flex w-full max-w-md flex-col items-start justify-start gap-4 self-stretch rounded border-none bg-yellow-100 px-6 py-4 text-yellow-700 shadow-none">
            <CardHeader className="flex-inline px-0">
                <div className="flex w-full items-start justify-between">
                    <div className="flex items-start gap-4">
                        <AvgTime className="size-5" />
                        <span className="justify-start pt-0.5 text-sm font-medium leading-tight">
                            Job queued and waiting to start
                        </span>
                    </div>
                </div>
            </CardHeader>
        </Card>
    )
};

export const StatusCardRunning: Story = {
    render: () => (
        <Card className="inline-flex w-full max-w-md flex-col items-start justify-start gap-4 self-stretch rounded border-none bg-violet-100 px-6 py-4 text-indigo-600 shadow-none">
            <CardHeader className="flex-inline px-0">
                <div className="flex w-full items-start justify-between">
                    <div className="flex items-start gap-4">
                        <DeveloperBoard className="size-5" />
                        <span className="justify-start pt-0.5 text-sm font-medium leading-tight">
                            Model training in progress...
                        </span>
                    </div>
                </div>
            </CardHeader>
        </Card>
    )
};

export const StatusCardDone: Story = {
    render: () => (
        <Card className="inline-flex w-full max-w-md flex-col items-start justify-start gap-4 self-stretch rounded border-none bg-green-100 px-6 py-4 text-green-700 shadow-none">
            <CardHeader className="flex-inline px-0">
                <div className="flex w-full items-start justify-between">
                    <div className="flex items-start gap-4">
                        <DeveloperBoard className="size-5" />
                        <span className="justify-start pt-0.5 text-sm font-medium leading-tight">
                            Training completed successfully
                        </span>
                    </div>
                </div>
            </CardHeader>
        </Card>
    )
};

export const StatusCardFailed: Story = {
    render: () => (
        <Card className="inline-flex w-full max-w-md flex-col items-start justify-start gap-4 self-stretch rounded border-none bg-red-100 px-6 py-4 text-red-700 shadow-none">
            <CardHeader className="flex-inline px-0">
                <div className="flex w-full items-start justify-between">
                    <div className="flex items-start gap-4">
                        <DeveloperBoard className="size-5" />
                        <span className="justify-start pt-0.5 text-sm font-medium leading-tight">
                            Job failed: Out of memory error
                        </span>
                    </div>
                </div>
            </CardHeader>
        </Card>
    )
};

export const QueryCardStyle: Story = {
    render: () => (
        <div className="w-[400px]">
            <Card className="self-stretch overflow-hidden rounded-bl-none rounded-br-none rounded-tl rounded-tr py-2 shadow-none">
                <CardHeader data-slot="query-card-with-icon-header">
                    <div className="flex items-center">
                        <DeveloperBoard className="size-4 text-slate-500" />
                        <span className="mx-2 text-sm font-medium leading-tight text-slate-500">
                            Predictive Query
                        </span>
                    </div>
                </CardHeader>
                <CardContent className="relative whitespace-pre-line text-sm font-semibold leading-tight text-gray-800 shadow-none">
                    PREDICT COUNT(Transaction.amount, 0, 30, days) FOR EACH
                    User.user_id
                </CardContent>
            </Card>
            <CardFooter className="rounded-xs border-1 flex-row border-slate-200 bg-white px-0 py-0 shadow-none">
                <div className="inline-flex items-center justify-start gap-2 px-2 py-1.5">
                    <CalendarBasic className="size-4 text-slate-500" />
                    <div className="justify-start text-sm font-medium leading-tight text-slate-500">
                        Jan 15, 2025
                    </div>
                </div>
                <div className="inline-flex items-center justify-start gap-2 border-l border-slate-200 px-2 py-1.5">
                    <AvgTime className="size-4 text-slate-500" />
                    <div className="justify-start text-sm font-medium leading-tight text-slate-500">
                        45m
                    </div>
                </div>
            </CardFooter>
        </div>
    )
};

export const WithSkeleton: Story = {
    render: () => (
        <Card className="w-[350px]">
            <CardHeader>
                <Skeleton className="h-5 w-32" />
                <Skeleton className="mt-2 h-4 w-48" />
            </CardHeader>
            <CardContent>
                <Skeleton className="h-20 w-full" />
            </CardContent>
            <CardFooter>
                <Skeleton className="h-9 w-24" />
            </CardFooter>
        </Card>
    )
};

export const MinimalCard: Story = {
    render: () => (
        <Card className="w-[350px]">
            <CardContent>
                <p>A minimal card with only content.</p>
            </CardContent>
        </Card>
    )
};
