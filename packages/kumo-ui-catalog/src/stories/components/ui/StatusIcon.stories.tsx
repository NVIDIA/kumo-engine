// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusIcon } from "@/components/ui/status-icon";
import {
    AutoRenew,
    BrightnessAlert,
    CheckCircle,
    NextPlan,
    TabClose
} from "@/icons";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect } from "storybook/test";

type JobStatus =
    | "queued"
    | "running"
    | "done"
    | "failed"
    | "cancelled"
    | "untrained";

const jobIconMap: Record<JobStatus, React.ReactNode> = {
    queued: <NextPlan className="fill-yellow-500" />,
    running: <AutoRenew className="fill-violet-500" />,
    done: <CheckCircle className="fill-emerald-500" />,
    failed: <BrightnessAlert className="fill-red-500" />,
    cancelled: <TabClose className="fill-slate-500" />,
    untrained: null
};

const meta: Meta<typeof StatusIcon<JobStatus>> = {
    title: "UI/StatusIcon",
    component: StatusIcon,
    tags: ["autodocs"],
    parameters: { layout: "centered" }
};

export default meta;

type Story = StoryObj<typeof StatusIcon<JobStatus>>;

export const Done: Story = {
    args: { status: "done", iconMap: jobIconMap },
    play: async ({ canvasElement }) => {
        const el = canvasElement.querySelector('[data-slot="status-icon"]');
        await expect(el).not.toBeNull();
        await expect(el).toHaveAttribute("data-status", "done");
    }
};

export const Running: Story = {
    args: { status: "running", iconMap: jobIconMap }
};

export const Failed: Story = {
    args: { status: "failed", iconMap: jobIconMap }
};

export const UnknownFallback: Story = {
    args: {
        status: "untrained",
        iconMap: jobIconMap,
        fallback: <span className="text-title-secondary text-xs">N/A</span>
    }
};
