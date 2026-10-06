// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Separator } from "@/components/ui/separator";
import { Meta, StoryObj } from "@storybook/react-vite";

const meta = {
    title: "UI/Separator",
    component: Separator,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"]
} satisfies Meta<typeof Separator>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Horizontal: Story = {
    render: () => (
        <div className="w-64">
            <div className="space-y-1">
                <h4 className="text-sm font-medium leading-none">
                    Radix Primitives
                </h4>
                <p className="text-muted-foreground text-sm">
                    An open-source UI component library.
                </p>
            </div>
            <Separator className="my-4" />
            <div className="flex h-5 items-center space-x-4 text-sm">
                <div>Blog</div>
                <div>Docs</div>
                <div>Source</div>
            </div>
        </div>
    )
};

export const Vertical: Story = {
    render: () => (
        <div className="flex h-5 items-center space-x-4 text-sm">
            <div>Blog</div>
            <Separator orientation="vertical" />
            <div>Docs</div>
            <Separator orientation="vertical" />
            <div>Source</div>
        </div>
    )
};

export const SettingsPanel: Story = {
    render: () => (
        <div className="bg-card w-80 rounded-sm border p-4">
            <div className="space-y-4">
                <div>
                    <h3 className="text-sm font-medium">Prediction Settings</h3>
                    <p className="text-muted-foreground text-xs">
                        Configure prediction job parameters
                    </p>
                </div>
                <Separator />
                <div className="space-y-2">
                    <label className="text-sm font-medium">Entity Filter</label>
                    <div className="bg-muted/20 text-muted-foreground h-8 rounded-sm border px-3 py-1.5 text-sm">
                        WHERE entity_id IS NOT NULL
                    </div>
                </div>
                <div className="space-y-2">
                    <label className="text-sm font-medium">
                        Parallel Workers
                    </label>
                    <div className="bg-muted/20 h-8 rounded-sm border px-3 py-1.5 text-sm">
                        4
                    </div>
                </div>
                <Separator />
                <div className="space-y-2">
                    <label className="text-sm font-medium">
                        Output Configuration
                    </label>
                    <div className="flex items-center gap-2">
                        <div className="bg-muted/20 text-muted-foreground h-8 flex-1 rounded-sm border px-3 py-1.5 text-sm">
                            s3://bucket/predictions
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
};

export const UserProfileCard: Story = {
    render: () => (
        <div className="bg-card w-64 rounded-sm border p-4">
            <div className="flex flex-col items-center gap-2 text-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-sm bg-slate-200 text-lg font-medium text-slate-800">
                    JD
                </div>
                <div>
                    <p className="text-sm font-medium">John Doe</p>
                    <p className="text-muted-foreground text-xs">
                        john.doe@example.com
                    </p>
                </div>
            </div>
            <Separator className="my-4" />
            <button className="hover:bg-muted/50 w-full rounded-sm border px-3 py-2 text-center text-sm font-medium">
                Log Out
            </button>
        </div>
    )
};

export const SidebarMenuGroups: Story = {
    render: () => (
        <div className="bg-card w-56 rounded-sm border p-2">
            <div className="space-y-1">
                <div className="hover:bg-muted/50 flex cursor-pointer items-center gap-2 rounded-sm px-3 py-2 text-sm">
                    <span className="h-4 w-4 rounded-sm bg-slate-300" />
                    Models
                </div>
                <div className="hover:bg-muted/50 flex cursor-pointer items-center gap-2 rounded-sm px-3 py-2 text-sm">
                    <span className="h-4 w-4 rounded-sm bg-slate-300" />
                    Predictions
                </div>
            </div>
            <Separator className="my-2" />
            <div className="space-y-1">
                <div className="hover:bg-muted/50 flex cursor-pointer items-center gap-2 rounded-sm px-3 py-2 text-sm">
                    <span className="h-4 w-4 rounded-sm bg-slate-300" />
                    Graphs
                </div>
                <div className="hover:bg-muted/50 flex cursor-pointer items-center gap-2 rounded-sm px-3 py-2 text-sm">
                    <span className="h-4 w-4 rounded-sm bg-slate-300" />
                    Tables
                </div>
                <div className="hover:bg-muted/50 flex cursor-pointer items-center gap-2 rounded-sm px-3 py-2 text-sm">
                    <span className="h-4 w-4 rounded-sm bg-slate-300" />
                    Connectors
                </div>
            </div>
            <Separator className="my-2" />
            <div className="space-y-1">
                <div className="hover:bg-muted/50 flex cursor-pointer items-center gap-2 rounded-sm px-3 py-2 text-sm">
                    <span className="h-4 w-4 rounded-sm bg-slate-300" />
                    Documentation
                </div>
                <div className="hover:bg-muted/50 flex cursor-pointer items-center gap-2 rounded-sm px-3 py-2 text-sm">
                    <span className="h-4 w-4 rounded-sm bg-slate-300" />
                    Admin
                </div>
            </div>
        </div>
    )
};

export const ToolbarWithDividers: Story = {
    render: () => (
        <div className="bg-card flex items-center gap-2 rounded-sm border p-2">
            <button
                type="button"
                aria-label="Bold"
                className="hover:bg-muted/50 rounded-sm p-2">
                <div className="h-4 w-4 rounded-sm bg-slate-400" />
            </button>
            <button
                type="button"
                aria-label="Italic"
                className="hover:bg-muted/50 rounded-sm p-2">
                <div className="h-4 w-4 rounded-sm bg-slate-400" />
            </button>
            <Separator orientation="vertical" className="h-6" />
            <button
                type="button"
                aria-label="Align left"
                className="hover:bg-muted/50 rounded-sm p-2">
                <div className="h-4 w-4 rounded-sm bg-slate-400" />
            </button>
            <button
                type="button"
                aria-label="Align center"
                className="hover:bg-muted/50 rounded-sm p-2">
                <div className="h-4 w-4 rounded-sm bg-slate-400" />
            </button>
            <button
                type="button"
                aria-label="Align right"
                className="hover:bg-muted/50 rounded-sm p-2">
                <div className="h-4 w-4 rounded-sm bg-slate-400" />
            </button>
            <Separator orientation="vertical" className="h-6" />
            <button
                type="button"
                aria-label="Insert link"
                className="hover:bg-muted/50 rounded-sm p-2">
                <div className="h-4 w-4 rounded-sm bg-slate-400" />
            </button>
        </div>
    )
};
