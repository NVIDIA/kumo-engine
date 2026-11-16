// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { ScrollArea } from "@/components/ui/scroll-area";

const meta: Meta<typeof ScrollArea> = {
    title: "UI/ScrollArea",
    component: ScrollArea,
    tags: ["autodocs"],
    args: {
        className: "h-40 w-80 border rounded-md"
    }
};

export default meta;
type Story = StoryObj<typeof ScrollArea>;

export const Default: Story = {
    render: (args) => (
        <ScrollArea {...args}>
            <div className="space-y-2 p-4">
                {Array.from({ length: 30 }, (_, i) => (
                    <div
                        key={i}
                        className="rounded bg-white px-3 py-2 text-sm text-slate-700">
                        Item {i + 1}
                    </div>
                ))}
            </div>
        </ScrollArea>
    )
};
