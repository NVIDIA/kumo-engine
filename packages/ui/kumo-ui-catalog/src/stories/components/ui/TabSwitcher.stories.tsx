// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";

import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, fn, userEvent, within } from "storybook/test";

import { TabSwitcher } from "@/components/ui/tab-switcher";

const meta = {
    title: "Components/UI/TabSwitcher",
    component: TabSwitcher,
    tags: ["autodocs"],
    parameters: { layout: "centered" },
    argTypes: {
        onValueChange: { action: "valueChanged" }
    }
} satisfies Meta<typeof TabSwitcher>;

export default meta;
type Story = StoryObj<typeof meta>;

const groupTabs = [
    { label: "Single group", value: "single" },
    { label: "Multiple groups", value: "multiple" }
];

const manyTabs = [
    { label: "Overview", value: "overview" },
    { label: "Analytics", value: "analytics" },
    { label: "Reports", value: "reports" },
    { label: "Settings", value: "settings" }
];

export const Default: Story = {
    args: {
        tabs: groupTabs,
        defaultValue: "single"
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const singleTab = canvas.getByRole("tab", { name: /single group/i });
        await expect(singleTab).toHaveAttribute("aria-selected", "true");

        const multipleTab = canvas.getByRole("tab", {
            name: /multiple groups/i
        });
        await userEvent.click(multipleTab);
        await expect(multipleTab).toHaveAttribute("aria-selected", "true");
        await expect(singleTab).toHaveAttribute("aria-selected", "false");
    }
};

export const Controlled: Story = {
    args: {
        tabs: groupTabs,
        value: "multiple",
        onValueChange: fn()
    }
};

export const ManyTabs: Story = {
    args: {
        tabs: manyTabs,
        defaultValue: "overview"
    }
};

export const WithCallback: Story = {
    args: {
        tabs: groupTabs,
        defaultValue: "single",
        onValueChange: fn()
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const multipleTab = canvas.getByRole("tab", {
            name: /multiple groups/i
        });
        await userEvent.click(multipleTab);
        await expect(args.onValueChange).toHaveBeenCalledWith("multiple");
    }
};

function WithContentDemo() {
    const [active, setActive] = React.useState("single");
    return (
        <div className="w-96 space-y-4">
            <TabSwitcher
                tabs={groupTabs}
                value={active}
                onValueChange={setActive}
            />
            <div className="text-muted-foreground text-sm">
                {active === "single"
                    ? "Configure a single group."
                    : "Configure multiple groups at once."}
            </div>
        </div>
    );
}

export const WithContent: Story = {
    args: { tabs: groupTabs, defaultValue: "single" },
    render: () => <WithContentDemo />
};
