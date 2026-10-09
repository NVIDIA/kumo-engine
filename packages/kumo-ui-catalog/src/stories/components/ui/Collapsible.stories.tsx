// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import {
    Collapsible,
    CollapsibleTrigger,
    CollapsibleContent
} from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { ChevronDown } from "@/icons";
import { expect, userEvent, within } from "storybook/test";

const meta: Meta<typeof Collapsible> = {
    title: "UI/Collapsible",
    component: Collapsible,
    tags: ["autodocs"],
    parameters: {
        docs: {
            description: {
                component:
                    "A collapsible component that can be expanded or collapsed to show or hide content."
            }
        }
    },
    argTypes: {
        open: {
            control: "boolean",
            description: "Whether the collapsible is open"
        },
        defaultOpen: {
            control: "boolean",
            description: "The default open state when initially rendered"
        },
        disabled: {
            control: "boolean",
            description: "Whether the collapsible is disabled"
        },
        onOpenChange: {
            action: "open changed",
            description: "Function called when the open state changes"
        }
    }
};

export default meta;
type Story = StoryObj<typeof Collapsible>;

export const Default: Story = {
    render: () => (
        <Collapsible className="w-[350px] space-y-2">
            <div className="flex items-center justify-between space-x-4 px-4">
                <h4 className="text-sm font-semibold">Toggle Section</h4>
                <CollapsibleTrigger asChild>
                    <Button variant="link" size="sm" className="w-9 p-0">
                        <ChevronDown className="size-4" />
                        <span className="sr-only">Toggle</span>
                    </Button>
                </CollapsibleTrigger>
            </div>
            <CollapsibleContent className="space-y-2">
                <div className="rounded-md border px-4 py-3 text-sm">
                    This is the collapsible content that can be shown or hidden.
                </div>
            </CollapsibleContent>
        </Collapsible>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(
            canvas.queryByText(
                "This is the collapsible content that can be shown or hidden."
            )
        ).not.toBeInTheDocument();

        const trigger = canvas.getByRole("button", { name: /toggle/i });
        await userEvent.click(trigger);
        await expect(
            canvas.getByText(
                "This is the collapsible content that can be shown or hidden."
            )
        ).toBeVisible();
        await userEvent.click(trigger);
        await expect(
            canvas.queryByText(
                "This is the collapsible content that can be shown or hidden."
            )
        ).not.toBeInTheDocument();
    }
};

export const DefaultOpen: Story = {
    render: () => (
        <Collapsible defaultOpen className="w-[350px] space-y-2">
            <div className="flex items-center justify-between space-x-4 px-4">
                <h4 className="text-sm font-semibold">Open by Default</h4>
                <CollapsibleTrigger asChild>
                    <Button variant="link" size="sm" className="w-9 p-0">
                        <ChevronDown className="size-4" />
                        <span className="sr-only">Toggle</span>
                    </Button>
                </CollapsibleTrigger>
            </div>
            <CollapsibleContent className="space-y-2">
                <div className="rounded-md border px-4 py-3 text-sm">
                    This content is visible by default when the component
                    mounts.
                </div>
            </CollapsibleContent>
        </Collapsible>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(
            canvas.getByText(/This content is visible by default/)
        ).toBeVisible();

        const trigger = canvas.getByRole("button", { name: /toggle/i });
        await userEvent.click(trigger);
        await expect(
            canvas.queryByText(/This content is visible by default/)
        ).not.toBeInTheDocument();
    }
};

export const Controlled: Story = {
    render: function ControlledCollapsible() {
        const [isOpen, setIsOpen] = useState(false);

        return (
            <Collapsible
                open={isOpen}
                onOpenChange={setIsOpen}
                className="w-[350px] space-y-2">
                <div className="flex items-center justify-between space-x-4 px-4">
                    <h4 className="text-sm font-semibold">
                        Controlled Collapsible
                    </h4>
                    <CollapsibleTrigger asChild>
                        <Button variant="link" size="sm" className="w-9 p-0">
                            <ChevronDown
                                className={`size-4 transition-transform duration-200 ${
                                    isOpen ? "rotate-180" : ""
                                }`}
                            />
                            <span className="sr-only">Toggle</span>
                        </Button>
                    </CollapsibleTrigger>
                </div>
                <div className="text-muted-foreground px-4 text-sm">
                    State: {isOpen ? "Open" : "Closed"}
                </div>
                <CollapsibleContent className="space-y-2">
                    <div className="rounded-md border px-4 py-3 text-sm">
                        This is a controlled collapsible. The open state is
                        managed externally.
                    </div>
                </CollapsibleContent>
            </Collapsible>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("State: Closed")).toBeVisible();

        const trigger = canvas.getByRole("button", { name: /toggle/i });
        await userEvent.click(trigger);
        await expect(canvas.getByText("State: Open")).toBeVisible();
        await expect(
            canvas.getByText(/This is a controlled collapsible/)
        ).toBeVisible();

        await userEvent.click(trigger);
        await expect(canvas.getByText("State: Closed")).toBeVisible();
    }
};

export const Disabled: Story = {
    render: () => (
        <Collapsible disabled className="w-[350px] space-y-2">
            <div className="flex items-center justify-between space-x-4 px-4">
                <h4 className="text-muted-foreground text-sm font-semibold">
                    Disabled Collapsible
                </h4>
                <CollapsibleTrigger asChild>
                    <Button
                        variant="link"
                        size="sm"
                        className="w-9 p-0"
                        disabled>
                        <ChevronDown className="size-4" />
                        <span className="sr-only">Toggle</span>
                    </Button>
                </CollapsibleTrigger>
            </div>
            <CollapsibleContent className="space-y-2">
                <div className="rounded-md border px-4 py-3 text-sm">
                    This content cannot be toggled.
                </div>
            </CollapsibleContent>
        </Collapsible>
    ),
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getByRole("button", { name: /toggle/i });
        await expect(trigger).toBeDisabled();
        await expect(
            canvas.queryByText("This content cannot be toggled.")
        ).not.toBeInTheDocument();
    }
};

export const WithMultipleItems: Story = {
    render: function MultipleItemsCollapsible() {
        const [isOpen, setIsOpen] = useState(false);

        return (
            <Collapsible
                open={isOpen}
                onOpenChange={setIsOpen}
                className="w-[350px] space-y-2">
                <div className="flex items-center justify-between space-x-4 px-4">
                    <h4 className="text-sm font-semibold">
                        @kumo/ui-catalog
                        <span className="bg-muted text-muted-foreground ml-2 rounded-sm px-1.5 py-0.5 text-xs font-normal">
                            3 items
                        </span>
                    </h4>
                    <CollapsibleTrigger asChild>
                        <Button variant="link" size="sm" className="w-9 p-0">
                            <ChevronDown
                                className={`size-4 transition-transform duration-200 ${
                                    isOpen ? "rotate-180" : ""
                                }`}
                            />
                            <span className="sr-only">Toggle</span>
                        </Button>
                    </CollapsibleTrigger>
                </div>
                <div className="rounded-md border px-4 py-3 font-mono text-sm">
                    @kumo/components
                </div>
                <CollapsibleContent className="space-y-2">
                    <div className="rounded-md border px-4 py-3 font-mono text-sm">
                        @kumo/icons
                    </div>
                    <div className="rounded-md border px-4 py-3 font-mono text-sm">
                        @kumo/utils
                    </div>
                </CollapsibleContent>
            </Collapsible>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("@kumo/components")).toBeVisible();
        await expect(canvas.queryByText("@kumo/icons")).not.toBeInTheDocument();
        await expect(canvas.queryByText("@kumo/utils")).not.toBeInTheDocument();

        const trigger = canvas.getByRole("button", { name: /toggle/i });
        await userEvent.click(trigger);
        await expect(canvas.getByText("@kumo/icons")).toBeVisible();
        await expect(canvas.getByText("@kumo/utils")).toBeVisible();
    }
};

export const CustomTrigger: Story = {
    render: function CustomTriggerCollapsible() {
        const [isOpen, setIsOpen] = useState(false);

        return (
            <Collapsible
                open={isOpen}
                onOpenChange={setIsOpen}
                className="w-[350px]">
                <CollapsibleTrigger asChild>
                    <Button
                        variant="light-outline"
                        className="w-full justify-between">
                        Click to {isOpen ? "collapse" : "expand"}
                        <ChevronDown
                            className={`size-4 transition-transform duration-200 ${
                                isOpen ? "rotate-180" : ""
                            }`}
                        />
                    </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="mt-2">
                    <div className="bg-muted rounded-md border p-4 text-sm">
                        <p>
                            This example shows a custom trigger using a
                            full-width button with dynamic text.
                        </p>
                    </div>
                </CollapsibleContent>
            </Collapsible>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const trigger = canvas.getByRole("button", {
            name: /click to expand/i
        });
        await expect(trigger).toBeVisible();
        await userEvent.click(trigger);
        await expect(
            canvas.getByRole("button", { name: /click to collapse/i })
        ).toBeVisible();
        await expect(
            canvas.getByText(/custom trigger using a full-width button/)
        ).toBeVisible();

        await userEvent.click(
            canvas.getByRole("button", { name: /click to collapse/i })
        );
        await expect(
            canvas.getByRole("button", { name: /click to expand/i })
        ).toBeVisible();
    }
};
