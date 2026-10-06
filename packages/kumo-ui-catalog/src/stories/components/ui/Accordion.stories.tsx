// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";

import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { DeveloperBoard } from "@/icons";
import { useState } from "react";
import { expect, userEvent, within } from "storybook/test";

const meta: Meta<typeof Accordion> = {
    title: "UI/Accordion",
    component: Accordion,
    tags: ["autodocs"],
    parameters: {
        docs: {
            description: {
                component:
                    "Accordion component with customizable styling. Borders shown by default, use showBorder={false} to remove."
            }
        }
    }
};
export default meta;

type Story = StoryObj<typeof Accordion>;

function ContentComponent({ val }: { val: string }) {
    return <div>Accordion Content {val}</div>;
}

export const Default: Story = {
    render: function DefaultFunc() {
        const [openItem, setOpenItem] = useState<string[] | undefined>([]);
        return (
            <Accordion
                {...{ type: "multiple" }}
                value={openItem}
                onValueChange={setOpenItem}>
                <AccordionItem value="item-1" className="mb-2">
                    <AccordionTrigger>
                        <div className="flex items-center justify-between gap-1">
                            <div className="flex items-center gap-1">
                                <DeveloperBoard className="size-4" />
                                <div className="text-title-primary text-xs font-bold">
                                    GRAPH SNAPSHOT (WITH BORDERS - DEFAULT)
                                </div>
                                {!openItem?.includes("item-1") && (
                                    <div className="text-muted-foreground text-xs font-bold">
                                        JOB ID
                                    </div>
                                )}
                            </div>
                            <div className="flex items-center gap-1">
                                <Badge size="sm" variant="done">
                                    Done
                                </Badge>
                            </div>
                        </div>
                    </AccordionTrigger>
                    <AccordionContent className="p-2" forceMount>
                        <ContentComponent val={"item-1"} />
                    </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-2">
                    <AccordionTrigger>
                        <div className="flex items-center justify-between gap-1">
                            <div className="flex items-center gap-1">
                                <DeveloperBoard className="size-4" />
                                <div className="text-title-primary text-xs font-bold">
                                    PREDICTION TABLE GENERATION SNAPSHOT
                                </div>
                            </div>
                            <div className="flex items-center gap-1">
                                <Badge size="sm" variant="done">
                                    Done
                                </Badge>
                            </div>
                        </div>
                    </AccordionTrigger>
                    <AccordionContent className="p-2" forceMount>
                        <ContentComponent val={"item-2"} />
                    </AccordionContent>
                </AccordionItem>
            </Accordion>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("JOB ID")).toBeVisible();

        const firstTrigger = canvas.getByText(
            "GRAPH SNAPSHOT (WITH BORDERS - DEFAULT)"
        );
        await userEvent.click(firstTrigger);
        await expect(canvas.queryByText("JOB ID")).not.toBeInTheDocument();
        await expect(
            canvas.getByText("Accordion Content item-1")
        ).toBeVisible();

        const secondTrigger = canvas.getByText(
            "PREDICTION TABLE GENERATION SNAPSHOT"
        );
        await userEvent.click(secondTrigger);
        await expect(
            canvas.getByText("Accordion Content item-1")
        ).toBeVisible();
        await expect(
            canvas.getByText("Accordion Content item-2")
        ).toBeVisible();
        await userEvent.click(firstTrigger);
        await expect(canvas.getByText("JOB ID")).toBeVisible();
    }
};

export const No_Borders: Story = {
    render: function DefaultFunc() {
        const [openItem, setOpenItem] = useState<string[] | undefined>([]);
        return (
            <Accordion
                {...{ type: "multiple" }}
                value={openItem}
                onValueChange={setOpenItem}>
                <AccordionItem value="item-1" className="mb-2">
                    <AccordionTrigger showBorder={false}>
                        <div className="flex items-center justify-between gap-1">
                            <div className="flex items-center gap-1">
                                <DeveloperBoard className="size-4" />
                                <div className="text-title-primary text-xs font-bold">
                                    WITHOUT BORDERS
                                </div>
                            </div>
                            <div className="flex items-center gap-1">
                                <Badge size="sm" variant="done">
                                    Done
                                </Badge>
                            </div>
                        </div>
                    </AccordionTrigger>
                    <AccordionContent className="p-2" showBorder={false}>
                        <ContentComponent val={"item-1"} />
                    </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-2">
                    <AccordionTrigger showBorder={false}>
                        <div className="flex items-center justify-between gap-1">
                            <div className="flex items-center gap-1">
                                <DeveloperBoard className="size-4" />
                                <div className="text-title-primary text-xs font-bold">
                                    SECOND ITEM WITHOUT BORDER
                                </div>
                            </div>
                            <div className="flex items-center gap-1">
                                <Badge size="sm" variant="done">
                                    Done
                                </Badge>
                            </div>
                        </div>
                    </AccordionTrigger>
                    <AccordionContent className="p-2" showBorder={false}>
                        <ContentComponent val={"item-2"} />
                    </AccordionContent>
                </AccordionItem>
            </Accordion>
        );
    }
};

export const No_Arrow: Story = {
    render: function DefaultFunc() {
        const [openItem, setOpenItem] = useState<string[] | undefined>(
            undefined
        );
        return (
            <Accordion
                {...{ type: "multiple" }}
                value={openItem}
                onValueChange={setOpenItem}>
                <AccordionItem value="item-1">
                    <AccordionTrigger showArrow={false}>
                        <div className="flex items-center justify-between gap-1">
                            <div className="flex items-center gap-1">
                                <DeveloperBoard className="size-4" />
                                <div className="text-title-primary text-xs font-bold">
                                    NO ARROW
                                </div>
                            </div>
                            <div className="flex items-center gap-1">
                                <Badge size="sm" variant="done">
                                    Done
                                </Badge>
                            </div>
                        </div>
                    </AccordionTrigger>
                    <AccordionContent className="bg-gray-100 p-2">
                        This is an Accordion content.
                    </AccordionContent>
                </AccordionItem>
            </Accordion>
        );
    }
};
