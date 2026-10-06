// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as AccordionPrimitive from "@radix-ui/react-accordion";
import * as React from "react";

import { ChevronDown } from "@/icons/MaterialIcons/ChevronDown";
import { ChevronUp } from "@/icons/MaterialIcons/ChevronUp";
import { cn } from "@/lib/utils";
import { useState } from "react";

function Accordion({
    ...props
}: React.ComponentProps<typeof AccordionPrimitive.Root>) {
    return <AccordionPrimitive.Root data-slot="accordion" {...props} />;
}

function AccordionItem({
    className,
    ...props
}: React.ComponentProps<typeof AccordionPrimitive.Item>) {
    return (
        <AccordionPrimitive.Item
            data-slot="accordion-item"
            className={cn("", className)}
            {...props}
        />
    );
}

function AccordionTrigger({
    className,
    children,
    showArrow = true,
    showBorder = true,
    ...props
}: React.ComponentProps<typeof AccordionPrimitive.Trigger> & {
    asChild?: boolean;
    showArrow?: boolean;
    showBorder?: boolean;
}) {
    const [open, setOpen] = useState<boolean>(false);
    return (
        <AccordionPrimitive.Header className="!m-0">
            <AccordionPrimitive.Trigger
                asChild
                onClick={() => setOpen(!open)}
                data-slot="accordion-trigger"
                className={cn(
                    "grid w-full cursor-pointer grid-cols-[1fr_auto] rounded bg-white p-2",
                    showBorder &&
                        "outline-1 outline-offset-[-2px] outline-slate-200",
                    className
                )}
                {...props}>
                <button type="button">
                    <span>{children}</span>
                    {showArrow && (
                        <span className="text-muted-foreground ml-auto flex items-center">
                            {open ? (
                                <ChevronUp className="size-5.5" />
                            ) : (
                                <ChevronDown className="size-5.5" />
                            )}
                        </span>
                    )}
                </button>
            </AccordionPrimitive.Trigger>
        </AccordionPrimitive.Header>
    );
}

function AccordionContent({
    className,
    children,
    showBorder = true,
    ...props
}: React.ComponentProps<typeof AccordionPrimitive.Content> & {
    showBorder?: boolean;
}) {
    return (
        <AccordionPrimitive.Content
            data-slot="accordion-content"
            className={cn(
                "data-[state=closed]:h-0 data-[state=closed]:overflow-hidden",
                "data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down overflow-hidden bg-white text-sm",
                showBorder &&
                    "outline-1 outline-offset-[-2px] outline-slate-200"
            )}
            {...props}>
            <div className={cn("", className)}>{children}</div>
        </AccordionPrimitive.Content>
    );
}

export { Accordion, AccordionContent, AccordionItem, AccordionTrigger };
