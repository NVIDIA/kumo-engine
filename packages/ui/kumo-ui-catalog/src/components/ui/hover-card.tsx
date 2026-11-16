// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";
import * as HoverCardPrimitive from "@radix-ui/react-hover-card";

import { cn } from "@/lib/utils";

function HoverCard({
    openDelay = 0,
    closeDelay = 100,
    ...props
}: React.ComponentProps<typeof HoverCardPrimitive.Root> & { id?: string }) {
    return (
        <HoverCardPrimitive.Root
            data-slot="hover-card"
            openDelay={openDelay}
            closeDelay={closeDelay}
            {...props}
        />
    );
}

function HoverCardTrigger({
    id,
    ...props
}: React.ComponentProps<typeof HoverCardPrimitive.Trigger> & { id?: string }) {
    return (
        <HoverCardPrimitive.Trigger
            data-slot="hover-card-trigger"
            id={id}
            {...props}
        />
    );
}

function HoverCardContent({
    id,
    className,
    align = "center",
    sideOffset = 15,
    side = "bottom",
    arrow = true,
    ...props
}: React.ComponentProps<typeof HoverCardPrimitive.Content> & {
    id?: string;
    arrow?: boolean;
}) {
    return (
        <HoverCardPrimitive.Portal data-slot="hover-card-portal">
            <HoverCardPrimitive.Content
                id={id}
                data-slot="hover-card-content"
                align={align}
                sideOffset={sideOffset}
                side={side}
                className={cn(
                    "z-1051 border-border-secondary bg-surface-primary inline-flex flex-col items-start justify-center gap-2 rounded-sm border p-4 shadow-[2px_2px_8px_0_rgba(0,0,0,0.25)]",
                    className
                )}
                {...props}>
                {props.children}
                {arrow && (
                    <HoverCardPrimitive.Arrow
                        className="h-2 w-3"
                        style={{
                            fill: "white",
                            stroke: "gray"
                        }}
                    />
                )}
            </HoverCardPrimitive.Content>
        </HoverCardPrimitive.Portal>
    );
}

export { HoverCard, HoverCardTrigger, HoverCardContent };
