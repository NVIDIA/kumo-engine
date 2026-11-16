// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area";

import { cn } from "@/lib/utils";

function ScrollArea({
    className,
    children,
    ...props
}: React.ComponentProps<typeof ScrollAreaPrimitive.Root>) {
    return (
        <ScrollAreaPrimitive.Root
            data-slot="scroll-area"
            style={{ height: 200 }}
            className={cn("relative", className)}
            {...props}>
            <ScrollAreaPrimitive.Viewport
                data-slot="scroll-area-viewport"
                tabIndex={0}
                role="region"
                aria-label="Scrollable content"
                className="focus-visible:ring-ring/50 size-full rounded-[inherit] outline-none transition-[color,box-shadow] focus-visible:outline-1 focus-visible:ring-[3px]">
                {children}
            </ScrollAreaPrimitive.Viewport>
            <ScrollBar />
            <ScrollAreaPrimitive.Corner />
        </ScrollAreaPrimitive.Root>
    );
}

function ScrollBar({
    className,
    orientation = "vertical",
    ...props
}: React.ComponentProps<typeof ScrollAreaPrimitive.ScrollAreaScrollbar>) {
    return (
        <ScrollAreaPrimitive.ScrollAreaScrollbar
            data-slot="scroll-area-scrollbar"
            orientation={orientation}
            className={cn(
                "flex touch-none select-none transition-colors",
                orientation === "vertical" &&
                    "h-full w-3 border-l border-l-transparent",
                orientation === "horizontal" &&
                    "h-2.5 flex-col border-t border-t-transparent",
                className
            )}
            style={
                orientation === "vertical"
                    ? {
                          position: "absolute",
                          top: 5,
                          bottom: 20,
                          height: "calc(100% - 30px)"
                      }
                    : undefined
            }
            {...props}>
            <ScrollAreaPrimitive.ScrollAreaThumb
                data-slot="scroll-area-thumb"
                className="relative flex-1 rounded-full bg-[var(--scrollbar)]"
            />
        </ScrollAreaPrimitive.ScrollAreaScrollbar>
    );
}

export { ScrollArea, ScrollBar };
