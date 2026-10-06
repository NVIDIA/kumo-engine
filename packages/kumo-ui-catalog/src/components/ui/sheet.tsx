"use client";
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-License-Identifier: Apache-2.0 AND MIT


import * as SheetPrimitive from "@radix-ui/react-dialog";
import * as React from "react";

import { Close } from "@/icons/MaterialIcons/Close";
import { cn } from "@/lib/utils";

function Sheet({ ...props }: React.ComponentProps<typeof SheetPrimitive.Root>) {
    return <SheetPrimitive.Root data-slot="sheet" {...props} />;
}

function SheetTrigger({
    ...props
}: React.ComponentProps<typeof SheetPrimitive.Trigger>) {
    return <SheetPrimitive.Trigger data-slot="sheet-trigger" {...props} />;
}

function SheetClose({
    ...props
}: React.ComponentProps<typeof SheetPrimitive.Close>) {
    return <SheetPrimitive.Close data-slot="sheet-close" {...props} />;
}

function SheetPortal({
    ...props
}: React.ComponentProps<typeof SheetPrimitive.Portal>) {
    return <SheetPrimitive.Portal data-slot="sheet-portal" {...props} />;
}

function SheetOverlay({
    className,
    ...props
}: React.ComponentProps<typeof SheetPrimitive.Overlay>) {
    return (
        <SheetPrimitive.Overlay
            data-slot="sheet-overlay"
            className={cn(
                "data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0 fixed inset-0 z-50 bg-black/80",
                className
            )}
            {...props}
        />
    );
}

function SheetContent({
    className,
    children,
    side = "right",
    showCloseButton = false,
    ...props
}: React.ComponentProps<typeof SheetPrimitive.Content> & {
    side?: "top" | "right" | "bottom" | "left";
    showCloseButton?: boolean;
}) {
    return (
        <SheetPortal>
            <SheetOverlay />
            <SheetPrimitive.Content
                data-slot="sheet-content"
                className={cn(
                    "bg-background data-[state=closed]:animate-out data-[state=open]:animate-in fixed z-50 flex flex-col rounded-lg shadow-lg transition ease-in-out data-[state=closed]:duration-300 data-[state=open]:duration-500",
                    side === "right" &&
                        "data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right inset-y-2 right-2 h-[calc(100%-16px)] w-[80vw]",
                    side === "left" &&
                        "data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left inset-y-2 left-2 h-[calc(100%-16px)] w-[80vw]",
                    side === "top" &&
                        "data-[state=closed]:slide-out-to-top data-[state=open]:slide-in-from-top inset-x-2 top-2 h-[80vh] w-[calc(100%-16px)]",
                    side === "bottom" &&
                        "data-[state=closed]:slide-out-to-bottom data-[state=open]:slide-in-from-bottom inset-x-2 bottom-2 h-[80vh] w-[calc(100%-16px)]",
                    className
                )}
                {...props}
                onWheel={(e) => {
                    // Allow scroll events to propagate to nested scrollable elements
                    // This fixes dropdown scrolling issues inside Sheet
                    const target = e.target as HTMLElement;

                    // Check if the target or any of its parents (up to current element) is scrollable
                    let element: HTMLElement | null = target;
                    while (element && element !== e.currentTarget) {
                        const isScrollable =
                            element.scrollHeight > element.clientHeight;

                        if (isScrollable) {
                            // Found a scrollable ancestor, stop propagation to allow it to scroll
                            e.stopPropagation();
                            return;
                        }

                        element = element.parentElement;
                    }
                }}>
                {/* Custom scrollbar with div */}
                <div className="flex-1 overflow-x-auto overflow-y-auto outline-none [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-[var(--scrollbar-dark)] [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar]:h-2.5 [&::-webkit-scrollbar]:w-2.5">
                    {children}
                </div>

                {showCloseButton && (
                    <SheetPrimitive.Close className="focus:outline-hidden absolute right-4 top-4 hover:opacity-100 disabled:pointer-events-none">
                        <Close />
                        <span className="sr-only">Close</span>
                    </SheetPrimitive.Close>
                )}
            </SheetPrimitive.Content>
        </SheetPortal>
    );
}

function SheetHeader({ className, ...props }: React.ComponentProps<"div">) {
    return (
        <div
            data-slot="sheet-header"
            className={cn("flex flex-col p-4", className)}
            {...props}
        />
    );
}

function SheetFooter({ className, ...props }: React.ComponentProps<"div">) {
    return (
        <div
            data-slot="sheet-footer"
            className={cn("mt-auto flex flex-col p-4", className)}
            {...props}
        />
    );
}

function SheetTitle({
    className,
    ...props
}: React.ComponentProps<typeof SheetPrimitive.Title>) {
    return (
        <SheetPrimitive.Title
            data-slot="sheet-title"
            className={cn("text-foreground font-semibold", className)}
            {...props}
        />
    );
}

function SheetDescription({
    className,
    ...props
}: React.ComponentProps<typeof SheetPrimitive.Description>) {
    return (
        <SheetPrimitive.Description
            data-slot="sheet-description"
            className={cn("text-muted-foreground text-sm", className)}
            {...props}
        />
    );
}

export {
    Sheet,
    SheetClose,
    SheetContent,
    SheetDescription,
    SheetFooter,
    SheetHeader,
    SheetTitle,
    SheetTrigger
};
