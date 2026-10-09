"use client";
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-License-Identifier: Apache-2.0 AND MIT


import * as DialogPrimitive from "@radix-ui/react-dialog";
import * as React from "react";

import { Close } from "@/icons/MaterialIcons/Close";
import { cn } from "@/lib/utils";
import { Separator } from "./separator";

function Dialog({
    ...props
}: React.ComponentProps<typeof DialogPrimitive.Root>) {
    return <DialogPrimitive.Root data-slot="dialog" {...props} />;
}

function DialogTrigger({
    ...props
}: React.ComponentProps<typeof DialogPrimitive.Trigger>) {
    return <DialogPrimitive.Trigger data-slot="dialog-trigger" {...props} />;
}

function DialogPortal({
    ...props
}: React.ComponentProps<typeof DialogPrimitive.Portal>) {
    return <DialogPrimitive.Portal data-slot="dialog-portal" {...props} />;
}

function DialogClose({
    ...props
}: React.ComponentProps<typeof DialogPrimitive.Close>) {
    return <DialogPrimitive.Close data-slot="dialog-close" {...props} />;
}

function DialogOverlay({
    className,
    ...props
}: React.ComponentProps<typeof DialogPrimitive.Overlay>) {
    return (
        <DialogPrimitive.Overlay
            data-slot="dialog-overlay"
            className={cn(
                "z-1200 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0 fixed inset-0 bg-black/50",
                className
            )}
            {...props}
        />
    );
}

function DialogContent({
    enableClose = false,
    className,
    children,
    ...props
}: React.ComponentProps<typeof DialogPrimitive.Content> & {
    enableClose?: boolean;
}) {
    const initialFocusRef = React.useRef<HTMLDivElement>(null);
    return (
        <DialogPortal data-slot="dialog-portal">
            <DialogOverlay />
            <DialogPrimitive.Content
                data-slot="dialog-content"
                className={cn(
                    "z-1200 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 fixed left-[50%] top-[35%] grid w-full max-w-[calc(100%-2rem)] translate-x-[-50%] translate-y-[-50%] gap-1 rounded-lg border bg-white p-1 shadow-lg duration-200 sm:max-w-lg",
                    className
                )}
                onOpenAutoFocus={(e) => {
                    e.preventDefault();
                    initialFocusRef.current?.focus();
                }}
                {...props}>
                {children}
                {enableClose && (
                    <DialogPrimitive.Close className="absolute right-4 top-4 border-none bg-transparent">
                        <Close className="size-4" />
                    </DialogPrimitive.Close>
                )}
            </DialogPrimitive.Content>
        </DialogPortal>
    );
}

function DialogHeader({ className, ...props }: React.ComponentProps<"div">) {
    return (
        <>
            <div
                data-slot="dialog-header"
                className={cn(
                    "flex flex-col gap-1 px-3 py-2 text-center sm:text-left",
                    className
                )}
                {...props}
            />
            <Separator className="ml-0 flex w-full" />
        </>
    );
}

function DialogFooter({ className, ...props }: React.ComponentProps<"div">) {
    return (
        <>
            <Separator className="ml-0 flex w-full" />
            <div
                data-slot="dialog-footer"
                className={cn(
                    "flex flex-col-reverse gap-2 px-3 py-2 sm:flex-row sm:justify-end",
                    className
                )}
                {...props}
            />
        </>
    );
}

function DialogTitle({
    className,
    ...props
}: React.ComponentProps<typeof DialogPrimitive.Title>) {
    return (
        <DialogPrimitive.Title
            data-slot="dialog-title"
            className={cn("!m-0 text-xl font-semibold leading-none", className)}
            {...props}
        />
    );
}

function DialogDescription({
    className,
    ...props
}: React.ComponentProps<typeof DialogPrimitive.Description>) {
    return (
        <DialogPrimitive.Description
            data-slot="dialog-description"
            className={cn("!m-0 px-3 py-2 text-sm", className)}
            {...props}
        />
    );
}

export {
    Dialog,
    DialogClose,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogOverlay,
    DialogPortal,
    DialogTitle,
    DialogTrigger
};
