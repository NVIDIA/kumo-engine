// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";
import { Command as CommandPrimitive } from "cmdk";

import { cn } from "@/lib/utils";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle
} from "@/components/ui/dialog";
import { Search } from "@/icons/MaterialIcons/Search";

function Command({
    className,
    ...props
}: React.ComponentProps<typeof CommandPrimitive>) {
    return (
        <CommandPrimitive
            data-slot="command"
            className={cn(
                "bg-popover text-popover-foreground flex h-full w-full flex-col overflow-hidden rounded-md",
                className
            )}
            {...props}
        />
    );
}

function CommandDialog({
    title = "Command Palette",
    description = "Search for a command to run...",
    children,
    ...props
}: React.ComponentProps<typeof Dialog> & {
    title?: string;
    description?: string;
}) {
    return (
        <Dialog {...props}>
            <DialogHeader className="sr-only">
                <DialogTitle>{title}</DialogTitle>
                <DialogDescription>{description}</DialogDescription>
            </DialogHeader>
            <DialogContent className="overflow-hidden p-0">
                <Command className="**:data-[slot=command-input-wrapper]:h-12 **:[[cmdk-group-heading]]:px-2 **:[[cmdk-group-heading]]:font-medium **:[[cmdk-group-heading]]:text-muted-foreground **:[[cmdk-group]]:px-2 **:[[cmdk-input]]:h-12 **:[[cmdk-item]]:px-2 **:[[cmdk-item]]:py-3 [&_[cmdk-group]:not([hidden])_~[cmdk-group]]:pt-0 [&_[cmdk-input-wrapper]_svg]:h-5 [&_[cmdk-input-wrapper]_svg]:w-5 [&_[cmdk-item]_svg]:h-5 [&_[cmdk-item]_svg]:w-5">
                    {children}
                </Command>
            </DialogContent>
        </Dialog>
    );
}

function CommandInput({
    className,
    ...props
}: React.ComponentProps<typeof CommandPrimitive.Input>) {
    return (
        <div
            data-slot="command-input-wrapper"
            className={cn(
                "relative m-2 flex h-8 items-center rounded-sm border border-gray-200 bg-white px-2 py-2 focus-within:border-pink-500 hover:border-pink-500",
                className
            )}>
            <Search className="fill-title-secondary mr-1 size-4 shrink-0" />
            <CommandPrimitive.Input
                data-slot="command-input"
                className={cn(
                    "placeholder:text-text-secondary flex h-9 w-full border-none bg-transparent py-2 text-sm outline-none"
                )}
                {...props}
            />
        </div>
    );
}

function CommandList({
    className,
    ...props
}: React.ComponentProps<typeof CommandPrimitive.List>) {
    return (
        <CommandPrimitive.List
            data-slot="command-list"
            tabIndex={0}
            className={cn(
                "max-w-200 max-h-[300px] scroll-py-1 overflow-x-auto overflow-y-auto",
                "[&::-webkit-scrollbar-thumb]:bg-(--scrollbar) [&::-webkit-scrollbar-track]:bg-secondary relative w-full scroll-smooth [-ms-overflow-style:auto] [&::-webkit-scrollbar-thumb]:rounded-md [&::-webkit-scrollbar:horizontal]:h-3 [&::-webkit-scrollbar:vertical]:w-3 [&::-webkit-scrollbar]:opacity-100 [&::-webkit-scrollbar]:transition-opacity [&::-webkit-scrollbar]:duration-1000 [&::-webkit-scrollbar]:ease-in-out",
                className
            )}
            {...props}
        />
    );
}

function CommandEmpty({
    ...props
}: React.ComponentProps<typeof CommandPrimitive.Empty>) {
    return (
        <CommandPrimitive.Empty
            data-slot="command-empty"
            className="py-6 text-center text-sm"
            {...props}
        />
    );
}

function CommandGroup({
    className,
    ...props
}: React.ComponentProps<typeof CommandPrimitive.Group>) {
    return (
        <CommandPrimitive.Group
            data-slot="command-group"
            className={cn(
                "text-foreground **:[[cmdk-group-heading]]:px-2 **:[[cmdk-group-heading]]:py-1.5 **:[[cmdk-group-heading]]:text-xs **:[[cmdk-group-heading]]:font-medium **:[[cmdk-group-heading]]:text-muted-foreground overflow-visible p-1",
                className
            )}
            {...props}
        />
    );
}

function CommandSeparator({
    className
}: React.ComponentProps<typeof CommandPrimitive.Separator>) {
    // Render as a purely presentational element that's hidden from accessibility tree
    // The cmdk-separator attribute ensures cmdk still recognizes it
    return (
        <div
            data-slot="command-separator"
            cmdk-separator=""
            aria-hidden="true"
            className={cn(
                "bg-border pointer-events-none -mx-1 h-px",
                className
            )}
        />
    );
}

function CommandItem({
    className,
    ...props
}: React.ComponentProps<typeof CommandPrimitive.Item>) {
    return (
        <CommandPrimitive.Item
            data-slot="command-item"
            className={cn(
                "outline-hidden data-[selected=true]:text-text-primary data-[selected=true]:bg-kumo-grey-50 [&_svg:not([class*='text-'])]:text-title-secondary relative flex min-w-fit cursor-default select-none items-center gap-2 whitespace-nowrap rounded-sm px-2 py-1.5 text-sm data-[disabled=true]:pointer-events-none data-[disabled=true]:opacity-50 [&_svg:not([class*='size-'])]:size-4 [&_svg]:pointer-events-none [&_svg]:shrink-0",
                className
            )}
            {...props}
        />
    );
}

function CommandShortcut({
    className,
    ...props
}: React.ComponentProps<"span">) {
    return (
        <span
            data-slot="command-shortcut"
            className={cn(
                "text-muted-foreground ml-auto text-xs tracking-widest",
                className
            )}
            {...props}
        />
    );
}

export {
    Command,
    CommandDialog,
    CommandInput,
    CommandList,
    CommandEmpty,
    CommandGroup,
    CommandItem,
    CommandShortcut,
    CommandSeparator
};
