"use client";
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";
import * as ToggleGroupPrimitive from "@radix-ui/react-toggle-group";
import * as TogglePrimitive from "@radix-ui/react-toggle";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const toggleVariants = cva(
    "inline-flex items-center justify-start gap-2 rounded-sm text-sm font-medium cursor-pointer hover:bg-transparent hover:text-transparent disabled:pointer-events-none disabled:opacity-50 data-[state=on]:bg-transaparent data-[state=on]:text-transaparent [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 [&_svg]:shrink-0 focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] outline-none transition-[color,box-shadow] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive whitespace-nowrap",
    {
        variants: {
            variant: {
                default: "bg-transparent",
                outline:
                    "border border-slate-200 bg-transparent shadow-xs hover:bg-transparent hover:text-accent-transparent"
            },
            size: {
                default: "h-8 px-3 min-w-8",
                sm: "h-10 px-3 min-w-10",
                lg: "h-12 px-3 min-w-12"
            }
        },
        defaultVariants: {
            variant: "default",
            size: "default"
        }
    }
);

function Toggle({
    className,
    variant,
    size,
    ...props
}: React.ComponentProps<typeof TogglePrimitive.Root> &
    VariantProps<typeof toggleVariants>) {
    return (
        <TogglePrimitive.Root
            data-slot="toggle"
            className={cn(toggleVariants({ variant, size, className }))}
            {...props}
        />
    );
}

const ToggleGroupContext = React.createContext<
    VariantProps<typeof toggleVariants>
>({
    size: "default",
    variant: "default"
});

function ToggleGroup({
    className,
    variant,
    size,
    children,
    ...props
}: React.ComponentProps<typeof ToggleGroupPrimitive.Root> &
    VariantProps<typeof toggleVariants>) {
    return (
        <ToggleGroupPrimitive.Root
            data-slot="toggle-group"
            data-variant={variant}
            data-size={size}
            className={cn(
                "group/toggle-group flex w-full items-center gap-1 rounded-sm",
                className
            )}
            {...props}>
            <ToggleGroupContext.Provider value={{ variant, size }}>
                {children}
            </ToggleGroupContext.Provider>
        </ToggleGroupPrimitive.Root>
    );
}

function ToggleGroupItem({
    className,
    children,
    variant,
    size,
    isFirst = false,
    isLast = false,
    ...props
}: React.ComponentProps<typeof ToggleGroupPrimitive.Item> &
    VariantProps<typeof toggleVariants> & {
        isFirst?: boolean;
        isLast?: boolean;
    }) {
    const context = React.useContext(ToggleGroupContext);
    const roundingClasses = cn("rounded-none", {
        "!rounded-tl-sm !rounded-bl-sm": isFirst,
        "!rounded-tr-sm !rounded-br-sm": isLast
    });

    return (
        <ToggleGroupPrimitive.Item
            data-slot="toggle-group-item"
            data-variant={context.variant || variant}
            data-size={context.size || size}
            className={cn(
                toggleVariants({
                    variant: context.variant || variant,
                    size: context.size || size
                }),
                "flex-1 shadow-none focus:z-10 focus-visible:z-10",
                roundingClasses,
                className
            )}
            {...props}>
            {children}
        </ToggleGroupPrimitive.Item>
    );
}
export { Toggle, ToggleGroup, ToggleGroupItem };
