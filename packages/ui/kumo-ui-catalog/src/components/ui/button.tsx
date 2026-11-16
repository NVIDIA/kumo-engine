// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
    "inline-flex items-center justify-center gap-2 whitespace-nowrap !rounded-sm font-normal cursor-pointer disabled:pointer-events-none disabled:opacity-50 outline-none focus-visible:border-ring focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive",
    {
        variants: {
            variant: {
                primary:
                    "bg-brand-green text-kui-fg-primary font-medium border border-brand-green hover:bg-brand-green-hover hover:border-brand-green-hover active:bg-brand-green-active active:border-brand-green-active [&_svg]:fill-kui-fg-primary",
                secondary:
                    // KUI brand-outline button (Figma 15937:97491 + state variants), light theme.
                    // Default: green border (color/brand/border #76b900 / --brand-green), black text
                    //   (primary/foreground #000000), near-black icon (inverse/brand/foreground #161616).
                    // Hover: +5% black bg (interaction/hover/background #0000000d).
                    // Pressed: +10% black bg (interaction/pressed/background #0000001a).
                    // Disabled: grey border (#0000001a) + #898989 text/icon (disabled/foreground).
                    // Geometry / size / font weight unchanged.
                    "bg-transparent border border-brand-green text-kui-fg-primary font-medium [&_svg]:fill-kui-fg-on-brand hover:bg-kui-bg-hover active:bg-kui-bg-pressed disabled:opacity-100 disabled:border-kui-border-disabled disabled:text-kui-fg-disabled disabled:[&_svg]:fill-kui-fg-disabled",
                outline:
                    "bg-transparent border border-border-primary text-title-primary font-medium hover:bg-kui-bg-hover [&_svg]:fill-title-secondary",
                "light-outline":
                    "bg-transparent border border-border-secondary text-title-secondary font-medium hover:bg-kui-bg-hover [&_svg]:fill-title-secondary",
                destructive:
                    "bg-transparent border border-title-red text-title-red font-medium hover:bg-red-50 [&_svg]:fill-title-red",
                "no-outline":
                    "bg-transparent text-title-primary font-medium hover:bg-kui-bg-hover [&_svg]:fill-title-secondary",
                link: "group text-text-secondary bg-transparent border-none underline-offset-4 hover:text-slate-800 [&_svg]:fill-title-secondary [&_svg]:group-hover:fill-slate-800 hover:bg-slate-100",
                underline:
                    "group text-text-secondary bg-slate-100 border-none underline underline-offset-4 hover:text-slate-800 [&_svg]:fill-title-secondary [&_svg]:group-hover:fill-slate-800 hover:bg-slate-100"
            },
            size: {
                default: "h-9 px-4 py-2 text-base leading-5 [&_svg]:size-4",
                xs: "h-7 px-2 text-xs py-1 [&_svg]:size-3",
                sm: "h-8 px-3 text-sm py-1 [&_svg]:size-4",
                md: "h-8 px-3 text-base py-1 [&_svg]:size-4",
                lg: "h-12 px-6 text-lg py-3 [&_svg]:size-6"
            }
        },
        defaultVariants: {
            variant: "primary",
            size: "default"
        }
    }
);

// eslint-disable-next-line react-refresh/only-export-components
export { buttonVariants };

function Button({
    className,
    variant,
    size,
    asChild = false,
    ...props
}: React.ComponentProps<"button"> &
    VariantProps<typeof buttonVariants> & {
        asChild?: boolean;
    }) {
    const Comp = asChild ? Slot : "button";

    return (
        <Comp
            data-slot="button"
            className={cn(buttonVariants({ variant, size, className }))}
            {...props}
        />
    );
}

export { Button };
