// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
    "text-xs font-medium px-2 rounded inline-flex justify-center items-center gap-0.5 overflow-hidden leading-5",
    {
        variants: {
            variant: {
                default: "bg-slate-300 text-slate-800",
                done: "bg-emerald-100 text-emerald-700",
                queued: "bg-yellow-100 text-yellow-700",
                running: "bg-violet-100 text-violet-700",
                cancelled: "bg-slate-100 text-slate-700",
                failed: "bg-red-100 text-red-700",
                pink: "bg-pink-200 text-pink-600"
            },
            size: {
                xs: "text-xs",
                sm: "text-sm",
                md: "text-base",
                lg: "text-lg"
            }
        },
        defaultVariants: {
            variant: "default",
            size: "md"
        }
    }
);

function Badge({
    className,
    variant,
    size,
    asChild = false,
    ...props
}: React.ComponentProps<"span"> &
    VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
    const Comp = asChild ? Slot : "span";

    return (
        <Comp
            data-slot="badge"
            className={cn(badgeVariants({ variant, size }), className)}
            {...props}
        />
    );
}

export { Badge };
