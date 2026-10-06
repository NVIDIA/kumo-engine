// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const descriptionVariants = cva("w-full py-1 font-medium", {
    variants: {
        variant: {
            default: "text-title-primary",
            secondary: "text-title-secondary",
            success: "text-green-700",
            destructive: "text-destructive"
        },
        size: {
            xs: "text-xs justify-start leading-4",
            sm: "text-sm justify-start leading-4",
            md: "text-base justify-start leading-4",
            lg: "text-lg justify-start leading-4"
        }
    },
    defaultVariants: {
        variant: "default",
        size: "md"
    }
});

function Description({
    className,
    variant,
    size,
    ...props
}: React.ComponentProps<"p"> & VariantProps<typeof descriptionVariants>) {
    return (
        <p
            data-slot="description"
            className={cn(descriptionVariants({ variant, size }), className)}
            {...props}
        />
    );
}

export { Description };
