// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const titleVariants = cva("", {
    variants: {
        variant: {
            "page-heading": "text-title-primary font-bold text-2xl leading-4",
            "primary-heading":
                "text-title-primary font-bold text-2xl leading-4",
            "secondary-heading":
                "text-title-primary font-bold text-base leading-4",
            "tertiary-heading":
                "text-title-primary font-bold text-base leading-4",
            "section-header":
                "text-title-secondary font-medium text-base leading-4",
            "label-header": "text-title-primary font-medium text-md leading-4"
        }
    },
    defaultVariants: {
        variant: "page-heading"
    }
});

export interface TitleProps
    extends React.HTMLAttributes<HTMLHeadingElement>,
        VariantProps<typeof titleVariants> {}

const Title = ({ className, variant, ...props }: TitleProps) => {
    const Header =
        variant === "label-header"
            ? "h6"
            : variant === "section-header"
              ? "h5"
              : variant === "tertiary-heading"
                ? "h4"
                : variant === "secondary-heading"
                  ? "h3"
                  : variant === "primary-heading"
                    ? "h2"
                    : "h1";

    return (
        <Header
            className={cn(titleVariants({ variant, className }))}
            {...props}
        />
    );
};
Title.displayName = "Title";

export { Title };
