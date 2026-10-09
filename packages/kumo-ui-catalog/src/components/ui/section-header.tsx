// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { cva, VariantProps } from "class-variance-authority";
import { ReactNode } from "react";

const titleVariants = cva("!mt-0 !mb-0 !font-semibold", {
    variants: {
        titleVariant: {
            default: "!text-text-primary",
            secondary: "!text-text-secondary"
        },
        titleSize: {
            xs: "text-xs",
            sm: "text-sm",
            md: "text-base",
            lg: "text-lg",
            xl: "text-xl",
            xxl: "text-2xl"
        }
    },
    defaultVariants: {
        titleVariant: "default",
        titleSize: "xxl"
    }
});

const descriptionVariants = cva("flex items-center !mt-1 !mb-0 font-medium", {
    variants: {
        descriptionVariant: {
            default: "text-text-primary",
            secondary: "text-text-secondary"
        },
        descriptionSize: {
            xs: "text-xs ",
            sm: "text-sm ",
            md: "text-base",
            lg: "text-lg"
        }
    },
    defaultVariants: {
        descriptionVariant: "secondary",
        descriptionSize: "sm"
    }
});

interface SectionHeaderProps {
    title: string | ReactNode;
    titleVariant?: string;
    titleSize?: string;
    titleLetterSpacing?: number;
    subtitle?: string | ReactNode;
    description?: string;
    descriptionVariant?: string;
    descriptionSize?: string;
    className?: string;
    titleClassName?: string;
    subtitleClassName?: string;
    descriptionClassName?: string;
}

function SectionHeader({
    title,
    titleVariant,
    titleSize,
    titleLetterSpacing,
    subtitle,
    description,
    descriptionVariant,
    descriptionSize,
    className,
    titleClassName,
    subtitleClassName,
    descriptionClassName
}: SectionHeaderProps &
    VariantProps<typeof titleVariants> &
    VariantProps<typeof descriptionVariants>) {
    const Header =
        titleSize === "xs"
            ? "h6"
            : titleSize === "sm"
              ? "h5"
              : titleSize === "md"
                ? "h4"
                : titleSize === "lg"
                  ? "h3"
                  : titleSize === "xl"
                    ? "h2"
                    : "h1";

    return (
        <div className={cn("leading-tight", className)}>
            <div className="flex items-center justify-start leading-none">
                <Header
                    className={cn(
                        titleVariants({ titleVariant, titleSize }),
                        titleClassName
                    )}
                    style={{
                        ...(titleLetterSpacing != null
                            ? { letterSpacing: `${titleLetterSpacing}px` }
                            : {})
                    }}>
                    {title}
                </Header>

                {subtitle && (
                    <span
                        className={cn(
                            "ml-3 text-2xl font-normal text-slate-500",
                            subtitleClassName
                        )}>
                        {subtitle}
                    </span>
                )}
            </div>

            {description && (
                <p
                    className={cn(
                        descriptionVariants({
                            descriptionVariant,
                            descriptionSize
                        }),
                        descriptionClassName
                    )}>
                    {description}
                </p>
            )}
        </div>
    );
}

export { SectionHeader };
