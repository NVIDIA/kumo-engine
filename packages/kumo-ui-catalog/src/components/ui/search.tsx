// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from "react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { Search } from "@/icons/MaterialIcons/Search";

export interface SearchProps
    extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "ref"> {
    variant?: "default" | "secondary";
    position?: "default" | "top";
    inputClassName?: string;
}

const SearchBar: React.FC<SearchProps> = ({
    className,
    inputClassName,
    variant = "default",
    position = "default",
    ...props
}) => {
    const [isFocused, setIsFocused] = useState(false);

    const variantStyles = {
        default: {
            container: "border-border-input bg-white hover:bg-slate-100",
            icon: "fill-title-secondary",
            input: "text-title-primary placeholder:text-title-secondary"
        },
        secondary: {
            container:
                "border-border-secondary bg-transparent hover:bg-kumo-grey-50",
            icon: "fill-title-secondary",
            input: "text-title-primary placeholder:text-title-secondary"
        }
    };

    const styles = variantStyles[variant];

    const handleFocus = () => {
        setIsFocused(true);
    };

    const handleBlur = () => {
        setIsFocused(false);
    };

    return (
        <div
            className={cn(
                "flex h-8 w-full items-center rounded-sm border px-3 py-2",
                styles.container,
                position === "top" && "rounded-b-none",
                isFocused &&
                    (variant === "default" ? "bg-slate-200" : "bg-white"),
                className
            )}>
            <Search className={cn("mr-2 h-5 w-5 flex-shrink-0", styles.icon)} />
            <Input
                className={cn(
                    "border-0 bg-transparent p-0 !text-base !font-normal shadow-none focus-visible:ring-0 focus-visible:ring-offset-0",
                    styles.input,
                    inputClassName
                )}
                aria-label={
                    props["aria-label"] || props.placeholder || "Search"
                }
                onFocus={handleFocus}
                onBlur={handleBlur}
                {...props}
            />
        </div>
    );
};

export { SearchBar };
