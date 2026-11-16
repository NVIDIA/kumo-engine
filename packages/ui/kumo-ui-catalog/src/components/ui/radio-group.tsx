// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";
import * as RadioGroupPrimitive from "@radix-ui/react-radio-group";

import { cn } from "@/lib/utils";

function RadioGroup({
    className,
    ...props
}: React.ComponentProps<typeof RadioGroupPrimitive.Root>) {
    return (
        <RadioGroupPrimitive.Root
            data-slot="radio-group"
            className={cn("grid gap-3", className)}
            {...props}
        />
    );
}

/**
 * KUI `.nv-radio-group-input`: a 16px circle with a 2px resting outline that
 * becomes a 5px brand-green ring around a black centre once selected. On a
 * 16px box a 5px border leaves exactly the 6px the indicator dot occupies.
 */
const radioInputClassName = cn(
    "border-kui-border-interaction-base size-4 shrink-0 cursor-pointer rounded-full border-2 bg-white outline-none transition-colors",
    "hover:border-kui-hover-border hover:bg-kui-bg-hover",
    "active:border-kui-hover-border active:bg-white",
    "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-current",
    "data-[state=checked]:border-brand-green data-[state=checked]:border-[5px] data-[state=checked]:bg-black",
    "data-[state=checked]:hover:border-brand-green-hover data-[state=checked]:hover:bg-black",
    "data-[state=checked]:active:border-brand-green-active",
    "disabled:border-kui-border-disabled-strong disabled:bg-kui-bg-disabled disabled:cursor-not-allowed",
    "disabled:hover:border-kui-border-disabled-strong disabled:hover:bg-kui-bg-disabled",
    "disabled:data-[state=checked]:bg-kui-border-disabled disabled:data-[state=checked]:border-transparent"
);

function RadioGroupItem({
    className,
    labelClassName = "",
    title,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    active: _active,
    id,
    disabled,
    ...props
}: React.ComponentProps<typeof RadioGroupPrimitive.Item> & {
    title?: React.ReactNode;
    labelClassName?: string;
    active?: boolean;
}) {
    return (
        // Label typography is deliberately left as-is — this change restyles the
        // control only.
        <div className="flex items-center gap-2 leading-5">
            <RadioGroupPrimitive.Item
                id={id}
                data-slot="radio-group-item"
                disabled={disabled}
                className={cn(radioInputClassName, className)}
                {...props}>
                <RadioGroupPrimitive.Indicator
                    data-slot="radio-group-indicator"
                    className="flex size-full items-center justify-center">
                    <span
                        className={cn(
                            "size-1.5 rounded-full bg-black transition-[opacity,scale] duration-150 ease-out",
                            disabled && "bg-kui-fg-disabled"
                        )}
                    />
                </RadioGroupPrimitive.Indicator>
            </RadioGroupPrimitive.Item>
            {title && (
                <label
                    htmlFor={id}
                    // Merged rather than concatenated: Tailwind resolves
                    // conflicts by stylesheet order, so a plain suffix could
                    // never override the defaults here and `labelClassName`
                    // silently did nothing for size, weight or colour.
                    className={cn(
                        "text-title-primary cursor-pointer text-base font-medium leading-5",
                        labelClassName
                    )}>
                    {title}
                </label>
            )}
        </div>
    );
}

export { RadioGroup, RadioGroupItem };
