// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";

import { cn, createScrollbarInjector } from "@/lib/utils";

interface TextAreaProps extends React.ComponentProps<"textarea"> {
    clickToEdit?: boolean;
}

const injectScrollbarStyles = createScrollbarInjector("textarea-scrollable");
function TextArea({ className, clickToEdit, ...props }: TextAreaProps) {
    const [isFocused, setIsFocused] = React.useState(false);

    // Inject scrollbar styles once on first render
    injectScrollbarStyles();
    if (!clickToEdit) {
        return (
            <textarea
                data-slot="textarea"
                className={cn(
                    "textarea-scrollable border-border-input shadow-xs aria-invalid:border-destructive aria-invalid:ring-destructive/20 dark:bg-input/30 dark:aria-invalid:ring-destructive/40 placeholder:text-text-secondary flex min-h-16 w-full !rounded-sm border bg-transparent px-3 py-1 text-sm outline-none transition-[color,box-shadow] focus-visible:border-slate-100 focus-visible:ring-[1px] focus-visible:ring-slate-500 disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
                    className
                )}
                {...props}
            />
        );
    }

    return (
        <div
            className={cn(
                "border-border-secondary shadow-xs w-full !rounded-sm border transition-[color,box-shadow]",
                isFocused && "ring-[1px] ring-slate-500"
            )}>
            <textarea
                data-slot="textarea"
                className={cn(
                    "textarea-scrollable aria-invalid:border-destructive dark:bg-input/30 placeholder:text-text-secondary flex min-h-16 w-full border-0 bg-transparent px-3 py-1 text-sm outline-none focus-visible:ring-0 disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
                    className
                )}
                onFocus={(e) => {
                    setIsFocused(true);
                    props.onFocus?.(e);
                }}
                onBlur={(e) => {
                    setIsFocused(false);
                    props.onBlur?.(e);
                }}
                {...props}
            />
            <div
                className={cn(
                    "border-t px-4 py-2",
                    isFocused ? "border-slate-500" : "border-border-secondary"
                )}>
                <span className="text-title-secondary text-base font-medium leading-5">
                    {isFocused ? "Editing" : "Click to edit"}
                </span>
            </div>
        </div>
    );
}

export { TextArea };
