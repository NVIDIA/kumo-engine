// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";

import { cn } from "@/lib/utils";

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
    return (
        <input
            type={type}
            data-slot="input"
            className={cn(
                "shadow-xs dark:bg-input/30 selection:text-title-primary file:text-title-secondary placeholder:text-title-secondary border-border-input flex h-8 w-full min-w-0 rounded-sm border bg-white px-3 py-1 text-sm font-normal outline-none transition-[color,box-shadow] selection:bg-slate-300 file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-90 md:text-sm",
                "focus-visible:border-slate-100 focus-visible:ring-[1px] focus-visible:ring-slate-500",
                "aria-invalid:border-destructive aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40",
                className
            )}
            {...props}
        />
    );
}

export { Input };
