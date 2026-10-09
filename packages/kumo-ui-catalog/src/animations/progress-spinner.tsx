// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";

import { cn } from "@/lib/utils";

function ProgressSpinner({
    className,
    innerClassName,
    ...props
}: React.ComponentProps<"div"> & {
    innerClassName?: string;
}) {
    return (
        <div
            data-slot="progress-spinner"
            role="status"
            aria-label="Loading"
            className={cn(
                // Consistent with <Spinner/>: NVIDIA Kaizen green circular spinner
                // (pale-green track + brand-green arc, via the brand-green tokens).
                "border-brand-green-subtle border-t-brand-green h-5 w-5 animate-spin rounded-full border-2",
                className,
                innerClassName
            )}
            {...props}
        />
    );
}

export { ProgressSpinner };
