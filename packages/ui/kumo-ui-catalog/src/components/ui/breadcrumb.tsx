// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { MoreHorizontal } from "lucide-react";

import { cn } from "@/lib/utils";

function Breadcrumb({ ...props }: React.ComponentProps<"nav">) {
    return <nav aria-label="breadcrumb" data-slot="breadcrumb" {...props} />;
}

function BreadcrumbList({ className, ...props }: React.ComponentProps<"ol">) {
    return (
        <ol
            data-slot="breadcrumb-list"
            className={cn(
                "flex flex-wrap items-center gap-1 break-words text-sm font-medium text-slate-500 sm:gap-1.5",
                className
            )}
            {...props}
        />
    );
}

function BreadcrumbItem({ className, ...props }: React.ComponentProps<"li">) {
    return (
        <li
            data-slot="breadcrumb-item"
            className={cn("inline-flex items-center gap-1", className)}
            {...props}
        />
    );
}

function BreadcrumbLink({
    asChild,
    className,
    ...props
}: React.ComponentProps<"a"> & {
    asChild?: boolean;
}) {
    const Comp = asChild ? Slot : "a";

    return (
        <Comp
            data-slot="breadcrumb-link"
            className={cn("transition-colors hover:text-black", className)}
            {...props}
        />
    );
}

function BreadcrumbPage({ className, ...props }: React.ComponentProps<"span">) {
    return (
        <span
            data-slot="breadcrumb-page"
            role="link"
            aria-disabled="true"
            aria-current="page"
            className={cn("text-black", className)}
            {...props}
        />
    );
}

function BreadcrumbSeparator({
    children,
    className,
    ...props
}: React.ComponentProps<"li">) {
    return (
        <li
            data-slot="breadcrumb-separator"
            role="presentation"
            aria-hidden="true"
            className={cn("[&>svg]:size-3.5", className)}
            {...props}>
            {children ?? <span>/</span>}
        </li>
    );
}

function BreadcrumbEllipsis({
    className,
    ...props
}: React.ComponentProps<"span">) {
    return (
        <span
            data-slot="breadcrumb-ellipsis"
            role="presentation"
            aria-hidden="true"
            className={cn("flex size-9 items-center justify-center", className)}
            {...props}>
            <MoreHorizontal className="size-4" />
            <span className="sr-only">More</span>
        </span>
    );
}

export {
    Breadcrumb,
    BreadcrumbList,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbPage,
    BreadcrumbSeparator,
    BreadcrumbEllipsis
};
