// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";
import { MoreHorizontalIcon } from "lucide-react";

import { cn } from "@/lib/utils";
import { ArrowLeft } from "@/icons/MaterialIcons/ArrowLeft";
import { ArrowRight } from "@/icons/MaterialIcons/ArrowRight";

function Pagination({ className, ...props }: React.ComponentProps<"nav">) {
    return (
        <nav
            role="navigation"
            aria-label="pagination"
            data-slot="pagination"
            className={cn("mx-auto flex w-full justify-center", className)}
            {...props}
        />
    );
}

function PaginationContent({
    className,
    ...props
}: React.ComponentProps<"ul">) {
    return (
        <ul
            data-slot="pagination-content"
            className={cn(
                "flex flex-row items-center gap-1 px-3 py-1.5",
                className
            )}
            {...props}
        />
    );
}

function PaginationItem({ ...props }: React.ComponentProps<"li">) {
    return <li data-slot="pagination-item" {...props} />;
}

type PaginationLinkProps = {
    isActive?: boolean;
} & React.ComponentProps<"a">;

function PaginationLink({
    className,
    isActive,
    ...props
}: PaginationLinkProps) {
    return (
        <a
            role="button"
            aria-current={isActive ? "page" : undefined}
            data-slot="pagination-link"
            data-active={isActive}
            className={cn(
                "!text-slate-800",
                isActive ? "px-4" : "px-4",
                className
            )}
            {...props}
        />
    );
}

function PaginationPrevious({
    className,
    ...props
}: React.ComponentProps<typeof PaginationLink>) {
    return (
        <PaginationLink
            aria-label="Go to previous page"
            className={cn("gap-1 px-0 sm:pl-0", className)}
            {...props}>
            <ArrowLeft />
        </PaginationLink>
    );
}

function PaginationNext({
    className,
    ...props
}: React.ComponentProps<typeof PaginationLink>) {
    return (
        <PaginationLink
            aria-label="Go to next page"
            className={cn("gap-1 px-0 sm:pr-0", className)}
            {...props}>
            <ArrowRight />
        </PaginationLink>
    );
}

function PaginationEllipsis({
    className,
    ...props
}: React.ComponentProps<"span">) {
    return (
        <span
            aria-hidden
            data-slot="pagination-ellipsis"
            className={cn("flex size-4 items-center justify-center", className)}
            {...props}>
            <MoreHorizontalIcon className="size-4" />
        </span>
    );
}

export {
    Pagination,
    PaginationContent,
    PaginationLink,
    PaginationItem,
    PaginationPrevious,
    PaginationNext,
    PaginationEllipsis
};
