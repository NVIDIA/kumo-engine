// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as React from "react";

import { cn } from "@/lib/utils";
import { StickyColumnOptions } from "@/types/data-table";

const TableContext = React.createContext<{ maxHeight?: number }>({});

function Table({
    className,
    dref,
    containerLabel,
    maxHeight, //maximum height
    ...props
}: React.ComponentProps<"table"> & {
    dref?: React.RefObject<HTMLDivElement>;
    containerLabel?: string;
    maxHeight?: number;
}) {
    return (
        <TableContext.Provider value={{ maxHeight }}>
            <div
                data-slot="table-container"
                tabIndex={0}
                role="region"
                aria-label={containerLabel || "Scrollable table"}
                className="relative w-full overflow-auto scroll-smooth [-ms-overflow-style:auto] [&::-webkit-scrollbar-thumb]:rounded-sm [&::-webkit-scrollbar-thumb]:bg-[var(--scrollbar)] [&::-webkit-scrollbar-track]:bg-[var(--secondary)] [&::-webkit-scrollbar]:h-3 [&::-webkit-scrollbar]:w-3 [&::-webkit-scrollbar]:opacity-100 [&::-webkit-scrollbar]:transition-opacity [&::-webkit-scrollbar]:duration-1000 [&::-webkit-scrollbar]:ease-in-out"
                style={maxHeight ? { maxHeight } : undefined}
                ref={dref}>
                <table
                    data-slot="table"
                    className={cn("w-full caption-bottom text-base", className)}
                    {...props}
                />
            </div>
        </TableContext.Provider>
    );
}

function TableHeader({ className, ...props }: React.ComponentProps<"thead">) {
    return (
        <thead
            data-slot="table-header"
            data-testid={`table-header`}
            className={cn("bg-surface-secondary [&_tr]:border-b", className)}
            {...props}
        />
    );
}

function TableBody({ className, ...props }: React.ComponentProps<"tbody">) {
    return (
        <tbody
            data-slot="table-body"
            className={cn("[&_tr:last-child]:border-0", className)}
            {...props}
        />
    );
}

function TableFooter({ className, ...props }: React.ComponentProps<"tfoot">) {
    return (
        <tfoot
            data-slot="table-footer"
            className={cn(
                "bg-surface-secondary border-t font-medium [&>tr]:last:border-b-0",
                className
            )}
            {...props}
        />
    );
}

function TableRow({
    className,
    disableRowSelectionHover = false,
    ...props
}: React.ComponentProps<"tr"> & { disableRowSelectionHover?: boolean }) {
    return (
        <tr
            data-slot="table-row"
            className={cn(
                "border-border-secondary group !border-b transition-colors data-[state=selected]:bg-white",
                !disableRowSelectionHover && "hover:bg-kumo-grey-50", // same as surface-grey, surface-secondary
                className
            )}
            {...props}
        />
    );
}

function TableHead({
    className,
    width = 0,
    sticky = null,
    scrollLeft = 0,
    ...props
}: React.ComponentProps<"th"> & {
    width?: number | null;
} & {
    sticky?: StickyColumnOptions | null;
} & {
    scrollLeft?: number;
}) {
    const stickyStyle: React.CSSProperties = {};
    const isVisible =
        sticky?.type === "on-scroll" && scrollLeft > 10 ? true : false;

    if (sticky?.left !== undefined) {
        stickyStyle.position = "sticky";
        if (typeof sticky.left === "boolean") {
            stickyStyle.left = sticky.left ? 0 : undefined;
        } else if (typeof sticky.left === "number") {
            stickyStyle.left = `${sticky.left}px`;
        } else {
            stickyStyle.left = sticky.left;
        }
        stickyStyle.boxShadow = isVisible
            ? "-2px 0px 3px -3px grey inset"
            : "none";
    }

    if (sticky?.right !== undefined) {
        stickyStyle.position = "sticky";
        if (typeof sticky.right === "boolean") {
            stickyStyle.right = sticky.right ? 0 : undefined;
        } else if (typeof sticky.right === "number") {
            stickyStyle.right = `${sticky.right}px`;
        } else {
            stickyStyle.right = sticky.right;
        }
        stickyStyle.boxShadow = isVisible
            ? "2px 0px 3px -3px grey inset"
            : "none";
    }

    if (sticky?.zIndex !== undefined) {
        stickyStyle.zIndex = sticky.zIndex;
    }

    if (width) {
        stickyStyle.width = `${width}px`;
        stickyStyle.minWidth = `${width}px`;
    }

    return (
        <th
            data-slot="table-head"
            className={cn(
                `bg-surface-secondary text-title-primary h-10 whitespace-nowrap px-3 py-1.5 text-left align-middle text-base font-medium leading-5 [&:has([role=checkbox])]:pr-0 [&>[role=checkbox]]:translate-y-[2px]`,
                className
            )}
            style={stickyStyle}
            {...props}
        />
    );
}

function TableCell({
    className,
    width = 0,
    sticky = null,
    scrollLeft = 0,
    disableRowSelectionHover = false,
    ...props
}: React.ComponentProps<"td"> & {
    width?: number | null;
} & {
    sticky?: StickyColumnOptions | null;
} & {
    scrollLeft?: number;
} & {
    disableRowSelectionHover?: boolean;
}) {
    const stickyStyle: React.CSSProperties = {};
    const isVisible =
        sticky?.type === "on-scroll" && scrollLeft > 10 ? true : false;

    if (sticky?.left !== undefined) {
        stickyStyle.position = "sticky";
        if (typeof sticky.left === "boolean") {
            stickyStyle.left = sticky.left ? 0 : undefined;
        } else if (typeof sticky.left === "number") {
            stickyStyle.left = `${sticky.left}px`;
        } else {
            stickyStyle.left = sticky.left;
        }
        stickyStyle.boxShadow = isVisible
            ? "-2px 0px 3px -3px grey inset"
            : "none";
    }

    if (sticky?.right !== undefined) {
        stickyStyle.position = "sticky";
        if (typeof sticky.right === "boolean") {
            stickyStyle.right = sticky.right ? 0 : undefined;
        } else if (typeof sticky.right === "number") {
            stickyStyle.right = `${sticky.right}px`;
        } else {
            stickyStyle.right = sticky.right;
        }
        stickyStyle.boxShadow = isVisible
            ? "2px 0px 3px -3px grey inset"
            : "none";
    }

    if (sticky?.zIndex !== undefined) {
        stickyStyle.zIndex = sticky.zIndex;
    }

    if (width) {
        stickyStyle.width = `${width}px`;
        stickyStyle.minWidth = `${width}px`;
    }

    return (
        <td
            data-slot="table-cell"
            className={cn(
                `text-title-secondary h-[44px] whitespace-nowrap bg-white px-3 py-1.5 align-middle text-base font-medium leading-5 group-data-[state=selected]:bg-white [&:has([role=checkbox])]:pr-0 [&>[role=checkbox]]:translate-y-[2px]`,
                !disableRowSelectionHover && "group-hover:bg-kumo-grey-50", // same as surface-grey, surface-secondary
                className
            )}
            style={stickyStyle}
            {...props}
        />
    );
}

function TableCaption({
    className,
    ...props
}: React.ComponentProps<"caption">) {
    return (
        <caption
            data-slot="table-caption"
            className={cn("text-muted-foreground mt-4 text-sm", className)}
            {...props}
        />
    );
}

export {
    Table,
    TableHeader,
    TableBody,
    TableFooter,
    TableHead,
    TableRow,
    TableCell,
    TableCaption
};
