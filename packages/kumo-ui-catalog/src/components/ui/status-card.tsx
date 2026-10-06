// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { cn, createScrollbarInjector } from "@/lib/utils";
import { ReactNode } from "react";

// Inject scrollbar styles once globally
const injectScrollbarStyles = createScrollbarInjector("status-card-scrollable");

// If content overflows max-height by less than this (~3 lines), expand instead of scrolling.
const OVERFLOW_THRESHOLD_PX = 72;

function expandIfSmallOverflow(el: HTMLDivElement | null) {
    if (!el) return;
    const overflow = el.scrollHeight - el.clientHeight;
    if (overflow > 0 && overflow <= OVERFLOW_THRESHOLD_PX) {
        el.style.maxHeight = "none";
        el.style.overflowY = "visible";
    }
}

interface StatusCardProps extends React.ComponentProps<"div"> {
    id: string;
    variant: string;
    title?: string;
    titleIcon?: ReactNode;
    enableDivider?: boolean;
    contentChild?: ReactNode;
    className?: string;
    headerClassName?: string;
    contentClassName?: string;
}

const statusCardVariants = {
    queued: {
        enableDivider: false,
        className: "bg-yellow-100 text-yellow-700 border-none ",
        headerClassName: "",
        contentClassName: "px-0"
    },
    done: {
        enableDivider: false,
        className: "bg-emerald-100 text-emerald-700 border-none ",
        headerClassName: "",
        contentClassName: "px-10"
    },
    failed: {
        enableDivider: false,
        className:
            "bg-red-50 text-title-primary fill-red-500 border-red-500 rounded-md",
        headerClassName: "font-semibold ",
        contentClassName: "px-10"
    },
    running: {
        enableDivider: false,
        className: "bg-violet-100 text-violet-700 border-none ",
        headerClassName: "",
        contentClassName: "px-0"
    },
    cancelled: {
        enableDivider: false,
        className: "bg-slate-100 text-slate-700 border-none ",
        headerClassName: "",
        contentClassName: "px-10"
    }
};

function StatusCard({
    id,
    variant,
    title,
    titleIcon,
    enableDivider,
    contentChild,
    className,
    headerClassName,
    contentClassName,
    ...props
}: StatusCardProps) {
    const variantDefaults =
        statusCardVariants[variant as keyof typeof statusCardVariants] || {};

    const mergedClassName = cn(
        "status-card-scrollable inline-flex w-full flex-col items-start justify-start gap-4 self-stretch overflow-y-auto max-h-[15rem] rounded px-6 py-4 shadow-none",
        variantDefaults.className,
        className
    );

    const mergedHeaderClassName = cn(
        "flex-inline px-0",
        variantDefaults.headerClassName,
        headerClassName
    );
    const mergedContentClassName = cn(
        "w-full break-words whitespace-pre-wrap [&_*]:break-words [&_*]:whitespace-pre-wrap",
        variantDefaults.contentClassName,
        contentClassName
    );
    const showDivider = enableDivider ?? variantDefaults.enableDivider;

    // Inject styles once on first render
    injectScrollbarStyles();

    return (
        <Card
            ref={expandIfSmallOverflow}
            id={id}
            className={mergedClassName}
            tabIndex={0}
            role="region"
            aria-label={title || "Status card"}
            {...props}>
            <CardHeader
                data-slot="status-card-icon-header"
                className={mergedHeaderClassName}>
                <div className="flex w-full items-start justify-between">
                    <div className="flex items-start gap-4">
                        <div>{titleIcon}</div>
                        <span className="pt-0.75 justify-start whitespace-pre-wrap text-base font-semibold leading-tight">
                            {title}
                        </span>
                    </div>
                </div>
            </CardHeader>

            {showDivider && (
                <div className="h-0 self-stretch border-[0.5px] border-black/10" />
            )}

            {contentChild && (
                <CardContent
                    className={mergedContentClassName}
                    data-slot="query-card-content">
                    {contentChild}
                </CardContent>
            )}
        </Card>
    );
}

export { StatusCard };
