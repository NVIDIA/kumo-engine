// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-License-Identifier: Apache-2.0 AND MIT

import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import * as React from "react";

import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";

function TooltipProvider({
    delayDuration = 0,
    skipDelayDuration = 300,
    ...props
}: React.ComponentProps<typeof TooltipPrimitive.Provider>) {
    return (
        <TooltipPrimitive.Provider
            data-slot="tooltip-provider"
            delayDuration={delayDuration}
            skipDelayDuration={skipDelayDuration}
            {...props}
        />
    );
}

function Tooltip({
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    id,
    delayDuration = 100,
    skipDelayDuration = 0,
    showDuration,
    ...props
}: React.ComponentProps<typeof TooltipPrimitive.Root> & {
    id?: string;
    delayDuration?: number;
    skipDelayDuration?: number;
    showDuration?: number;
}) {
    const [open, setOpen] = React.useState(false);

    React.useEffect(() => {
        if (showDuration && showDuration > 0) {
            setOpen(true);
            const timeout = setTimeout(() => {
                setOpen(false);
            }, showDuration);

            return () => clearTimeout(timeout);
        }
    }, [showDuration]);

    return (
        <TooltipProvider
            delayDuration={delayDuration}
            skipDelayDuration={skipDelayDuration}>
            <TooltipPrimitive.Root
                open={showDuration ? open : undefined}
                data-slot="tooltip"
                {...props}
            />
        </TooltipProvider>
    );
}

function TooltipTrigger({
    className,
    ...props
}: React.ComponentProps<typeof TooltipPrimitive.Trigger>) {
    return (
        <TooltipPrimitive.Trigger
            asChild
            className={cn(
                "justify-start overflow-hidden border-none bg-transparent",
                className
            )}
            data-slot="tooltip-trigger"
            {...props}
        />
    );
}

function TooltipContent({
    className,
    sideOffset = 5,
    side = "bottom",
    align = "center",
    arrow = true,
    variant = "dark",
    children,
    ...props
}: React.ComponentProps<typeof TooltipPrimitive.Content> & {
    arrow?: boolean;
    variant?: "dark" | "light";
}) {
    const isDark = variant === "dark";
    const contentClassName = isDark ? `bg-(--tooltip-dark)` : `bg-white`;
    const divClassName = isDark
        ? "text-white"
        : "border rounded-sm border-slate-200 text-slate-800";
    const arrowClassName = isDark ? "fill-(--tooltip-dark)" : "fill-white";
    return (
        <TooltipPrimitive.Portal>
            {children && (
                <TooltipPrimitive.Content
                    data-slot="tooltip-content"
                    sideOffset={sideOffset}
                    side={side}
                    align={align}
                    className={cn(
                        "z-1051 flex flex-col items-center justify-start rounded-sm",
                        contentClassName,
                        className
                    )}
                    {...props}>
                    <div
                        className={
                            "wrap-break-word max-w-[90vw] overflow-auto whitespace-normal px-4 py-1 text-left text-sm font-medium leading-5 " +
                            divClassName
                        }>
                        {children}
                    </div>
                    {arrow && (
                        <TooltipPrimitive.Arrow
                            className={
                                "z-50 h-2 w-3 stroke-gray-200 stroke-[1px] " +
                                arrowClassName
                            }
                        />
                    )}
                </TooltipPrimitive.Content>
            )}
        </TooltipPrimitive.Portal>
    );
}

function TruncatedTooltip({
    children,
    tooltipContent,
    className,
    testId
}: {
    children: React.ReactNode;
    tooltipContent: React.ReactNode;
    className?: string;
    testId?: string;
}) {
    const textRef = useRef<HTMLDivElement>(null);
    const [isTruncated, setIsTruncated] = useState(false);

    useEffect(() => {
        const el = textRef.current;
        if (el) {
            setIsTruncated(el.scrollWidth > el.offsetWidth);
        }
    }, [tooltipContent]);

    const content = (
        <div ref={textRef} className={className} data-testid={testId}>
            {children}
        </div>
    );

    return isTruncated ? (
        <Tooltip>
            <TooltipTrigger>{content}</TooltipTrigger>
            <TooltipContent arrow={false} side="top" sideOffset={5}>
                {tooltipContent}
            </TooltipContent>
        </Tooltip>
    ) : (
        content
    );
}

export {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
    TruncatedTooltip
};
