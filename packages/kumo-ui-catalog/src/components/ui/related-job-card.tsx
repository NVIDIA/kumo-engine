// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card } from "@/components/ui/card";
import { ChevronDown, ChevronRight, DeveloperBoard } from "@/icons";
import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface RelatedJobCardProps extends React.ComponentProps<"div"> {
    title: string;
    id: string;
    badge: ReactNode;
    onClick?: () => void;
    expanded: boolean;
    children: ReactNode;
}

export function RelatedJobCard({
    className,
    title,
    id,
    badge,
    onClick,
    expanded,
    children,
    ...props
}: RelatedJobCardProps) {
    return (
        <>
            <Card
                onClick={onClick}
                className={cn(
                    "inline-flex w-full cursor-pointer items-center justify-between self-stretch overflow-hidden rounded-sm bg-white px-2 py-2 shadow-none outline-1 outline-offset-[-1px] outline-slate-200",
                    className
                )}
                {...props}>
                <div className="flex items-center justify-items-start gap-1 font-sans">
                    <DeveloperBoard className="size-4" />
                    <span className="text-text-primary text-xs font-bold leading-none">
                        {title}
                    </span>
                    <span className="text-text-secondary font-sans text-sm font-medium leading-tight">
                        {id}
                    </span>
                </div>
                <div className="flex items-center justify-items-end gap-1">
                    {badge}
                    <div
                        className="cursor-pointer pt-1"
                        onClick={(e) => e.stopPropagation()}>
                        {expanded ? (
                            <ChevronDown className="size-4" />
                        ) : (
                            <ChevronRight className="size-4" />
                        )}
                    </div>
                </div>
            </Card>
            {expanded && (
                <div className="rounded-none px-2 py-2">{children}</div>
            )}
        </>
    );
}

export default RelatedJobCard;
