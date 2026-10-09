// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import React from "react";
import { Skeleton } from "./skeleton";

interface EmptyTableProps extends React.ComponentProps<"div"> {
    id: string;
    className?: string;
    contentClassName?: string;
    label?: string;
    showLoading?: boolean;
    enableCard?: boolean;
    children?: React.ReactNode;
}

function EmptyTable({
    id,
    className,
    contentClassName,
    label = "",
    showLoading = false,
    enableCard = true,
    children,
    ...props
}: EmptyTableProps) {
    const Wrapper = enableCard ? Card : React.Fragment;

    return (
        <Wrapper
            {...(enableCard && {
                id,
                className: cn(
                    "items-center rounded border-1 border-slate-200 bg-white shadow-none",
                    className
                ),
                ...props
            })}>
            <CardContent
                className={cn(
                    "flex flex-col items-center gap-2 py-4 text-center",
                    contentClassName
                )}>
                {showLoading ? (
                    <Skeleton className="h-3 w-14 rounded-none bg-neutral-200" />
                ) : (
                    <>
                        {label && (
                            <div className="flex justify-center">
                                <div className="text-title-secondary text-sm md:w-auto">
                                    {label}
                                </div>
                            </div>
                        )}
                        {children}
                    </>
                )}
            </CardContent>
        </Wrapper>
    );
}

export { EmptyTable };
