// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { ReactNode } from "react";
import {
    Card,
    CardContent,
    CardFooter,
    CardHeader
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Skeleton } from "./skeleton";
import { CopyToClipBoard } from "./copy-to-clipboard";

interface QueryCardProps {
    id: string;
    headerClassName?: string;
    title?: string;
    titleIcon?: React.ReactNode;
    className?: string;
    content?: ReactNode | string;
    contentClassName?: string;
    footerClassName?: string;
    propertyList?: PropertyItem[];
    copyEnabled?: boolean;
}

type PropertyItem = {
    icon: React.ReactNode;
    label: string | React.ReactNode;
};

function QueryCard({
    id,
    className,
    title,
    titleIcon,
    content,
    headerClassName,
    footerClassName,
    contentClassName,
    propertyList,
    copyEnabled,
    ...props
}: QueryCardProps) {
    const isCopyAllowed = typeof content === "string";

    return (
        <>
            <Card
                id={id}
                className={cn(
                    "self-stretch overflow-hidden rounded-bl-none rounded-br-none rounded-tl rounded-tr py-2 shadow-none",
                    className
                )}
                {...props}>
                {title && (
                    <CardHeader data-slot="query-card-with-icon-header">
                        <div
                            className={cn(
                                "flex items-center",
                                headerClassName
                            )}>
                            {titleIcon}
                            <span className="mx-2 text-sm font-medium leading-tight text-slate-500">
                                {title}
                            </span>
                        </div>
                    </CardHeader>
                )}
                {
                    <CardContent
                        className={cn(
                            "relative whitespace-pre-line text-sm font-semibold leading-tight text-gray-800 shadow-none",
                            contentClassName
                        )}
                        data-slot="query-card-content">
                        {copyEnabled && content && isCopyAllowed && (
                            <CopyToClipBoard
                                copiedText={content}
                                iconSize={16}
                                tooltipText={"Copied"}
                                className="absolute"
                            />
                        )}

                        {!content ? (
                            <Skeleton className="w-200 h-10" />
                        ) : (
                            content
                        )}
                    </CardContent>
                }
            </Card>
            {propertyList && (
                <CardFooter
                    className={cn(
                        "rounded-xs border-1 flex-row border-slate-200 bg-white px-0 py-0 shadow-none",
                        footerClassName
                    )}>
                    {propertyList?.map((item, index) => (
                        <div
                            key={`query-card-${index}`}
                            className={cn(
                                "inline-flex items-center justify-start gap-2 px-2 py-1.5",
                                index !== 0 && "border-l border-slate-200"
                            )}>
                            {item.icon}
                            {item.label ? (
                                <div className="justify-start text-sm font-medium leading-tight text-slate-500">
                                    {item.label}
                                </div>
                            ) : (
                                <Skeleton className="h-3 w-12" />
                            )}
                        </div>
                    ))}
                </CardFooter>
            )}
        </>
    );
}

export { QueryCard };
