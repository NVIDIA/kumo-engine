// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { OutputCardContent } from "@/types/output-card";
import React from "react";
import { Card, CardContent, CardHeader } from "./card";

interface OutputCardProps {
    id: string;

    className?: string;
    headerClassName?: string;
    headerLeft: (string | React.ReactNode)[];
    headerRight: (string | React.ReactNode)[];

    contentClassName?: string;
    contentItems: OutputCardContent[];
}

function OutputCard({
    id,
    className,

    headerClassName,
    headerLeft,
    headerRight,

    contentClassName,
    contentItems,

    ...props
}: OutputCardProps): JSX.Element {
    return (
        <>
            <Card
                id={id}
                className={cn(
                    "w-full rounded bg-white shadow-none outline-slate-200",
                    className
                )}
                {...props}>
                {(headerLeft?.length > 0 || headerRight?.length > 0) && (
                    <CardHeader
                        data-slot="output-card-header"
                        className={cn(
                            "text-text-secondary [&_svg]:fill-title-secondary",
                            headerClassName
                        )}>
                        <div
                            className={cn(
                                "flex items-center gap-2 text-xs font-bold"
                            )}>
                            {headerLeft?.length > 0 &&
                                headerLeft?.map((item, index) => (
                                    <span
                                        key={`output-card-header-left-${index}`}
                                        className="gap-2 leading-tight">
                                        {item}
                                    </span>
                                ))}
                            {headerRight?.length > 0 && (
                                <span className="ml-auto flex gap-2">
                                    {headerRight?.map((item, index) => (
                                        <span
                                            key={`output-card-header-right-${index}`}>
                                            {item}
                                        </span>
                                    ))}
                                </span>
                            )}
                        </div>
                    </CardHeader>
                )}
                {contentItems?.length && (
                    <CardContent
                        className={cn("text-sm", contentClassName)}
                        data-slot="output-card-content">
                        {contentItems?.map((item, index) => (
                            <div
                                key={`output-card-contents-${index}`}
                                className="flex items-center gap-3">
                                {item?.icon ?? <></>}
                                {item?.text ?? <></>}
                            </div>
                        ))}
                    </CardContent>
                )}
            </Card>
        </>
    );
}

export { OutputCard };
