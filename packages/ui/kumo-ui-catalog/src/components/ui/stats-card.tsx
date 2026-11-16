// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Download } from "@/icons/MaterialIcons/Download";
import { Downloading } from "@/icons/MaterialIcons/Downloading";
import { HomeStorage } from "@/icons/MaterialIcons/HomeStorage";
import { SplitScene } from "@/icons/MaterialIcons/SplitScene";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader } from "./card";
import { Symbol } from "./symbol";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";

interface StatsCardProps {
    id: string;
    clickable: boolean;
    isActive?: boolean;
    className?: string;
    headerClassName?: string;
    title: string;
    enableDownload: boolean;
    downloading?: boolean;
    onClick?: () => void;
    onDownloadClick?: () => void;
    percent?: string;
    contentClassName?: string;
    numTrainRows: number | string | undefined;
    dateRange: string | undefined;
}

function StatsCard({
    id,
    clickable,
    isActive = false,
    className,
    headerClassName,
    title,
    enableDownload,
    downloading = false,
    onClick = undefined,
    onDownloadClick = undefined,
    percent = undefined,
    contentClassName,
    numTrainRows,
    dateRange,

    ...props
}: StatsCardProps) {
    return (
        <Card
            id={id}
            className={cn(
                clickable ? "cursor-pointer" : "",
                isActive
                    ? "border-primary border bg-pink-50"
                    : "bg-white outline-slate-200",
                "w-full rounded shadow-none",
                className
            )}
            onClick={clickable && onClick ? onClick : undefined}
            {...props}>
            {title && (
                <CardHeader data-slot="stats-card-header">
                    <div
                        className={cn(
                            isActive
                                ? "text-slate-800 [&_svg]:fill-slate-800"
                                : "text-text-secondary [&_svg]:fill-title-secondary",
                            "flex items-center gap-2",
                            headerClassName
                        )}>
                        <span className="text-medium font-bold leading-tight">
                            {title}
                        </span>
                        {enableDownload && (
                            <div
                                className="m-0 inline-flex items-center p-0 leading-none"
                                onClick={onDownloadClick}>
                                {downloading ? (
                                    <Tooltip
                                        showDuration={3000}
                                        delayDuration={0}>
                                        <TooltipTrigger>
                                            <div>
                                                <Downloading />
                                            </div>
                                        </TooltipTrigger>
                                        <TooltipContent
                                            arrow={false}
                                            side={"bottom"}
                                            sideOffset={10}>
                                            {"Downloading ..."}
                                        </TooltipContent>
                                    </Tooltip>
                                ) : (
                                    <Download className="cursor-pointer" />
                                )}
                            </div>
                        )}
                        {percent && (
                            <span className="ml-auto">
                                <>{percent}%</>
                            </span>
                        )}
                    </div>
                </CardHeader>
            )}
            {
                <CardContent
                    className={cn(
                        "space-y-2 text-base text-slate-800 [&_svg]:fill-slate-800",
                        contentClassName
                    )}
                    data-slot="stats-card-content">
                    <div className="flex items-center space-x-4">
                        <HomeStorage className="size-4.5" />
                        {numTrainRows ? (
                            <span>{numTrainRows}</span>
                        ) : (
                            <Symbol type="emdash" />
                        )}
                    </div>
                    <div className="flex items-center space-x-4">
                        <SplitScene className="size-4.5" />
                        {dateRange ? (
                            <span>{dateRange}</span>
                        ) : (
                            <Symbol type="emdash" />
                        )}
                    </div>
                </CardContent>
            }
        </Card>
    );
}

export { StatsCard };
