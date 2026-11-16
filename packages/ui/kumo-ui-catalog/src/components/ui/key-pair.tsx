// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { cn } from "@/lib/utils";
import { CopyToClipBoard } from "@/components/ui/copy-to-clipboard";
import { Skeleton } from "./skeleton";
import { Label } from "./label";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";

interface KeyPairProps {
    keyLabel: string | React.ReactNode;
    value: string;
    full?: boolean;
    loadingStatus?: boolean;
    className?: string;
    keyClassName?: string;
    valueClassName?: string;
    paddingY?: number;
    showCopy?: boolean;
}

export const KeyPair: React.FC<KeyPairProps> = ({
    keyLabel,
    value,
    full = false,
    loadingStatus = false,
    className = "",
    keyClassName = "",
    valueClassName = "",
    paddingY = 2,
    showCopy = false
}) => {
    const isTrimmed = !full && value.length > 50;

    const displayValue =
        full || value.length <= 50 ? value : value.slice(0, 50) + "...";

    return (
        <div
            className={cn(
                "flex items-center gap-2",
                className,
                `py-${paddingY}`
            )}>
            {keyLabel && (
                <Label
                    className={cn(
                        "mb-0 whitespace-pre font-bold",
                        keyClassName
                    )}>
                    {loadingStatus ? (
                        <Skeleton className="h-5 w-24" />
                    ) : (
                        `${keyLabel} : `
                    )}
                </Label>
            )}
            <Label className={cn("mb-0", valueClassName)}>
                {loadingStatus ? (
                    <Skeleton className="h-5 w-36" />
                ) : isTrimmed ? (
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <span className="cursor-pointer">
                                {displayValue}
                            </span>
                        </TooltipTrigger>
                        <TooltipContent className="z-1200 max-w-300 whitespace-normal break-words">
                            {value}
                        </TooltipContent>
                    </Tooltip>
                ) : (
                    displayValue
                )}
            </Label>
            {!loadingStatus && showCopy && (
                <CopyToClipBoard
                    copiedText={value}
                    tooltipText={`Copied ${keyLabel} to clipboard`}
                    size="default"
                    variant="outline"
                    className="border-1 h-7 border-gray-500 p-1"
                />
            )}
        </div>
    );
};
