// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CheckRight, ContentCopy } from "@/icons";
import { cn } from "@/lib/utils";
import React, { useState } from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Button, buttonVariants } from "./button";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";
import { type VariantProps } from "class-variance-authority";

export interface CopyToClipBoardProps {
    copiedText: string;
    tooltipText?: string;
    size?: "xs" | "sm" | "default" | "lg";
    variant?: VariantProps<typeof buttonVariants>["variant"];
    className?: string;
    iconSize?: number;
    title?: string;
}

export const CopyToClipBoard: React.FC<CopyToClipBoardProps> = ({
    copiedText,
    tooltipText = "Copied",
    size = "sm",
    variant = "outline",
    className = "",
    iconSize = 16,
    title
}) => {
    const [isCopied, setIsCopied] = useState(false);

    const handleCopy = () => {
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
    };

    const copyIcon = isCopied ? (
        <Tooltip delayDuration={100} showDuration={1000}>
            <TooltipTrigger>
                <span>
                    <CheckRight className={`!size-[${iconSize}px]`} />
                </span>
            </TooltipTrigger>
            <TooltipContent className="z-5000">{tooltipText}</TooltipContent>
        </Tooltip>
    ) : (
        <ContentCopy className={`!size-[${iconSize}px]`} />
    );

    return (
        <CopyToClipboard text={copiedText} onCopy={handleCopy}>
            <Button
                variant={variant === "outline" ? "light-outline" : variant}
                size={size}
                aria-label={title || "Copy to clipboard"}
                className={cn(
                    "right-2 h-5 border-0 bg-transparent p-2 shadow-none hover:bg-transparent",
                    className
                )}>
                {copyIcon}
                {title && (
                    <span className="text-sm font-semibold text-slate-500">
                        {title}
                    </span>
                )}
            </Button>
        </CopyToClipboard>
    );
};
