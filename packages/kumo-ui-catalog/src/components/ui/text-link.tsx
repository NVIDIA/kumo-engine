// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import * as React from "react";

interface TextLinkProps {
    id: string;
    title: string | React.ReactNode;
    data_testid?: string;
    className?: string;
    pageLink?: string;
    disabled?: boolean;
    style?: React.CSSProperties;
    tooltipText?: string;
    newTab?: boolean;
    onClick?: ((e: React.MouseEvent<HTMLAnchorElement>) => void) | null;
}

export const TextLink: React.FC<TextLinkProps> = ({
    id,
    title,
    data_testid,
    className = "",
    pageLink,
    disabled = false,
    style,
    tooltipText,
    newTab = false,
    onClick
}) => {
    // KUI inline Anchor pattern (foundations `.nv-anchor`): primary text color
    // with a brand-green underline (4px offset, 1px thickness). On hover AND
    // keyboard focus-visible the underline shifts to the interaction-hover
    // border color and the text gets the shared 5% interaction-hover background
    // — the same tokens KUI (and our tabs/buttons) use for hover/focus states.
    const defaultClassName =
        "transition-colors text-text-primary underline decoration-brand-green decoration-1 underline-offset-4 hover:decoration-kui-hover-border hover:bg-kui-bg-hover focus-visible:decoration-kui-hover-border focus-visible:bg-kui-bg-hover";
    if (disabled) {
        return (
            <Tooltip>
                <TooltipTrigger>
                    <span
                        id={id}
                        style={style}
                        className={cn(
                            "text-muted-foreground cursor-not-allowed",
                            className
                        )}>
                        {title}
                    </span>
                </TooltipTrigger>
                <TooltipContent side="bottom" sideOffset={5}>
                    {tooltipText}
                </TooltipContent>
            </Tooltip>
        );
    }

    return (
        <span className={cn(defaultClassName, className)}>
            <a
                id={id}
                href={pageLink}
                style={style}
                target={newTab ? "_blank" : undefined}
                rel={newTab ? "noreferrer" : undefined}
                onClick={onClick || undefined}
                data-testid={data_testid}>
                {title}
            </a>
        </span>
    );
};
