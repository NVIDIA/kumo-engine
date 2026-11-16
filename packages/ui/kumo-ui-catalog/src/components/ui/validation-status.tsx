// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";
import { useMemo } from "react";

import {
    HoverCard,
    HoverCardContent,
    HoverCardTrigger
} from "@/components/ui/hover-card";
import {
    CheckCircle,
    Error as ErrorIcon,
    ProgressActivity,
    TextSelectStart
} from "@/icons";
import { cn } from "@/lib/utils";

export type ValidationState =
    | "none"
    | "valid"
    | "validating"
    | "warning"
    | "invalid";

export interface ValidationStatusProps {
    state: ValidationState;
    emptyText?: string;
    /** Rendered inside a hover card when provided. If absent, just the pill renders. */
    hoverContent?: React.ReactNode;
    className?: string;
}

interface Visual {
    title: string;
    icon: React.ReactNode;
    /** Pill text + border color classes. */
    pillClass: string;
}

function getVisual(state: ValidationState, emptyText: string): Visual {
    switch (state) {
        case "valid":
            return {
                title: "Looks good!",
                icon: <CheckCircle className="size-4.5 fill-green-700" />,
                pillClass: "text-emerald-700 border-emerald-700"
            };
        case "validating":
            return {
                title: "Checking query",
                icon: <ProgressActivity className="size-4.5 animate-spin" />,
                pillClass: "text-text-secondary border-text-secondary"
            };
        case "warning":
            return {
                title: "Looks good!",
                icon: <CheckCircle className="size-4.5 fill-amber-700" />,
                pillClass: "text-amber-700 border-amber-500"
            };
        case "invalid":
            return {
                title: "Issues found",
                icon: <ErrorIcon className="size-4.5 fill-red-500" />,
                pillClass: "text-red-500 border-red-500"
            };
        case "none":
        default:
            return {
                title: emptyText,
                icon: <TextSelectStart className="size-4.5" />,
                pillClass: "text-text-secondary border-text-secondary"
            };
    }
}

export function ValidationStatus({
    state,
    emptyText = "Empty Query",
    hoverContent,
    className
}: ValidationStatusProps) {
    const { title, icon, pillClass } = useMemo(
        () => getVisual(state, emptyText),
        [state, emptyText]
    );

    const pill = (
        <div
            data-testid="validation-status"
            data-slot="validation-status"
            data-state={state}
            className={cn(
                "inline-flex h-9 items-center rounded-full border px-4 py-2",
                pillClass,
                className
            )}>
            {icon}
            <span className="font-inter ms-2">{title}</span>
        </div>
    );

    if (!hoverContent) return pill;

    return (
        <HoverCard>
            <HoverCardTrigger asChild>{pill}</HoverCardTrigger>
            <HoverCardContent
                className="gap-1.5 border-none p-0"
                sideOffset={-33}
                arrow={false}>
                {hoverContent}
            </HoverCardContent>
        </HoverCard>
    );
}
