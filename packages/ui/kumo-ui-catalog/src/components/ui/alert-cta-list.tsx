// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";
import { useEffect, useMemo, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type AlertCtaLevel = "info" | "warning" | "error";

export interface AlertCtaItem {
    title?: string;
    description?: string;
    cta?: { name?: string; url?: string };
}

export interface AlertCtaListProps {
    id?: string;
    data: AlertCtaItem[];
    level: AlertCtaLevel;
    title?: string;
    hideTitle?: boolean;
    /** How many items to show before the Show More toggle. Defaults to 3. */
    previewCount?: number;
    /** Hide the Show More / Less toggle entirely (e.g. when rendered in a popover). */
    disablePreview?: boolean;
    onClose?: () => void;
    onCtaClick?: (url: string, e: React.MouseEvent<HTMLButtonElement>) => void;
    /** Caller decides whether each CTA renders (e.g., hide CTAs that point at the current page). */
    shouldShowCta?: (url: string) => boolean;
    /** Hook for callers to normalize description text (e.g., adjustError). */
    formatDescription?: (text?: string) => React.ReactNode;
    className?: string;
}

const LEVEL_TO_VARIANT: Record<
    AlertCtaLevel,
    "default" | "warn" | "destructive"
> = {
    info: "default",
    warning: "warn",
    error: "destructive"
};

function defaultTitle(level: AlertCtaLevel, plural: boolean) {
    if (level === "info") return "Info";
    if (level === "warning") return plural ? "Warnings" : "Warning";
    return plural ? "Errors" : "Error";
}

export function AlertCtaList({
    id,
    data,
    level,
    title,
    hideTitle = false,
    previewCount = 3,
    disablePreview = false,
    onClose,
    onCtaClick,
    shouldShowCta,
    formatDescription = (text) => text,
    className
}: AlertCtaListProps) {
    const total = data?.length ?? 0;
    const hasOverflow = !disablePreview && total > previewCount;
    const [expanded, setExpanded] = useState(false);

    useEffect(() => {
        // Reset expansion when the data set changes
        setExpanded(false);
    }, [data]);

    const visible = useMemo(() => {
        if (disablePreview || expanded) return data;
        if (total > previewCount) return data.slice(0, previewCount);
        return data;
    }, [data, disablePreview, expanded, previewCount, total]);

    if (!data || total === 0) return null;

    const resolvedTitle =
        title && title !== "" ? title : defaultTitle(level, visible.length > 1);
    const isDismissible = typeof onClose === "function";

    return (
        <Alert
            id={id}
            data-slot="alert-cta-list"
            data-level={level}
            variant={LEVEL_TO_VARIANT[level]}
            className={cn("w-full items-start gap-4 p-4", className)}
            enableClose={isDismissible}
            onClose={onClose}>
            {!hideTitle && (
                <AlertTitle className="pb-0 pt-0.5 leading-5">
                    {resolvedTitle}
                </AlertTitle>
            )}

            <AlertDescription className={!hideTitle ? "pt-4" : "pt-0"}>
                {visible.map((item, idx) => {
                    const ctaUrl = item.cta?.url;
                    const showCta =
                        item.cta?.name &&
                        ctaUrl &&
                        (shouldShowCta ? shouldShowCta(ctaUrl) : true);
                    return (
                        <div
                            key={`alert-cta-${idx}`}
                            data-slot="alert-cta-item">
                            {item.title ? <strong>{item.title}:</strong> : null}
                            <span>{formatDescription(item.description)}</span>
                            {showCta && (
                                <Button
                                    className="h-6 py-0"
                                    variant="link"
                                    size="sm"
                                    onClick={(e) => {
                                        e.preventDefault();
                                        onCtaClick?.(ctaUrl as string, e);
                                    }}>
                                    {item.cta?.name}
                                </Button>
                            )}
                        </div>
                    );
                })}

                {hasOverflow && (
                    <Button
                        className="p-0"
                        size="sm"
                        variant="link"
                        onClick={() => setExpanded((v) => !v)}>
                        {expanded ? "Show Less" : "Show More"}
                    </Button>
                )}
            </AlertDescription>
        </Alert>
    );
}
