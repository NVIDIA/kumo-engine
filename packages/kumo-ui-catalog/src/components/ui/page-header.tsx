// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";

import { Button } from "@/components/ui/button";
import { CopyToClipBoard } from "@/components/ui/copy-to-clipboard";
import { Description } from "@/components/ui/description";
import { Skeleton } from "@/components/ui/skeleton";
import { Title } from "@/components/ui/title";
import { CloseSmall } from "@/icons";
import { cn } from "@/lib/utils";

export interface PageHeaderProps {
    title?: React.ReactNode;
    titleCopyText?: string | null;
    subTitles?: React.ReactNode;
    preTitle?: string;
    icon?: React.ReactNode;
    loadingStatus?: boolean;
    onCloseDialog?: () => void;
    hidePreTitle?: boolean;
    headerButton?: React.ReactNode;
    disableCopy?: boolean;
    screenshotTestId?: string;
    className?: string;
}

export function PageHeader({
    title,
    titleCopyText,
    subTitles,
    preTitle,
    icon,
    loadingStatus,
    onCloseDialog,
    hidePreTitle = false,
    headerButton,
    disableCopy = false,
    screenshotTestId,
    className
}: PageHeaderProps) {
    return (
        <div data-slot="page-header" className={cn("w-full", className)}>
            {!hidePreTitle && (
                <div className="flex items-center gap-2">
                    {icon && <div data-slot="page-header-icon">{icon}</div>}
                    <div data-slot="page-header-pre-title" className="flex-1">
                        {preTitle && (
                            <Title variant="page-heading">{preTitle}</Title>
                        )}
                    </div>
                    <div className="ms-auto flex items-center gap-2">
                        {loadingStatus ? (
                            <Skeleton className="h-8 w-60" />
                        ) : (
                            (headerButton ?? null)
                        )}
                    </div>
                    {onCloseDialog && (
                        <div className="flex items-center">
                            <Button
                                data-testid="close-icon"
                                aria-label="Close"
                                variant="outline"
                                className="border-border-primary hover:bg-kui-bg-hover w-9 px-1.5 [&_svg]:size-5"
                                onClick={onCloseDialog}>
                                <CloseSmall />
                            </Button>
                        </div>
                    )}
                </div>
            )}
            {!hidePreTitle && <div className="h-4" />}

            {title && (
                <Title
                    data-testid={screenshotTestId}
                    data-screenshot-ready={
                        loadingStatus ? undefined : screenshotTestId
                    }
                    aria-label={loadingStatus ? "Loading" : undefined}
                    aria-busy={loadingStatus || undefined}
                    variant="primary-heading">
                    {loadingStatus ? (
                        <Skeleton className="h-7 w-40" />
                    ) : (
                        <div className="flex items-center gap-1">
                            <span>{title}</span>
                            {!disableCopy && titleCopyText && (
                                <CopyToClipBoard
                                    copiedText={titleCopyText}
                                    iconSize={16}
                                    tooltipText="Copied"
                                />
                            )}
                        </div>
                    )}
                </Title>
            )}
            {subTitles !== undefined && (
                <Description size="sm" variant="secondary" className="my-0">
                    {subTitles}
                </Description>
            )}
        </div>
    );
}
