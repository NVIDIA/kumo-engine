// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { ReactNode } from "react";

import { GridCol } from "@/components/blocks/layout/GridCol";
import { GridRow } from "@/components/blocks/layout/GridRow";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { CopyToClipBoard } from "@/components/ui/copy-to-clipboard";
import { Description } from "@/components/ui/description";
import { SectionHeader } from "@/components/ui/section-header";
import { Skeleton } from "@/components/ui/skeleton";
import { TagList } from "@/components/ui/tag-list";
import { Title } from "@/components/ui/title";
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export type OverviewProperties = {
    name: string;
    value: string | number | React.ReactNode;
    tooltip?: React.ReactNode | string;
    icon?: React.ReactNode;
    copyText?: string;
    /** "project" variant only: let this property span the full row width. */
    fullWidth?: boolean;
};

export interface JobOverviewProps {
    properties: OverviewProperties[];
    hideTitle?: boolean;
    queryContent?: ReactNode;
    tags?: Record<string, string>;
    loading?: boolean;
    cardClassName?: string;
    titleSize?: "sm" | "md" | "lg" | "xl";
    title?: string;
    protectedKeys?: Set<string>;
    onTagAdd?: (key: string, value: string | undefined) => void;
    onTagRemove?: (key: string) => void;
    /**
     * Layout variant.
     * - "properties" (default): the job/table/connector overview layout.
     * - "project": the project layout (SectionHeader labels, min-width
     *   columns, optional `asCol` bordered column, no query block).
     */
    variant?: "properties" | "project";
    /** "project" variant only: stack the properties in a bordered column. */
    asCol?: boolean;
}

// Static size → Tailwind class map. A dynamic `text-${titleSize}` is invisible
// to Tailwind's JIT (so the class is never generated), and `text-md` isn't even
// a valid utility — map each size to a real, statically-detectable class.
const titleSizeClasses: Record<
    NonNullable<JobOverviewProps["titleSize"]>,
    string
> = {
    sm: "text-sm",
    md: "text-base",
    lg: "text-lg",
    xl: "text-xl"
};

const PropertiesOverviewCard = ({
    properties,
    hideTitle = false,
    queryContent,
    tags,
    loading = false,
    cardClassName,
    title = "Overview",
    titleSize = "xl",
    protectedKeys,
    onTagAdd,
    onTagRemove,
    variant = "properties",
    asCol = false
}: JobOverviewProps) => {
    const isProject = variant === "project";

    // Skeleton placeholders: the project layout keys off `loading`, the
    // properties layout off an empty list — preserving each original behaviour.
    const showSkeleton = isProject ? loading : properties?.length < 1;

    const titleNode = isProject ? (
        <SectionHeader
            titleSize={titleSize}
            title={title}
            titleClassName="leading-none"
        />
    ) : (
        <Title
            variant="tertiary-heading"
            className={cn(titleSizeClasses[titleSize])}
            children={title}
        />
    );

    return (
        <Card
            className={cn(
                isProject
                    ? cn(
                          "gap-4 rounded-lg border-none px-6",
                          hideTitle ? "py-0 pb-4" : "py-4"
                      )
                    : "gap-6 rounded-lg border-none p-6",
                cardClassName
            )}>
            {!hideTitle && !asCol && (
                <CardHeader className="p-0">{titleNode}</CardHeader>
            )}
            <CardContent
                className={cn(
                    "flex flex-col p-0",
                    isProject ? "gap-4" : "gap-6"
                )}>
                {/* `!` overrides GridRow's default `gap-*` gutter. */}
                <GridRow
                    className={cn(
                        "flex",
                        isProject
                            ? asCol
                                ? "w-[616px] flex-col !gap-[31px] rounded-xl border p-6"
                                : "flex-wrap !gap-x-16 !gap-y-6"
                            : "flex-wrap !gap-x-16 !gap-y-4"
                    )}>
                    {isProject && !hideTitle && asCol && titleNode}
                    {showSkeleton &&
                        Array(5)
                            .fill(null)
                            .map((s, i) => (
                                <GridCol key={`overview-skeleton-${s}-${i}`}>
                                    <Skeleton className="h-8 w-32" />
                                </GridCol>
                            ))}
                    {!showSkeleton &&
                        properties?.map((item) => {
                            const value = item.tooltip ? (
                                <Tooltip id={"id-tooltip"}>
                                    <TooltipTrigger>
                                        <span>{item.value}</span>
                                    </TooltipTrigger>
                                    <TooltipContent
                                        arrow={false}
                                        side={"bottom"}
                                        sideOffset={5}>
                                        {item.tooltip}
                                    </TooltipContent>
                                </Tooltip>
                            ) : (
                                item.value
                            );
                            return (
                                <GridCol
                                    key={"overview-" + item.name}
                                    className={cn(
                                        isProject &&
                                            (item.fullWidth
                                                ? "w-full"
                                                : "min-w-[172px]")
                                    )}>
                                    {isProject ? (
                                        <SectionHeader
                                            title={item.name}
                                            titleSize="sm"
                                            titleVariant="secondary"
                                            titleClassName="leading-5 !font-medium"
                                            className="text-title-secondary"
                                        />
                                    ) : (
                                        <Description
                                            size="sm"
                                            variant="secondary"
                                            className="py-0 font-medium">
                                            {item.name}
                                        </Description>
                                    )}
                                    <div
                                        className={cn(
                                            "text-title-primary mt-2 flex items-center gap-1 text-base font-medium",
                                            isProject && "leading-5"
                                        )}>
                                        {item.icon && item.icon}
                                        {value}
                                        {item.copyText && (
                                            <CopyToClipBoard
                                                copiedText={
                                                    item.copyText as string
                                                }
                                                iconSize={16}
                                                tooltipText={"Copied"}
                                            />
                                        )}
                                    </div>
                                </GridCol>
                            );
                        })}
                </GridRow>

                {!isProject && (queryContent || loading) && (
                    <div className="flex flex-col">
                        <Description
                            size="sm"
                            variant="secondary"
                            className="py-0 font-medium">
                            {"Query"}
                        </Description>
                        <div className="h-2" />
                        <Card className="relative px-4 py-2">
                            {queryContent ?? (
                                <Skeleton className="h-24 w-full" />
                            )}
                        </Card>
                    </div>
                )}

                {tags && (
                    <TagList
                        tags={tags}
                        onTagAdd={onTagAdd}
                        onTagRemove={onTagRemove}
                        protectedKeys={protectedKeys}
                    />
                )}
            </CardContent>
        </Card>
    );
};

export { PropertiesOverviewCard };
