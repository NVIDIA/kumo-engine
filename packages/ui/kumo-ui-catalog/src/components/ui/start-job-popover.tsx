// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChevronRight } from "@/icons/MaterialIcons/ChevronRight";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { Button } from "./button";
import { Card, CardContent } from "./card";
import { Description } from "./description";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";
import { SectionHeader } from "./section-header";
import { TagList } from "./tag-list";

interface StartJobPopoverProps {
    children?: React.ReactNode;
    jobsSectionTitle?: string | React.ReactNode;
    headerLabel?: string | React.ReactNode;
    tagHeader?: string | React.ReactNode;
    tagDescription?: string | React.ReactNode;
    jobs?: string[];
    initialTags?: Record<string, string>;
    showPopover: boolean;
    setShowPopover: (val: boolean) => void;
    disablePopover?: boolean;
    popoverContentClassName?: string;
    startButtonTitle: string;
    startTraining: (tags: Record<string, string>) => void;
}

export function StartJobPopover({
    children = null,
    jobsSectionTitle = "",
    headerLabel = "",
    tagHeader = "",
    tagDescription = "",
    jobs,
    initialTags = {},
    showPopover,
    setShowPopover,
    disablePopover = false,
    popoverContentClassName = "w-[419px]", // matches Figma design spec width
    startButtonTitle,
    startTraining
}: StartJobPopoverProps) {
    const [tags, setTags] = useState<Record<string, string>>(initialTags);

    if (disablePopover) return <>{children}</>;

    return (
        <Popover open={showPopover} onOpenChange={setShowPopover} modal={true}>
            <PopoverTrigger asChild>{children}</PopoverTrigger>
            <PopoverContent
                className={cn("p-4", popoverContentClassName)}
                align="end"
                side="bottom">
                {jobsSectionTitle && (
                    <SectionHeader
                        title={jobsSectionTitle}
                        titleSize="md"
                        titleClassName="!font-bold !leading-5 pb-1"
                    />
                )}
                {headerLabel && (
                    <Description
                        size="sm"
                        variant="secondary"
                        className="m-0! leading-5! pb-2 pt-0">
                        {headerLabel}
                    </Description>
                )}

                <div className="flex flex-col gap-2">
                    {jobs?.map((job, i) => (
                        <Card
                            key={`${job}-${i}`}
                            className="border-border-secondary w-64 items-start rounded-sm py-0">
                            <CardContent className="px-4 py-3">
                                <SectionHeader
                                    title={job}
                                    titleVariant="default"
                                    titleSize="md"
                                    titleClassName="!font-medium !leading-5"
                                />
                            </CardContent>
                        </Card>
                    ))}
                </div>
                {tagHeader && (
                    <SectionHeader
                        title={tagHeader}
                        titleSize="md"
                        className="mt-6"
                        titleClassName="!font-bold !leading-5 pb-1"
                    />
                )}
                {tagDescription && (
                    <Description
                        size="sm"
                        variant="secondary"
                        className="m-0! leading-5! pb-2">
                        {tagDescription}
                    </Description>
                )}

                <Card className="border-border-secondary w-full items-start overflow-hidden rounded-sm px-0 py-0">
                    <CardContent className="w-full overflow-x-auto px-4 py-3">
                        <TagList
                            tags={tags}
                            showHeader={false}
                            onTagAdd={(key, value) =>
                                setTags((prev) => ({
                                    ...prev,
                                    [key]: value ?? ""
                                }))
                            }
                            onTagRemove={(key) =>
                                setTags((prev) => {
                                    const next = { ...prev };
                                    delete next[key];
                                    return next;
                                })
                            }
                        />
                    </CardContent>
                </Card>

                <Button
                    id="start-job"
                    className="mt-6 rounded-sm"
                    onClick={() => {
                        startTraining(tags);
                        setShowPopover(false);
                    }}>
                    <span className="text-base">{startButtonTitle}</span>
                    <ChevronRight />
                </Button>
            </PopoverContent>
        </Popover>
    );
}
