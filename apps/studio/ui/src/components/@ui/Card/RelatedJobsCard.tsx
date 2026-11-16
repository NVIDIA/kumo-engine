// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from "react";

import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
    Badge,
    cn,
    Skeleton,
    Title
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { jobStatus } from "utils/Constants";
import { capitalizeFirstLetter } from "utils/Helpers/stringHelpers";
import { getColorVariant } from "utils/jobs";

export interface RelatedJobsCardProps {
    jobName: string;
    jobId: string;
    loadingStatus?: boolean;
    status?: jobStatus | undefined;
    children: React.ReactNode;
}

const RelatedJobsCard = ({
    jobName = "",
    jobId,
    loadingStatus,
    status,
    children
}: RelatedJobsCardProps) => {
    const [activeKey, setActiveKey] = useState<string[] | undefined>([]); // Track the active key
    const valueKey = "related-jobs-" + jobId;

    return (
        <Accordion
            {...{ type: "multiple" }}
            value={activeKey}
            onValueChange={setActiveKey}>
            <AccordionItem
                className="mb-4"
                value={valueKey}
                data-testid={"acc-item-" + valueKey}>
                <AccordionTrigger
                    data-testid={"acc-trig-" + valueKey}
                    showBorder={false}
                    className={cn(
                        "rounded-lg items-center p-6 gap-2",
                        activeKey?.includes(valueKey) && "rounded-b-none pb-0"
                    )}>
                    {!jobName ? (
                        <Skeleton className="h-5 w-72.5" />
                    ) : (
                        <div className="p-0 flex items-center justify-between gap-2">
                            <div className="flex items-center gap-2">
                                <Title
                                    variant="tertiary-heading"
                                    className="text-xl"
                                    children={jobName}
                                />
                                {!activeKey?.includes(valueKey) && (
                                    <div className="text-xl font-bold text-title-secondary">
                                        {jobId}
                                    </div>
                                )}
                            </div>

                            {/* div that shows elements on right side */}
                            <div className="flex items-center gap-1">
                                {loadingStatus ? (
                                    <Skeleton className="h-3 w-20" />
                                ) : status && !activeKey?.includes(valueKey) ? (
                                    <Badge
                                        size="md"
                                        variant={getColorVariant(status)}
                                        id={"header-chip-id"}
                                        data-testid={"header-status-chip-id"}>
                                        {capitalizeFirstLetter(
                                            status?.toLowerCase()
                                        )}
                                    </Badge>
                                ) : (
                                    <></>
                                )}
                            </div>
                        </div>
                    )}
                </AccordionTrigger>

                <AccordionContent
                    showBorder={false}
                    forceMount
                    className="m-0 p-0"
                    data-testid={"acc-content-" + valueKey}>
                    {children}
                </AccordionContent>
            </AccordionItem>
        </Accordion>
    );
};

export default WithErrorBoundary(RelatedJobsCard);
