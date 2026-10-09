// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { MetricItem } from "@/types/metric-card";
import React from "react";
import { Symbol } from "./symbol";

interface MetricCardProps extends React.ComponentProps<"div"> {
    id: string;
    className?: string;
    metricList: MetricItem[];
}

function MetricCard({ id, metricList, className, ...props }: MetricCardProps) {
    return (
        <Card
            id={id}
            className={cn(
                "border-1 flex w-max flex-row flex-nowrap items-center gap-0 rounded border-slate-200 bg-white px-1 py-0 shadow-none",
                className
            )}
            {...props}>
            {metricList?.map((item, index) => (
                <div
                    key={`metric-item-${index}`}
                    className={cn(
                        "inline-flex shrink-0 items-center justify-start gap-1.5 px-2 py-1.5",
                        index !== 0 && "border-l border-slate-200"
                    )}>
                    <div className="text-title-secondary justify-start pl-1 text-base font-medium leading-4">
                        {item.name}
                    </div>
                    {item.val !== null && item.val !== undefined ? (
                        <div className="text-title-primary justify-start pr-1 text-base font-medium leading-5">
                            {item.val}
                        </div>
                    ) : (
                        <Symbol type="emdash" />
                    )}
                </div>
            ))}
        </Card>
    );
}

export { MetricCard };
