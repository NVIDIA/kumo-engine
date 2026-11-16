// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { ReactNode } from "react";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Skeleton } from "./skeleton";

interface PropertyCardProps extends React.ComponentProps<"div"> {
    id: string;
    className?: string;
    propertyList?: PropertyItem[];
}

type PropertyItem = {
    icon: React.ReactNode;
    label: ReactNode | string;
};

function PropertyCard({
    id,
    propertyList,
    className,
    ...props
}: PropertyCardProps) {
    return (
        <Card
            id={id}
            className={cn(
                "rounded-xs border-1 flex-row border-slate-200 bg-slate-50 px-0 py-0 shadow-none",
                className
            )}
            {...props}>
            {propertyList?.map((item, index) => (
                <div
                    key={`property-item-${index}`}
                    className={cn(
                        "inline-flex items-center justify-start gap-2 px-2 py-1.5",
                        index !== 0 && "border-l border-slate-200"
                    )}>
                    {item.icon}
                    {item.label ? (
                        <div className="text-text-primary justify-start text-sm font-medium leading-tight">
                            {item.label}
                        </div>
                    ) : (
                        <Skeleton className="h-3 w-12 rounded-none bg-neutral-200" />
                    )}
                </div>
            ))}
        </Card>
    );
}

export { PropertyCard };
