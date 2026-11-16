// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";

import { cn } from "@/lib/utils";

export interface StatusIconProps<TStatus extends string> {
    status: TStatus;
    iconMap: Partial<Record<TStatus, React.ReactNode>>;
    className?: string;
    fallback?: React.ReactNode;
}

export function StatusIcon<TStatus extends string>({
    status,
    iconMap,
    className,
    fallback = null
}: StatusIconProps<TStatus>) {
    const icon = iconMap[status];
    if (icon === undefined) return <>{fallback}</>;
    return (
        <span
            data-slot="status-icon"
            data-status={status}
            className={cn("inline-flex items-center", className)}>
            {icon}
        </span>
    );
}
