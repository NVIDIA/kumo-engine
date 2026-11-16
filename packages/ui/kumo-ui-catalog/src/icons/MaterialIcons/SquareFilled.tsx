// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

//Notes: This icon is used in ExperimentMonitoringMetric chart only. Since this chart
// has other custom icons which uses hex colors, the color prop is retained here.
// When that chart is migrated to catalog, the color prop will be removed here.
export const SquareFilled = ({
    className = "",
    color = "#4F888E",
    ...props
}: IconProps & { color?: string }) => {
    const defaultClassName = "size-6 ";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            fill={color}
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="M120-120v-720h720v720H120Z" />
        </svg>
    );
};
