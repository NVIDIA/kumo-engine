// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

interface ChartLabelProps {
    iconClassName?: string;
    labelClassName?: string;
    text?: string;
}

export const ChartLabel = ({
    iconClassName = "",
    labelClassName = "",
    text = ""
}: ChartLabelProps) => {
    const defIconClassName = "h-3 w-4 bg-slate-500";
    const defLabelClassName = "mb-0 flex-none leading-none text-slate-500";

    return (
        <div className="mb-2 flex items-center gap-2">
            <div className={cn(defIconClassName, iconClassName)} />
            <Label className={cn(defLabelClassName, labelClassName)}>
                {text}
            </Label>
        </div>
    );
};
