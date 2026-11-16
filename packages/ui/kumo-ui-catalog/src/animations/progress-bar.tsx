// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface ProgressBarProps {
    progress: number;
    color?: string;
    className?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
    progress,
    color,
    className
}) => {
    return (
        <div
            className={cn(
                "relative h-5 w-full overflow-hidden bg-transparent",
                className
            )}
            style={
                {
                    "--progress-width": `${progress}%`,
                    "--progress-color": color
                } as React.CSSProperties
            }>
            {/* Background stripes */}
            <div className="absolute inset-0 bg-[repeating-linear-gradient(90deg,white,white_2px,transparent_2px,transparent_8px)]" />

            {/* Use shadcn Progress but override its styles */}
            <Progress
                value={progress}
                className="absolute inset-0 bg-transparent [&>div]:bg-transparent"
            />

            {/* Custom progress indicator with stripes */}
            <div className="absolute left-0 top-0 h-full w-[var(--progress-width)] bg-[repeating-linear-gradient(90deg,var(--progress-color),var(--progress-color)_2px,transparent_2px,transparent_8px)] transition-all duration-500 ease-in-out" />
        </div>
    );
};
