// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const SortUpDown = ({
    className = "",
    upClassName = "",
    downClassName = "",
    ...props
}: IconProps & { upClassName?: string; downClassName?: string }) => {
    const defaultClassName = "size-6 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="m340-530 140-230 140 230H340Z" className={upClassName} />
            <path d="M480-200 340-430h280L480-200Z" className={downClassName} />
        </svg>
    );
};
