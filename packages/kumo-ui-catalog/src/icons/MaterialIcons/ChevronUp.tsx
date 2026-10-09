// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const ChevronUp = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-4 fill-title-secondary";
    return (
        <svg
            style={{ transform: "rotate(-90deg)", display: "inline-block" }}
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="M504-480 320-664l56-56 240 240-240 240-56-56 184-184Z" />
        </svg>
    );
};
