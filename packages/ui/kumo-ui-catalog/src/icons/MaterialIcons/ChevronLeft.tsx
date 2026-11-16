// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const ChevronLeft = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-4 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="M560-240 320-480l240-240 56 56-184 184 184 184-56 56Z" />
        </svg>
    );
};
