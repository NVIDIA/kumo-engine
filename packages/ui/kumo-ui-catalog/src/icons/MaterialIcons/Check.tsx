// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const Check = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-6 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 16 16"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="M6.36738 11.9991L2.56738 8.19909L3.51738 7.24909L6.36738 10.0991L12.4841 3.98242L13.4341 4.93242L6.36738 11.9991Z" />
        </svg>
    );
};
