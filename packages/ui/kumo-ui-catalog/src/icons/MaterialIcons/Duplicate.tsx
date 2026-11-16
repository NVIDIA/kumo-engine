// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const Duplicate = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-6 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="M11 10H9V13H6V15H9V18H11V15H14V13H11V10Z" />
            <path d="M4 22H16C17.103 22 18 21.103 18 20V8C18 6.897 17.103 6 16 6H4C2.897 6 2 6.897 2 8V20C2 21.103 2.897 22 4 22ZM4 8H16L16.002 20H4V8Z" />
            <path d="M20 2H8V4H20V16H22V4C22 2.897 21.103 2 20 2Z" />
        </svg>
    );
};
