// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const SwapVert = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-6 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="M320-440v-287L217-624l-57-56 200-200 200 200-57 56-103-103v287h-80ZM600-80 400-280l57-56 103 103v-287h80v287l103-103 57 56L600-80Z" />{" "}
        </svg>
    );
};
