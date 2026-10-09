// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const ArrowRange = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-6 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="M288-288 96-480l192-192 51 51-105 105h492L621-621l51-51 192 192-192 192-51-51 105-105H234l105 105-51 51Z" />
        </svg>
    );
};
