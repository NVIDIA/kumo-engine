// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const RightPanelOpen = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-6 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="M456-324v-312L288-480l168 156ZM211-144q-27.64 0-47.32-19.68T144-211v-538q0-27.64 19.68-47.32T211-816h538q27.64 0 47.32 19.68T816-749v538q0 27.64-19.68 47.32T749-144H211Zm341-72v-528H216v528h336Z" />
        </svg>
    );
};
