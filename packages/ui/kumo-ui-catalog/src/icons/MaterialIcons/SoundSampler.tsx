// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const SoundSampler = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-6 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="M169-227q-42-52-65.5-116.5T80-480q0-157 103.5-270T440-878v80q-120 14-200 104t-80 214q0 56 17 105.5t49 90.5l-57 57ZM480-80q-73 0-138-24t-117-67l56-57q41 32 91.5 50T480-160q57 0 107.5-18t91.5-50l56 57q-52 43-117 67T480-80Zm311-147-57-57q32-41 49-90.5T800-480q0-124-80-214T520-798v-80q153 15 256.5 128T880-480q0 72-23.5 136.5T791-227Zm-411-73v-360l280 180-280 180Z" />
        </svg>
    );
};
