// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const Add = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-6 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="M450-450H200v-60h250v-250h60v250h250v60H510v250h-60v-250Z" />{" "}
        </svg>
    );
};
