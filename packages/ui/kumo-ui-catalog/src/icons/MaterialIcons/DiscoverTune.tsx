// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const DiscoverTune = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-6 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="M528-600v-72h108v-144h72v144h108v72H528Zm108 456v-384h72v384h-72Zm-384 0v-144H144v-72h288v72H324v144h-72Zm0-288v-384h72v384h-72Z" />
        </svg>
    );
};
