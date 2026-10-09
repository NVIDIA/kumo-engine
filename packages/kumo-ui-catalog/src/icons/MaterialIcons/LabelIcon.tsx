// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const LabelIcon = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-6 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="M168-192q-29.7 0-50.85-21.16Q96-234.32 96-264.04v-432.24Q96-726 117.15-747T168-768h420q16.85 0 31.92 7.5Q635-753 646-739l194 259-194 259q-11 14-26.08 21.5Q604.85-192 588-192H168Z" />
        </svg>
    );
};
