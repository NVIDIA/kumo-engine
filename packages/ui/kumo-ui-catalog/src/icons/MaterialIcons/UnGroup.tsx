// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const UnGroup = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-6 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="m339-288-51-51 191-190H313v-72h288v288h-72v-166L339-288Zm405-192v-264H480v-72h336v336h-72ZM216-144q-29.7 0-50.85-21.15Q144-186.3 144-216v-600h72v600h600v72H216Z" />
        </svg>
    );
};
