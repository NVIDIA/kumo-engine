// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { IconProps } from "@/types/icon-types";

export const BackSpace = ({ className = "", ...props }: IconProps) => {
    const defaultClassName = "size-6 fill-title-secondary";
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 -960 960 960"
            className={cn(defaultClassName, className)}
            {...props}>
            <path d="m480-336 96-96 96 96 48-48-96-96 96-96-48-48-96 96-96-96-48 48 96 96-96 96 48 48ZM372-192q-17 0-32-7.5T314-221L120-480l194-259q11-14 26-21.5t32-7.5h420q30 0 51 21t21 51v432q0 30-21 51t-51 21H372Z" />
        </svg>
    );
};
