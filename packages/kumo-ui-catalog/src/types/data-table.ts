// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

declare module "@tanstack/react-table" {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    interface ColumnMeta<TData, TValue> {
        sticky?: StickyColumnOptions;
        className?: string;
    }
}

export interface StickyColumnOptions {
    type?: "always" | "on-scroll";
    left?: boolean | number;
    right?: boolean | number;
    zIndex?: number;
}
