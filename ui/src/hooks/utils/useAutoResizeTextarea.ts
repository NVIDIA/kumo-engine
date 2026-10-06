// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef } from "react";

export const useAutoResizeTextarea = (
    value: string,
    minWidth: number = 360
) => {
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const adjustTextareaSize = useCallback(() => {
        const textarea = textareaRef.current;
        if (!textarea) return;

        const maxWidth = 1128;
        const maxHeight = 76;

        // Reset height to calculate scrollHeight accurately
        textarea.style.height = "auto";
        const scrollHeight = textarea.scrollHeight;

        // Set height with max limit
        textarea.style.height = `${Math.min(scrollHeight, maxHeight)}px`;

        // Enable overflow if content exceeds max height
        textarea.style.overflowY = scrollHeight > maxHeight ? "auto" : "hidden";

        // Calculate width
        const lines = textarea.value.split("\n");
        const longestLine = lines.reduce(
            (max, line) => (line.length > max.length ? line : max),
            ""
        );

        const estimatedWidth = Math.max(longestLine.length * 7 + 15, minWidth);

        textarea.style.maxWidth = `${Math.min(estimatedWidth, maxWidth)}px`;
    }, [minWidth]);

    useEffect(() => {
        adjustTextareaSize();
    }, [adjustTextareaSize, value]);

    return { textareaRef };
};
