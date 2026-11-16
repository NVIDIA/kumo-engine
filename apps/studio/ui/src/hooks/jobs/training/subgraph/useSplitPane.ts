// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";

export function useSplitPane() {
    const [splitPercent, setSplitPercent] = useState(55);
    const containerRef = useRef<HTMLDivElement>(null);
    const leftCardRef = useRef<HTMLDivElement>(null);
    const isDragging = useRef(false);

    useEffect(() => {
        const onMove = (e: MouseEvent) => {
            if (!isDragging.current || !containerRef.current) return;
            const rect = containerRef.current.getBoundingClientRect();
            const pct = Math.max(
                30,
                Math.min(75, ((e.clientX - rect.left) / rect.width) * 100)
            );
            setSplitPercent(pct);
        };
        const onUp = () => {
            isDragging.current = false;
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        return () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
        };
    }, []);

    return { splitPercent, containerRef, leftCardRef, isDragging };
}
