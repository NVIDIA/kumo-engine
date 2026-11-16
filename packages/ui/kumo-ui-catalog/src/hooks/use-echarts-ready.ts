// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState, useRef } from "react";

/**
 * Hook to ensure ECharts container has dimensions before rendering
 * Prevents "Can't get DOM width or height" warning
 */
export function useEChartsReady() {
    const [isReady, setIsReady] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        // Check if container has dimensions
        const checkDimensions = () => {
            const { clientWidth, clientHeight } = container;
            if (clientWidth > 0 && clientHeight > 0) {
                setIsReady(true);
                return true;
            }
            return false;
        };

        // Try immediately
        if (checkDimensions()) return;

        // If not ready, wait a frame and try again
        const rafId = requestAnimationFrame(() => {
            if (checkDimensions()) return;

            // If still not ready, use ResizeObserver
            const observer = new ResizeObserver((entries) => {
                for (const entry of entries) {
                    if (
                        entry.contentRect.width > 0 &&
                        entry.contentRect.height > 0
                    ) {
                        setIsReady(true);
                        observer.disconnect();
                    }
                }
            });

            observer.observe(container);

            // Cleanup
            return () => observer.disconnect();
        });

        return () => cancelAnimationFrame(rafId);
    }, []);

    return { isReady, containerRef };
}
