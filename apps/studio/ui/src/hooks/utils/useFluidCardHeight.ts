// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";

interface UseFluidCardHeightOptions {
    /** Minimum height in px to clamp the result to. */
    minHeight?: number;
    /** Bottom padding (e.g., parent layout's bottom padding) to subtract. */
    bottomPadding?: number;
    /**
     * External values that should force a re-measurement when they change.
     * Use for layout-affecting state that sits above the measured element
     * (banners, validation alerts) and doesn't trigger ResizeObserver on the
     * header itself.
     */
    deps?: React.DependencyList;
}

interface UseFluidCardHeightReturn<T extends HTMLElement> {
    headerRef: React.RefObject<T>;
    cardHeight: number | undefined;
}

/**
 * Measures the bottom edge of `headerRef` and computes the remaining viewport
 * height below it, so a sibling card can fill the rest of the screen.
 *
 * Re-measures on window resize, when the header element itself resizes
 * (e.g., async data populates), AND when any value in `deps` changes
 * (e.g., a sibling banner above the header toggles visibility).
 */
const useFluidCardHeight = <T extends HTMLElement = HTMLDivElement>({
    minHeight = 300,
    bottomPadding = 24,
    deps = []
}: UseFluidCardHeightOptions = {}): UseFluidCardHeightReturn<T> => {
    const headerRef = useRef<T>(null);
    const [cardHeight, setCardHeight] = useState<number | undefined>(undefined);

    useEffect(() => {
        const measure = () => {
            if (!headerRef.current) return;
            const rect = headerRef.current.getBoundingClientRect();
            const h = window.innerHeight - rect.bottom - bottomPadding;
            setCardHeight(Math.max(h, minHeight));
        };

        measure();
        window.addEventListener("resize", measure);

        let observer: ResizeObserver | undefined;
        if (typeof ResizeObserver !== "undefined" && headerRef.current) {
            observer = new ResizeObserver(measure);
            observer.observe(headerRef.current);
        }

        return () => {
            window.removeEventListener("resize", measure);
            observer?.disconnect();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [minHeight, bottomPadding, ...deps]);

    return { headerRef, cardHeight };
};

export default useFluidCardHeight;
