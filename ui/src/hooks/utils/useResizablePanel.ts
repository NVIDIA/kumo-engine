// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo, useRef, useState } from "react";

interface UseResizablePanelReturn {
    topPx: number;
    isDragging: boolean;
    handleProps: {
        onPointerDown: (e: React.PointerEvent) => void;
        onPointerMove: (e: React.PointerEvent) => void;
        onPointerUp: (e: React.PointerEvent) => void;
        onLostPointerCapture: () => void;
        onKeyDown: (e: React.KeyboardEvent) => void;
    };
    minTop: number;
    maxTop: number;
}

const KEYBOARD_STEP = 20;

/**
 * Tracks the pixel height of the top panel via drag (mouse + touch).
 * The bottom panel should use `flex: 1` to fill remaining space.
 *
 * @param defaultTop  Initial top panel height in px
 * @param minTop      Minimum top panel height in px
 * @param maxTop      Maximum top panel height in px
 */
const useResizablePanel = (
    defaultTop: number,
    minTop: number,
    maxTop: number
): UseResizablePanelReturn => {
    const [topPx, setTopPx] = useState(defaultTop);
    const dragging = useRef(false);
    const [isDragging, setIsDragging] = useState(false);
    const startY = useRef(0);
    const startTop = useRef(0);

    const clamp = useCallback(
        (v: number) => Math.min(maxTop, Math.max(minTop, v)),
        [minTop, maxTop]
    );

    const stopDrag = useCallback(() => {
        dragging.current = false;
        setIsDragging(false);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
    }, []);

    const onPointerDown = useCallback(
        (e: React.PointerEvent) => {
            (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
            startY.current = e.clientY;
            startTop.current = topPx;
            dragging.current = true;
            setIsDragging(true);
            document.body.style.cursor = "row-resize";
            document.body.style.userSelect = "none";
        },
        [topPx]
    );

    const onPointerMove = useCallback(
        (e: React.PointerEvent) => {
            if (!dragging.current) return;
            const delta = e.clientY - startY.current;
            setTopPx(clamp(startTop.current + delta));
        },
        [clamp]
    );

    const onPointerUp = useCallback(() => {
        if (!dragging.current) return;
        stopDrag();
    }, [stopDrag]);

    const onKeyDown = useCallback(
        (e: React.KeyboardEvent) => {
            if (e.key === "ArrowUp") {
                e.preventDefault();
                setTopPx((prev) => clamp(prev - KEYBOARD_STEP));
            } else if (e.key === "ArrowDown") {
                e.preventDefault();
                setTopPx((prev) => clamp(prev + KEYBOARD_STEP));
            }
        },
        [clamp]
    );

    const handleProps = useMemo(
        () => ({
            onPointerDown,
            onPointerMove,
            onPointerUp,
            onLostPointerCapture: stopDrag,
            onKeyDown
        }),
        [onPointerDown, onPointerMove, onPointerUp, stopDrag, onKeyDown]
    );

    return { topPx, isDragging, handleProps, minTop, maxTop };
};

export default useResizablePanel;
