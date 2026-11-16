// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";

/**
 * Custom hook to lock/unlock body scroll
 * Useful for modals, side panels, and overlays
 *
 * @param isLocked - Whether the body scroll should be locked
 *
 * @example
 * ```tsx
 * const [isPanelOpen, setIsPanelOpen] = useState(false);
 * useBodyScrollLock(isPanelOpen);
 * ```
 */
export const useBodyScrollLock = (isLocked: boolean): void => {
    useEffect(() => {
        if (isLocked) {
            const originalOverflow = document.body.style.overflow;
            document.body.style.overflow = "hidden";

            return () => {
                document.body.style.overflow = originalOverflow;
            };
        } else {
            document.body.style.overflow = "";
        }
    }, [isLocked]);
};
