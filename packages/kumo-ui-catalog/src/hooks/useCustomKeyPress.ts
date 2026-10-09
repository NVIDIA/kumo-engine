// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect } from "react";

/*
    Custom hook that calls `callback(event)` whenever `key` is pressed. If
    `withMeta` is `true`, then we also require the CMD key to be pressed to
    trigger the callback. Otherwise, we also require the CTRL key to be pressed

    This can only be used inside functional components.

    Usage Example:
    ```
    const toggleModal = () => {
        setOpen(!open);
    };

    // Run `toggleModal` when CMD + d is clicked
    useCustomKeyPress(
        (key) "d",
        (withMeta) true,
        (callback) toggleModal
    );
    ```
*/
export const useCustomKeyPress = (
    key: string,
    withMeta: boolean,
    callback: (event: KeyboardEvent) => void
) => {
    const toggleOpen = useCallback(
        (e: KeyboardEvent) => {
            const metaKeyIsDown =
                e?.metaKey ||
                e?.getModifierState?.("OS") ||
                e?.getModifierState?.("Meta") ||
                e?.getModifierState?.("Win");
            const ctrlKeyIsDown = e.ctrlKey;
            if (
                e &&
                (withMeta ? metaKeyIsDown : ctrlKeyIsDown) &&
                e.key === key
            ) {
                e.preventDefault();
                callback(e);
            }
        },
        [key, callback, withMeta]
    );

    useEffect(() => {
        window.addEventListener("keydown", toggleOpen);
        return () => {
            window.removeEventListener("keydown", toggleOpen);
        };
    }, [toggleOpen]);
};
