// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

/**
 * Check whether the browser tab is focused
 * E.g. usage: Prevent polling backend for Model/Prediction tables when
 *             user isn't actively using frontend
 * @returns true if the browser tab is focused
 */
export const useWindowFocus = () => {
    const [isFocused, setIsFocused] = useState(true);

    useEffect(() => {
        const focusEventListener = () => setIsFocused(true);
        const blurEventListener = () => setIsFocused(false);

        window.addEventListener("blur", blurEventListener);
        window.addEventListener("focus", focusEventListener);

        return () => {
            window.removeEventListener("blur", blurEventListener);
            window.removeEventListener("focus", focusEventListener);
        };
    }, []);

    return isFocused;
};

export default useWindowFocus;
