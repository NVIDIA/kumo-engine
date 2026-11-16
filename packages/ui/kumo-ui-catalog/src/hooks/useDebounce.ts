// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from "react";

export const useWithDebounce = (debounceFunc: () => void, seconds: number) => {
    const timer = useRef<NodeJS.Timeout | null>(null);
    const restartFunc = useCallback(() => {
        clearTimeout(timer?.current as NodeJS.Timeout);

        // Start a timer to run debounceFunc in x seconds
        timer.current = setTimeout(debounceFunc, seconds * 1000);
    }, [debounceFunc, seconds]);
    return restartFunc;
};

export function useDebounce<T>(value: T, delay: number): T {
    const [debouncedValue, setDebouncedValue] = useState<T>(value);

    useEffect(() => {
        const timer = setTimeout(() => {
            setDebouncedValue(value);
        }, delay);

        return () => {
            clearTimeout(timer);
        };
    }, [value, delay]);

    return debouncedValue;
}
