// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getFromLocalStorage, storeToLocalStorage } from "@/utils/storage";
import { useEffect, useState } from "react";

export function useLocalStorage<T>(key: string, initialValue: T) {
    const [value, setValue] = useState<T>(
        () => getFromLocalStorage<T>(key) ?? initialValue
    );

    useEffect(() => {
        const handler = (e: StorageEvent) => {
            if (e.key === key && e.newValue) {
                setValue(getFromLocalStorage<T>(key) ?? initialValue);
            }
        };
        window.addEventListener("storage", handler);
        return () => window.removeEventListener("storage", handler);
    }, [key, initialValue]);

    const set = (newValue: T) => {
        setValue(newValue);
        storeToLocalStorage(key, newValue);
        window.dispatchEvent(
            new StorageEvent("storage", {
                key,
                newValue: JSON.stringify(newValue)
            })
        );
    };

    return [value, set] as const;
}
