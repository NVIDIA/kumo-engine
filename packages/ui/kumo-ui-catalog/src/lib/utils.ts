// SPDX-FileCopyrightText: Copyright (c) 2023 shadcn
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

// Inject scrollbar styles once globally for a given class name
export const createScrollbarInjector = (className: string) => {
    let injected = false;
    return () => {
        if (injected) return;
        injected = true;

        const style = document.createElement("style");
        style.textContent = `
            .${className}::-webkit-scrollbar {
                width: 10px;
            }
            .${className}::-webkit-scrollbar-track {
                background: transparent;
            }
            .${className}::-webkit-scrollbar-thumb {
                background: var(--scrollbar-dark);
                border-radius: 9999px;
            }
            .${className}::-webkit-scrollbar-thumb:hover {
                background: var(--scrollbar-dark);
            }
            `;
        document.head.appendChild(style);
    };
};

/*
    Null-safe version of Object.keys(obj)
*/
export const getKeys = <TObject extends object>(
    obj: TObject | null | undefined
) => {
    if (!obj) {
        console.warn("Getting keys for null or undefined object");
        return [];
    }
    return Object.keys(obj);
};

/*
    Null-safe version of Object.values(obj)
*/
export const getValues = <TObject extends object>(
    obj: TObject | null | undefined
) => {
    if (!obj) {
        console.warn("Getting values for null or undefined object");
        return [];
    }
    return Object.values(obj);
};
