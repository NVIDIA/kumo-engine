// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Method to sort the given two values.
 *
 * @param {object} a first value
 * @param {object} b second value
 * @param {string} key name of the field to be sorted in the given object
 */
export function customSort<TObject extends object>(
    a: { original: TObject },
    b: { original: TObject },
    key: string
): number {
    try {
        if (
            a?.original?.[key as keyof typeof a.original] &&
            b?.original?.[key as keyof typeof b.original]
        )
            return a?.original[key as keyof typeof a.original] >
                b?.original[key as keyof typeof b.original]
                ? -1
                : 1;
        else return 0;
    } catch (error) {
        console.warn("custom sort error", error);
        return 0;
    }
}

/**
 * Method to sort the given two values with support for nested keys.
 *
 * @param {object} a first value
 * @param {object} b second value
 * @param {string} key name of the field to be sorted in the given object (supports dot notation)
 */
export function dynamicCustomSort<TObject extends object>(
    a: { original: TObject },
    b: { original: TObject },
    key: string
): number {
    const resolveKey = (
        obj: Record<string, unknown> | undefined,
        keyPath: string
    ): unknown => {
        return keyPath?.split(".")?.reduce<unknown>((acc, part) => {
            if (acc && typeof acc === "object" && part in acc) {
                return (acc as Record<string, unknown>)[part];
            }
            return undefined;
        }, obj);
    };

    try {
        const aValue = resolveKey(a?.original as Record<string, unknown>, key);
        const bValue = resolveKey(b?.original as Record<string, unknown>, key);

        if (aValue === undefined || bValue === undefined) {
            return 0;
        }

        if (typeof aValue === "number" && typeof bValue === "number") {
            if (aValue < bValue) {
                return -1;
            }
            if (aValue > bValue) {
                return 1;
            }
        }
        if (typeof aValue === "string" && typeof bValue === "string") {
            if (aValue > bValue) {
                return -1;
            }
            if (aValue < bValue) {
                return 1;
            }
        }

        return 0;
    } catch (error) {
        console.warn("Custom sort error:", error);
        return 0;
    }
}
