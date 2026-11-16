// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Creating deep copy of the given data.
 *
 * This will convert the json object into string and
 * again converts back to json object and returning it
 * @param {object} data
 * @returns `deep copied object`
 */
export function copyObject<T>(data: T): T {
    try {
        return JSON.parse(JSON.stringify(data));
    } catch (error) {
        console.error("Unable to copyObject", error);
        return data;
    }
}

/**
 * Null-safe version of Object.keys(obj)
 */
export const getKeys = <TObject extends object>(
    obj: TObject | null | undefined
) => {
    if (!obj) {
        return [];
    }
    return Object.keys(obj);
};

/**
 * Null-safe version of Object.values(obj)
 */
export const getValues = <TObject extends object>(
    obj: TObject | null | undefined
) => {
    if (!obj) {
        return [];
    }
    return Object.values(obj);
};

export const isObjectEmpty = <TObject extends object>(
    obj: TObject | null | undefined
): boolean => {
    return (
        typeof obj === "object" &&
        obj !== null &&
        obj !== undefined &&
        getKeys(obj)?.length === 0
    );
};

export const isObjectNotEmpty = <TObject extends object>(
    obj: TObject | null | undefined
): boolean => {
    return (
        typeof obj === "object" &&
        obj !== null &&
        obj !== undefined &&
        getKeys(obj)?.length > 0
    );
};
