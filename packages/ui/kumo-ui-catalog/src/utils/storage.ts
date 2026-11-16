// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as b64 from "base64-js";

export function storeToSessionStorage(key: string, value: string): void {
    // Since we silently fail, session storage should NOT be used to support
    // any core flows
    try {
        sessionStorage.setItem(key, JSON.stringify(value));
    } catch (err) {
        console.error(err);
        console.error("Unable to save data to session storage");
    }
}

export function getFromSessionStorage<T>(key: string): T | null {
    try {
        const item = sessionStorage.getItem(key);
        return item ? (JSON.parse(item) as T) : null;
    } catch (error) {
        console.error(error);
        console.error("Unable to getFromSessionStorage");
        return null;
    }
}

export function clearSessionStorage(): void {
    try {
        sessionStorage.clear();
    } catch (error) {
        console.error(error);
        console.error("Unable to clearSessionStorage");
    }
}

export function storeToLocalStorage<T>(
    key: string,
    value: T,
    encode: boolean = false
): void {
    try {
        let data = value;
        if (encode) {
            data = encodeBase64(value as unknown as string) as unknown as T;
        }
        localStorage.setItem(key, JSON.stringify(data));
    } catch (error) {
        console.error(error);
        console.error("Unable to storeToLocalStorage");
    }
}

export function getFromLocalStorage<T>(
    key: string,
    decode: boolean = false
): T | null {
    try {
        let data = JSON.parse(localStorage.getItem(key) as string);

        if (decode && typeof data === "string") {
            data = decodeBase64(data) as unknown as T;
        }

        return data as T | null;
    } catch (error) {
        console.error(
            `Error getting item from localStorage with key "${key}":`,
            error
        );
        return null;
    }
}

export function removeFromLocalStorage(key: string): void {
    try {
        localStorage.removeItem(key);
    } catch (error) {
        console.error("Unable to removeFromLocalStorage", error);
    }
}

export function encodeBase64(data: string): string {
    try {
        const utf8Encode = new TextEncoder();
        return b64.fromByteArray(utf8Encode.encode(data));
    } catch (error) {
        console.error("Unable to encodeBase64", error);
        return data;
    }
}

export function decodeBase64(data: string): string {
    try {
        const byteArray = b64.toByteArray(data);
        const decoded = String.fromCharCode(...byteArray);
        return decoded;
    } catch (error) {
        console.error("Unable to decodeBase64", error);
        return data;
    }
}
