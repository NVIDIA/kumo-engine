// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    getFromLocalStorage,
    getFromSessionStorage,
    storeToLocalStorage,
    storeToSessionStorage
} from "@kumo-ai/kumo-ui-catalog";

import { storageKeys } from "./Constants";

export { getFromSessionStorage, storeToSessionStorage };

export function clearLocalStorage(): void {
    // Taking backup of navPath before clearing local storage
    // and storing it to local storage again if the path is valid
    try {
        const path = getFromLocalStorage(storageKeys.navPath);
        const debug = getFromLocalStorage("DEBUG");

        localStorage.clear();

        if (isValidForStorage(path)) {
            storeToLocalStorage(storageKeys.navPath, path);
        }
        if (isValidForStorage(debug)) {
            storeToLocalStorage("DEBUG", debug);
        }
    } catch (error) {
        console.error("Unable to clearLocalStorage", error);
    }
}

export function inDebugMode(): string | null {
    return getFromLocalStorage("DEBUG");
}

export function getProfileMode(): string {
    return getFromSessionStorage("PROFILE_MODE") ?? "DISABLED";
}

export function getReactQueryDevToolMode(): string {
    return getFromSessionStorage("REACT_QUERY_DEV_TOOL_MODE") ?? "DISABLED";
}

function isValidForStorage<T>(data: T | undefined | null): boolean {
    return (
        data !== undefined && data !== null && data?.toString()?.trim() !== ""
    );
}

export function getSelectedGroup(): string | null {
    return getFromLocalStorage(storageKeys.selectedGroup);
}

export function getStoredSelectedProjectId(): string | null {
    return getFromLocalStorage<string>(storageKeys.selectedProject);
}

export function removeFromSessionStorage(key: string): void {
    try {
        sessionStorage.removeItem(key);
    } catch (error) {
        console.error("Unable to removeFromSessionStorage", error);
    }
}
