// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { navRoutes } from "utils/Constants";

const isLoginPath = (path: string) =>
    path === "/" ||
    path.startsWith("/login") ||
    path.startsWith("/legacy-login") ||
    path.startsWith("/logout");

const hasUnsafeReturnPathCharacter = (path: string) => {
    for (const char of path) {
        const code = char.charCodeAt(0);
        if (code <= 31 || code === 127 || char === "\\") return true;
    }
    return false;
};

const safeRelativePath = (path?: string | null) => {
    if (!path || hasUnsafeReturnPathCharacter(path)) return null;
    try {
        const url = new URL(path, window.location.origin);
        if (url.origin !== window.location.origin) return null;
        return `${url.pathname}${url.search}`;
    } catch {
        return null;
    }
};

export const getDirectConnectReturnTo = (
    storedPath?: string | null,
    currentPath?: string | null
) => {
    const safeStoredPath = safeRelativePath(storedPath);
    if (safeStoredPath && !isLoginPath(safeStoredPath)) return safeStoredPath;

    const safeCurrentPath = safeRelativePath(currentPath);
    if (safeCurrentPath && !isLoginPath(safeCurrentPath))
        return safeCurrentPath;

    return navRoutes.Home;
};
