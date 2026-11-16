// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const directConnectErrorMessage = (
    error: unknown,
    fallback: string
): string => {
    if (!(error instanceof Error) || !error.message) return fallback;
    try {
        const parsed = JSON.parse(error.message) as { detail?: unknown };
        return typeof parsed.detail === "string" ? parsed.detail : fallback;
    } catch {
        return fallback;
    }
};
