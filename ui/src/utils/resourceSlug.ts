// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const RESOURCE_SLUG_PATTERN =
    /^[a-z](?:[a-z0-9]|-(?=[a-z0-9])){1,46}[a-z0-9]$/;

export const RESERVED_RESOURCE_SLUGS = new Set([
    "ambient-s3",
    "api",
    "by-id",
    "by-slug",
    "default",
    "new",
    "settings",
    "system"
]);

const isAsciiWhitespaceOrPunctuation = (character: string): boolean => {
    const code = character.charCodeAt(0);
    return (
        (code >= 9 && code <= 13) ||
        (code >= 32 && code <= 47) ||
        (code >= 58 && code <= 64) ||
        (code >= 91 && code <= 96) ||
        (code >= 123 && code <= 126)
    );
};

export const deriveResourceSlug = (displayName: string): string => {
    const normalized = displayName.normalize("NFKC").toLowerCase();
    const pieces: string[] = [];
    let pendingSeparator = false;

    for (const character of normalized) {
        if (isAsciiWhitespaceOrPunctuation(character)) {
            pendingSeparator = pieces.length > 0;
            continue;
        }
        if (pendingSeparator) {
            pieces.push("-");
            pendingSeparator = false;
        }
        pieces.push(character);
    }

    return pieces.join("").replace(/^-+|-+$/g, "");
};

export const validateResourceSlug = (slug: string): string => {
    if (!RESOURCE_SLUG_PATTERN.test(slug)) {
        return "ID must be 3–48 characters, start with a letter, and contain only lowercase letters, numbers, and single hyphens.";
    }
    if (RESERVED_RESOURCE_SLUGS.has(slug)) {
        return `ID “${slug}” is reserved. Choose another ID.`;
    }
    return "";
};

export const resourceSlugError = (
    explicitSlug: string,
    displayName: string
): string => {
    const candidate = explicitSlug || deriveResourceSlug(displayName);
    return candidate ? validateResourceSlug(candidate) : "";
};
