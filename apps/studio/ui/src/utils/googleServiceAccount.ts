// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface ParsedGoogleServiceAccount {
    project_id: string;
    private_key_id: string;
    private_key: string;
    client_email: string;
    client_id: string;
    auth_uri: string;
    token_uri: string;
    universe_domain: string;
}

// Fields the identity API requires. They are read out of the uploaded file so
// the request can be built at all -- the raw JSON is never uploaded. Whether
// their values are acceptable is decided by the server, which validates the
// service-account email, the canonical endpoints and the private key, and then
// mints a token to prove the key works.
const requiredFields = [
    "project_id",
    "private_key_id",
    "private_key",
    "client_email",
    "client_id",
    "auth_uri",
    "token_uri"
] as const;

const DEFAULT_UNIVERSE_DOMAIN = "googleapis.com";

const readGoogleServiceAccountDocument = (
    text: string
): ParsedGoogleServiceAccount => {
    const parsed = JSON.parse(text) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("invalid service-account JSON");
    }
    const document = parsed as Record<string, unknown>;
    const values: Record<string, string> = {};
    for (const field of requiredFields) {
        const value = document[field];
        if (typeof value !== "string" || !value.trim()) {
            throw new Error("incomplete service-account key");
        }
        values[field] = value;
    }
    const universeDomain = document.universe_domain;
    values.universe_domain =
        typeof universeDomain === "string" && universeDomain.trim()
            ? universeDomain.trim()
            : DEFAULT_UNIVERSE_DOMAIN;
    return values as unknown as ParsedGoogleServiceAccount;
};

export const parseGoogleServiceAccount = (
    text: string
): ParsedGoogleServiceAccount => {
    try {
        return readGoogleServiceAccountDocument(text);
    } catch {
        throw new Error("Invalid Google service-account JSON key.");
    }
};
