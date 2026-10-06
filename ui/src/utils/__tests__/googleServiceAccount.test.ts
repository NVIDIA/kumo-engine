// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { parseGoogleServiceAccount } from "../googleServiceAccount";

const validKey = {
    type: "service_account",
    project_id: "identity-project",
    private_key_id: "key-id-1",
    private_key:
        "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----\n",
    client_email: "kumo@identity-project.iam.gserviceaccount.com",
    client_id: "1234567890",
    auth_uri: "https://accounts.google.com/o/oauth2/auth",
    token_uri: "https://oauth2.googleapis.com/token"
};

describe("parseGoogleServiceAccount", () => {
    it("returns the complete decomposed key and defaults the universe", () => {
        expect(parseGoogleServiceAccount(JSON.stringify(validKey))).toEqual({
            project_id: validKey.project_id,
            private_key_id: validKey.private_key_id,
            private_key: validKey.private_key,
            client_email: validKey.client_email,
            client_id: validKey.client_id,
            auth_uri: validKey.auth_uri,
            token_uri: validKey.token_uri,
            universe_domain: "googleapis.com"
        });
    });

    it("keeps an explicit universe domain", () => {
        expect(
            parseGoogleServiceAccount(
                JSON.stringify({
                    ...validKey,
                    universe_domain: " example.com "
                })
            ).universe_domain
        ).toBe("example.com");
    });

    // The server decides whether a value is acceptable: it validates the
    // service-account email, the canonical endpoints and the private key, then
    // mints a token to prove the key works. Reproducing those rules here only
    // creates a second, drifting gate, so these documents are read as-is and
    // their verdict is left to the API.
    it.each([
        [
            "an App Engine default service account",
            { client_email: "identity-project@appspot.gserviceaccount.com" }
        ],
        [
            "a Compute Engine default service account",
            {
                client_email:
                    "847355853533-compute@developer.gserviceaccount.com"
            }
        ],
        ["a non-canonical auth URI", { auth_uri: "https://example.com/auth" }],
        ["a private key without PEM markers", { private_key: "secret" }],
        ["a non-service-account type", { type: "authorized_user" }]
    ])("defers %s to the server", (_label, override) => {
        const document = { ...validKey, ...override };
        const parsed = parseGoogleServiceAccount(JSON.stringify(document));
        for (const [field, value] of Object.entries(override)) {
            if (field === "type") continue;
            expect(parsed[field as keyof typeof parsed]).toBe(value);
        }
    });

    it.each([
        ["malformed JSON", "{"],
        ["a JSON array", "[]"],
        ["a JSON scalar", '"service-account"'],
        ["missing metadata", { ...validKey, client_id: "" }],
        ["an absent field", { ...validKey, private_key: undefined }]
    ])("rejects %s", (_label, value) => {
        const text = typeof value === "string" ? value : JSON.stringify(value);
        expect(() => parseGoogleServiceAccount(text)).toThrow(
            "Invalid Google service-account JSON key."
        );
    });
});
