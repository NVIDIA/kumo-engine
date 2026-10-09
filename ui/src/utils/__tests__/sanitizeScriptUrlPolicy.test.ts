// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";

/**
 * Regression tests for kumo-ai/kumo#30121: the Trusted-Types
 * "sanitizeScriptURL" policy bootstrap moved from an inline <script> in
 * ui/index.html into the bundled module utils/sanitizeScriptUrlPolicy so
 * the delivery CSP script-src needs no inline hash. These tests pin the
 * ported behavior to the removed inline original: same policy name, a
 * single createScriptURL handler delegating to DOMPurify.sanitize with
 * RETURN_TRUSTED_TYPE, installed exactly once, and a no-op on browsers
 * without Trusted Types.
 */

const POLICY_MODULE = "utils/sanitizeScriptUrlPolicy";

type CreatePolicyMock = ReturnType<typeof vi.fn>;

type WindowWithTrustedTypes = Window & {
    trustedTypes?: { createPolicy?: CreatePolicyMock };
};

const windowWithTrustedTypes = window as unknown as WindowWithTrustedTypes;

type PolicyRules = Record<string, (input: string) => unknown>;

function installTrustedTypesMock(): CreatePolicyMock {
    // Returns a functional policy object because DOMPurify also creates
    // its own internal policy through window.trustedTypes when sanitize
    // runs with RETURN_TRUSTED_TYPE.
    const createPolicy = vi.fn((_name: string, rules?: PolicyRules) => ({
        createHTML: (input: string) =>
            rules?.createHTML ? rules.createHTML(input) : input,
        createScript: (input: string) =>
            rules?.createScript ? rules.createScript(input) : input,
        createScriptURL: (input: string) =>
            rules?.createScriptURL ? rules.createScriptURL(input) : input
    }));

    windowWithTrustedTypes.trustedTypes = { createPolicy };

    return createPolicy;
}

afterEach(() => {
    delete windowWithTrustedTypes.trustedTypes;
    vi.restoreAllMocks();
    vi.resetModules();
});

describe("sanitizeScriptUrlPolicy bootstrap", () => {
    it("installs the sanitizeScriptURL policy exactly once", async () => {
        vi.resetModules();
        const createPolicy = installTrustedTypesMock();

        await import(POLICY_MODULE);

        expect(createPolicy).toHaveBeenCalledTimes(1);
        expect(createPolicy).toHaveBeenCalledWith(
            "sanitizeScriptURL",
            expect.objectContaining({
                createScriptURL: expect.any(Function)
            })
        );

        // The ported policy exposes exactly the surface the inline
        // index.html original did: a single createScriptURL handler.
        const [, policyOptions] = createPolicy.mock.calls[0] as [
            string,
            Record<string, unknown>
        ];

        expect(Object.keys(policyOptions)).toEqual(["createScriptURL"]);

        // A repeated import re-uses the evaluated module and must not
        // attempt to install a duplicate policy (createPolicy throws on
        // duplicate names in real browsers).
        await import(POLICY_MODULE);

        expect(createPolicy).toHaveBeenCalledTimes(1);
    });

    it("delegates createScriptURL to DOMPurify.sanitize with RETURN_TRUSTED_TYPE", async () => {
        vi.resetModules();
        const createPolicy = installTrustedTypesMock();

        await import(POLICY_MODULE);

        const [, policyOptions] = createPolicy.mock.calls[0] as [
            string,
            { createScriptURL: (input: string) => unknown }
        ];
        const { default: DOMPurify } = await import("dompurify");
        const sanitize = vi.spyOn(DOMPurify, "sanitize");

        const maliciousUrl =
            "javascript:alert(1)//<img src=x onerror=alert(1)>";
        const result = policyOptions.createScriptURL(maliciousUrl);

        expect(sanitize).toHaveBeenCalledTimes(1);
        expect(sanitize).toHaveBeenCalledWith(maliciousUrl, {
            RETURN_TRUSTED_TYPE: true
        });
        expect(result).toBe(sanitize.mock.results[0].value);
    });

    it("is a no-op on browsers without Trusted Types support", async () => {
        vi.resetModules();
        delete windowWithTrustedTypes.trustedTypes;

        await expect(import(POLICY_MODULE)).resolves.toBeDefined();

        vi.resetModules();
        windowWithTrustedTypes.trustedTypes = {};

        await expect(import(POLICY_MODULE)).resolves.toBeDefined();
    });
});
