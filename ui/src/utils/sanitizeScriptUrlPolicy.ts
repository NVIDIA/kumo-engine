// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Trusted-Types "sanitizeScriptURL" policy bootstrap.
 *
 * This used to live as an inline <script> in ui/index.html, which forced
 * the delivery CSP to carry a script-src sha256 hash that broke whenever
 * the snippet changed (kumo-ai/kumo#30121). Bundling it keeps index.html
 * free of inline scripts, so script-src needs no inline hash/nonce.
 *
 * It must stay the FIRST import of the application entry (src/entry.ts)
 * so the policy exists before any consumer can run.
 * scripts/check-built-index-csp.mjs enforces that ordering on the built
 * entry chunk during `npm run build`.
 *
 * The behavior is a faithful port of the inline original: same policy
 * name, a single createScriptURL handler delegating to DOMPurify.sanitize
 * with RETURN_TRUSTED_TYPE, and the policy handle deliberately not
 * retained. The only delta is that DOMPurify now comes from the bundled
 * npm package instead of the vendored /purify.min.js global, which this
 * module replaced.
 */
import DOMPurify from "dompurify";

// TypeScript's lib.dom does not model the Trusted Types API yet.
type TrustedTypesWindow = Window & {
    trustedTypes?: {
        createPolicy?: (
            name: string,
            rules: { createScriptURL: (input: string) => unknown }
        ) => unknown;
    };
};

const trustedTypes = (window as TrustedTypesWindow).trustedTypes;

if (trustedTypes && trustedTypes.createPolicy) {
    trustedTypes.createPolicy("sanitizeScriptURL", {
        createScriptURL: (input) =>
            DOMPurify.sanitize(input, { RETURN_TRUSTED_TYPE: true })
    });
}
