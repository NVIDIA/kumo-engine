// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * CSP guard for the production build output (kumo-ai/kumo#30121).
 *
 * Deployments front the UI with a Content-Security-Policy whose script-src
 * carries no inline hash/nonce ('self' plus explicit hosts only). That is
 * only sound while the shipped index.html contains ZERO inline <script>
 * bodies; a single inline script silently breaks under that CSP and forces
 * a hash that churns whenever the snippet changes.
 *
 * This check runs against the vite build output (ui/build) and fails when:
 *   1. build/index.html contains any inline <script> (no src attribute, or
 *      a non-empty element body), or
 *   2. the entry chunk referenced by build/index.html does not install the
 *      Trusted-Types "sanitizeScriptURL" policy bootstrap BEFORE the main
 *      application code. Rollup emits module statements in evaluation
 *      order, so marker positions in the chunk reflect execution order.
 *
 * Wired into `npm run build` so image/CI builds enforce the contract.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";

/** Trusted-Types bootstrap marker (src/utils/sanitizeScriptUrlPolicy.ts). */
export const POLICY_MARKER = "sanitizeScriptURL";

/** Marker from the top of the main application module (src/main.tsx). */
export const APP_MARKER = "__ciScreenshotAuth";

/**
 * Return every <script> occurrence in the given HTML that would require a
 * CSP inline allowance: script tags without a src attribute, or tags that
 * carry a src but still have a non-empty element body.
 */
export function findInlineScripts(html) {
    // The end-tag pattern must cover spec-legal variants like "</script >"
    // or "</script foo>"; a filter that only matches "</script>" can be
    // bypassed (CodeQL js/bad-tag-filter).
    const scriptRe = /<script\b([^>]*)>([\s\S]*?)<\/script\b[^>]*>/gi;
    const offenders = [];
    let match;

    while ((match = scriptRe.exec(html)) !== null) {
        const [, attrs, body] = match;
        const hasSrc = /\bsrc\s*=/i.test(attrs);

        if (!hasSrc || body.trim() !== "") {
            offenders.push({
                attrs: attrs.trim(),
                body: body.trim(),
                reason: hasSrc
                    ? "script tag with src also carries an inline body"
                    : "inline script tag (no src attribute)"
            });
        }
    }

    return offenders;
}

/** Return the src of every <script type="module" src=...> in the HTML. */
export function findModuleScriptSrcs(html) {
    const scriptRe = /<script\b([^>]*)>/gi;
    const srcs = [];
    let match;

    while ((match = scriptRe.exec(html)) !== null) {
        const attrs = match[1];

        if (!/\btype\s*=\s*["']?module["']?/i.test(attrs)) {
            continue;
        }

        const srcMatch = /\bsrc\s*=\s*["']([^"']+)["']/i.exec(attrs);

        if (srcMatch) {
            srcs.push(srcMatch[1]);
        }
    }

    return srcs;
}

/**
 * Verify the Trusted-Types policy bootstrap is part of the given chunk and
 * evaluates before the main application code.
 */
export function checkPolicyBeforeApp(
    chunkSource,
    { policyMarker = POLICY_MARKER, appMarker = APP_MARKER } = {}
) {
    const policyIndex = chunkSource.indexOf(policyMarker);
    const appIndex = chunkSource.indexOf(appMarker);

    return {
        policyIndex,
        appIndex,
        ok: policyIndex !== -1 && appIndex !== -1 && policyIndex < appIndex
    };
}

/**
 * Run the full check against a build directory. Returns a list of failure
 * messages; empty list means the build satisfies the CSP contract.
 */
export function checkBuildDir(buildDir) {
    const failures = [];
    const indexPath = path.join(buildDir, "index.html");

    if (!fs.existsSync(indexPath)) {
        return [`missing ${indexPath}; run the vite build first`];
    }

    const html = fs.readFileSync(indexPath, "utf8");

    for (const offender of findInlineScripts(html)) {
        failures.push(
            `${offender.reason} in ${indexPath}: ` +
                `<script ${offender.attrs}>${offender.body.slice(0, 120)}...`
        );
    }

    const moduleSrcs = findModuleScriptSrcs(html);

    if (moduleSrcs.length === 0) {
        failures.push(
            `no <script type="module" src=...> found in ${indexPath}`
        );
        return failures;
    }

    const entrySrc = moduleSrcs[0];
    const entryPath = path.join(buildDir, entrySrc.replace(/^\//, ""));

    if (!fs.existsSync(entryPath)) {
        failures.push(`entry chunk ${entryPath} (from ${entrySrc}) not found`);
        return failures;
    }

    const chunk = fs.readFileSync(entryPath, "utf8");
    const order = checkPolicyBeforeApp(chunk);

    if (order.policyIndex === -1) {
        failures.push(
            `entry chunk ${entrySrc} does not contain the Trusted-Types ` +
                `bootstrap marker "${POLICY_MARKER}"`
        );
    } else if (order.appIndex === -1) {
        failures.push(
            `entry chunk ${entrySrc} does not contain the app marker ` +
                `"${APP_MARKER}"; update APP_MARKER if src/main.tsx changed`
        );
    } else if (!order.ok) {
        failures.push(
            `Trusted-Types bootstrap (offset ${order.policyIndex}) does not ` +
                `precede the app code (offset ${order.appIndex}) in ${entrySrc}`
        );
    }

    return failures;
}

function main() {
    const scriptDir = path.dirname(fileURLToPath(import.meta.url));
    const buildDir = process.argv[2] || path.resolve(scriptDir, "..", "build");
    const failures = checkBuildDir(buildDir);

    if (failures.length > 0) {
        console.error("check-built-index-csp: FAIL");

        for (const failure of failures) {
            console.error(`  - ${failure}`);
        }

        process.exit(1);
    }

    console.log(
        `check-built-index-csp: PASS (${buildDir}: no inline scripts; ` +
            `Trusted-Types bootstrap precedes app code in the entry chunk)`
    );
}

if (
    process.argv[1] &&
    import.meta.url === pathToFileURL(process.argv[1]).href
) {
    try {
        main();
    } catch (error) {
        console.error(error.message || error);
        process.exit(1);
    }
}
