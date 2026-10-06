// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { afterEach, describe, expect, it } from "vitest";

import {
    APP_MARKER,
    checkBuildDir,
    checkPolicyBeforeApp,
    findInlineScripts,
    findModuleScriptSrcs,
    POLICY_MARKER
} from "../check-built-index-csp.mjs";

const scriptsDir = path.dirname(fileURLToPath(import.meta.url));
const uiDir = path.resolve(scriptsDir, "..", "..");

const tempDirs = [];

function makeBuildDir({ html, chunks = {} }) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "csp-check-"));

    tempDirs.push(dir);
    fs.mkdirSync(path.join(dir, "assets"), { recursive: true });
    fs.writeFileSync(path.join(dir, "index.html"), html);

    for (const [name, source] of Object.entries(chunks)) {
        fs.writeFileSync(path.join(dir, "assets", name), source);
    }

    return dir;
}

afterEach(() => {
    while (tempDirs.length > 0) {
        fs.rmSync(tempDirs.pop(), { recursive: true, force: true });
    }
});

describe("findInlineScripts", () => {
    it("flags a script tag without src", () => {
        const offenders = findInlineScripts(
            "<html><body><script>doEvil()</script></body></html>"
        );

        expect(offenders).toHaveLength(1);
        expect(offenders[0].reason).toContain("no src attribute");
    });

    it("flags a src script tag that also carries an inline body", () => {
        const offenders = findInlineScripts(
            '<script src="/a.js">inlineToo()</script>'
        );

        expect(offenders).toHaveLength(1);
        expect(offenders[0].reason).toContain("also carries an inline body");
    });

    it("flags an inline script closed with a spec-legal end-tag variant", () => {
        const offenders = findInlineScripts(
            "<html><body><script>doEvil()</script ></body></html>"
        );

        expect(offenders).toHaveLength(1);
        expect(offenders[0].reason).toContain("no src attribute");
    });

    it("accepts src-only script tags", () => {
        const offenders = findInlineScripts(
            '<script src="/env-config.js"></script>' +
                '<script type="module" crossorigin src="/assets/index-abc.js"></script>'
        );

        expect(offenders).toEqual([]);
    });
});

describe("findModuleScriptSrcs", () => {
    it("returns module script srcs and ignores classic scripts", () => {
        const srcs = findModuleScriptSrcs(
            '<script src="/env-config.js"></script>' +
                '<script type="module" crossorigin src="/assets/index-abc.js"></script>'
        );

        expect(srcs).toEqual(["/assets/index-abc.js"]);
    });
});

describe("checkPolicyBeforeApp", () => {
    it("passes when the policy marker precedes the app marker", () => {
        const result = checkPolicyBeforeApp(
            `${POLICY_MARKER}(); ${APP_MARKER}();`
        );

        expect(result.ok).toBe(true);
    });

    it("fails when the policy marker is missing or comes after the app", () => {
        expect(checkPolicyBeforeApp(`${APP_MARKER}();`).ok).toBe(false);
        expect(
            checkPolicyBeforeApp(`${APP_MARKER}(); ${POLICY_MARKER}();`).ok
        ).toBe(false);
    });
});

describe("checkBuildDir", () => {
    const goodHtml =
        "<html><head>" +
        '<script src="/env-config.js"></script>' +
        '<script type="module" crossorigin src="/assets/index-abc.js"></script>' +
        "</head><body></body></html>";

    it("passes a build with no inline scripts and policy-first entry", () => {
        const dir = makeBuildDir({
            html: goodHtml,
            chunks: {
                "index-abc.js": `${POLICY_MARKER}(); ${APP_MARKER}();`
            }
        });

        expect(checkBuildDir(dir)).toEqual([]);
    });

    it("fails a build whose index.html carries an inline script", () => {
        const dir = makeBuildDir({
            html: goodHtml.replace("<body>", "<body><script>inline()</script>"),
            chunks: {
                "index-abc.js": `${POLICY_MARKER}(); ${APP_MARKER}();`
            }
        });

        const failures = checkBuildDir(dir);

        expect(failures).toHaveLength(1);
        expect(failures[0]).toContain("inline script tag");
    });

    it("fails a build whose entry chunk lacks the policy bootstrap", () => {
        const dir = makeBuildDir({
            html: goodHtml,
            chunks: { "index-abc.js": `${APP_MARKER}();` }
        });

        const failures = checkBuildDir(dir);

        expect(failures).toHaveLength(1);
        expect(failures[0]).toContain(POLICY_MARKER);
    });
});

describe("shipped ui/index.html", () => {
    it("needs no CSP inline script allowance (kumo-ai/kumo#30121)", () => {
        const html = fs.readFileSync(path.join(uiDir, "index.html"), "utf8");

        expect(findInlineScripts(html)).toEqual([]);
    });
});
