// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "fs";
import path from "path";
import { describe, expect, it } from "vitest";

/**
 * Tests that serve.json is correctly placed and contains the required
 * security headers for pentest compliance (M1 — CVSS 5.3).
 *
 * serve.json MUST live in public/ so that Vite copies it into build/.
 * The `serve -s build` command reads config from the served directory,
 * so build/serve.json is the only location that takes effect.
 */
describe("serve.json security headers config", () => {
    const serveJsonPath = path.resolve(__dirname, "../../../public/serve.json");
    const config = JSON.parse(fs.readFileSync(serveJsonPath, "utf-8"));

    it("should exist in public/ directory", () => {
        expect(fs.existsSync(serveJsonPath)).toBe(true);
    });

    it("should be valid JSON with a headers array", () => {
        expect(config).toHaveProperty("headers");
        expect(Array.isArray(config.headers)).toBe(true);
        expect(config.headers.length).toBeGreaterThan(0);
    });

    it("should apply headers to all routes via ** source", () => {
        const rule = config.headers[0];
        expect(rule.source).toBe("**");
    });

    const requiredHeaders: Record<string, string> = {
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
        "X-Frame-Options": "SAMEORIGIN",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
    };

    const headers = config.headers[0].headers as {
        key: string;
        value: string;
    }[];

    it.each(Object.entries(requiredHeaders))(
        "should include %s header with value %s",
        (key, value) => {
            const found = headers.find(
                (h: { key: string; value: string }) => h.key === key
            );
            expect(found).toBeDefined();
            expect(found!.value).toBe(value);
        }
    );

    it("should NOT exist at the old (broken) location ui/serve.json", () => {
        const oldPath = path.resolve(__dirname, "../../../serve.json");
        expect(fs.existsSync(oldPath)).toBe(false);
    });
});
