// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
    buildCaptureArgs,
    buildCaptureEnv,
    buildPublishArgs,
    resolveLocalScreenshotConfig
} from "../build-ui-local-screenshot.mjs";

describe("build-ui local screenshot helper", () => {
    it("builds capture args from explicit routes and auto-enables auth-disabled for localhost", () => {
        const config = resolveLocalScreenshotConfig({
            issue: "123",
            routes: "/connectors/",
            "base-url": "http://localhost:3000"
        });

        expect(config.authDisabled).toBe(true);
        expect(buildCaptureArgs(config)).toEqual([
            "scripts/ci-screenshot.mjs",
            "--output",
            "/tmp/build-ui-screenshots/123",
            "--routes",
            "/connectors/",
            "--base-url",
            "http://localhost:3000",
            "--auth-disabled"
        ]);
    });

    it("does not enable auth-disabled for a non-localhost base url unless requested", () => {
        const config = resolveLocalScreenshotConfig({
            issue: "123",
            routes: "/connectors/",
            "base-url": "https://staging.example.test"
        });

        expect(config.authDisabled).toBe(false);
        expect(buildCaptureArgs(config)).not.toContain("--auth-disabled");
    });

    it("forces auth-disabled for any host when --auth-disabled is passed", () => {
        const config = resolveLocalScreenshotConfig({
            issue: "123",
            routes: "/connectors/",
            "base-url": "https://staging.example.test",
            authDisabled: true
        });

        expect(config.authDisabled).toBe(true);
        expect(buildCaptureArgs(config)).toContain("--auth-disabled");
    });

    it("forwards Direct Connect email/password credentials via env and skips the localhost default", () => {
        const config = resolveLocalScreenshotConfig({
            issue: "123",
            routes: "/connectors/",
            "base-url": "http://localhost:3000",
            "auth-email": "shots@kumo.ai",
            "auth-password": "secret"
        });

        expect(config.authDisabled).toBe(false);
        expect(buildCaptureEnv(config)).toEqual({
            SCREENSHOT_AUTH_EMAIL: "shots@kumo.ai",
            SCREENSHOT_AUTH_PASSWORD: "secret"
        });
    });

    it("emits no credential env when none are provided", () => {
        const config = resolveLocalScreenshotConfig({
            issue: "123",
            routes: "/connectors/",
            "base-url": "http://localhost:3000"
        });

        expect(buildCaptureEnv(config)).toEqual({});
    });

    it("defaults publication to the PR when pr-number is provided", () => {
        const config = resolveLocalScreenshotConfig({
            issue: "123",
            routes: "/connectors/",
            "pr-number": "456"
        });

        expect(config.publish).toBe(true);
        expect(buildPublishArgs(config)).toEqual([
            "scripts/build-ui-publish-screenshots.mjs",
            "--repo",
            "kumo-ai/kumo",
            "--issue",
            "123",
            "--manifest",
            "/tmp/build-ui-screenshots/123/results.json",
            "--resolution",
            "/tmp/build-ui-artifacts/123/screenshot-plan-resolution.json",
            "--override",
            "/tmp/build-ui-artifacts/123/screenshot-overrides.json",
            "--output-section",
            "/tmp/build-ui-artifacts/123/screenshot-section.md",
            "--pr-number",
            "456",
            "--patch-pr",
            "--comment-type",
            "pr",
            "--comment-target",
            "456"
        ]);
    });
});
