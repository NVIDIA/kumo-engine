#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

const DEFAULT_REPO = "kumo-ai/kumo";
const DEFAULT_BASE_URL = "http://localhost:3000";

const parseArgs = (argv) => {
    const args = {};
    for (let index = 2; index < argv.length; index += 1) {
        const key = argv[index];
        if (key === "--publish") {
            args.publish = true;
            continue;
        }
        if (key === "--skip-capture") {
            args.skipCapture = true;
            continue;
        }
        if (key === "--auth-disabled") {
            args.authDisabled = true;
            continue;
        }
        if (!key.startsWith("--")) {
            throw new Error(`Unexpected argument: ${key}`);
        }
        const value = argv[++index];
        if (!value || value.startsWith("--")) {
            throw new Error(`${key} requires a value`);
        }
        args[key.slice(2)] = value;
    }
    return args;
};

const defaultArtifactDir = (issue) => `/tmp/build-ui-artifacts/${issue}`;
const defaultScreenshotDir = (issue) => `/tmp/build-ui-screenshots/${issue}`;

export const resolveLocalScreenshotConfig = (rawArgs) => {
    const issue = rawArgs.issue;
    if (!issue) {
        throw new Error(
            "Usage: build-ui-local-screenshot.mjs --issue <number> [--plan <path> | --routes <routes>] [--publish --pr-number <number>] [--auth-disabled | --auth-email <email> --auth-password <pw>]"
        );
    }

    const artifactDir = rawArgs["artifact-dir"] || defaultArtifactDir(issue);
    const output = rawArgs.output || defaultScreenshotDir(issue);
    const defaultPlan = join(artifactDir, "screenshot-plan.json");
    const plan = rawArgs.plan || (existsSync(defaultPlan) ? defaultPlan : "");
    const routes = rawArgs.routes || "";
    const manifest = rawArgs.manifest || join(output, "results.json");
    const outputSection =
        rawArgs["output-section"] || join(artifactDir, "screenshot-section.md");

    if (!rawArgs.skipCapture && !plan && !routes) {
        throw new Error(
            `No screenshot plan found at ${defaultPlan}. Pass --plan <path> or --routes <comma-separated-routes>.`
        );
    }

    const baseUrl = rawArgs["base-url"] || DEFAULT_BASE_URL;
    const authEmail = rawArgs["auth-email"] || "";
    const authPassword = rawArgs["auth-password"] || "";
    // Real credentials passed as flags take precedence: when present we use the
    // validated auth path rather than the auth-disabled sentinel.
    const hasFlagCreds = Boolean(authEmail && authPassword);
    // Local dev servers run with auth disabled, so default localhost captures
    // to auth-disabled mode — this lets the same catalog recipe run locally
    // without a Direct Connect session. Explicit --auth-disabled forces it for any host;
    // flag credentials suppress the localhost default. ci-screenshot.mjs also
    // honors the backend's own auth_disabled flag and credential precedence.
    const isLocalhost = /(?:\/\/|^)(?:localhost|127\.0\.0\.1)(?::|\/|$)/.test(
        baseUrl
    );
    const authDisabled = Boolean(
        rawArgs.authDisabled || (isLocalhost && !hasFlagCreds)
    );

    return {
        issue,
        repo: rawArgs.repo || DEFAULT_REPO,
        prNumber: rawArgs["pr-number"] || "",
        baseUrl,
        authDisabled,
        authEmail,
        authPassword,
        artifactDir,
        output,
        plan,
        routes,
        manifest,
        resolution:
            rawArgs.resolution ||
            join(artifactDir, "screenshot-plan-resolution.json"),
        override:
            rawArgs.override || join(artifactDir, "screenshot-overrides.json"),
        outputSection,
        runUrl: rawArgs["run-url"] || "",
        publish: Boolean(rawArgs.publish || rawArgs["pr-number"]),
        skipCapture: Boolean(rawArgs.skipCapture)
    };
};

export const buildCaptureArgs = (config) => {
    const args = ["scripts/ci-screenshot.mjs", "--output", config.output];
    if (config.plan) {
        args.push("--plan", config.plan);
    } else {
        args.push("--routes", config.routes);
    }
    if (config.baseUrl) {
        args.push("--base-url", config.baseUrl);
    }
    if (config.authDisabled) {
        args.push("--auth-disabled");
    }
    return args;
};

// Credentials are forwarded to ci-screenshot.mjs via the environment rather
// than argv, so they never reach the (longer-running) child capture process's
// command line. Note this does NOT hide a --auth-password value passed to THIS
// wrapper: that still appears in the wrapper's own argv (and `ps`) for its
// lifetime. To avoid argv exposure entirely, export SCREENSHOT_AUTH_EMAIL /
// SCREENSHOT_AUTH_PASSWORD instead of using the flags. Only defined values are
// included, leaving any inherited env vars untouched.
export const buildCaptureEnv = (config) => {
    const env = {};
    if (config.authEmail) {
        env.SCREENSHOT_AUTH_EMAIL = config.authEmail;
    }
    if (config.authPassword) {
        env.SCREENSHOT_AUTH_PASSWORD = config.authPassword;
    }
    return env;
};

export const buildPublishArgs = (config) => {
    const args = [
        "scripts/build-ui-publish-screenshots.mjs",
        "--repo",
        config.repo,
        "--issue",
        config.issue,
        "--manifest",
        config.manifest,
        "--resolution",
        config.resolution,
        "--override",
        config.override,
        "--output-section",
        config.outputSection
    ];

    if (config.runUrl) {
        args.push("--run-url", config.runUrl);
    }

    if (config.prNumber) {
        args.push(
            "--pr-number",
            config.prNumber,
            "--patch-pr",
            "--comment-type",
            "pr",
            "--comment-target",
            config.prNumber
        );
    } else {
        args.push("--comment-type", "issue", "--comment-target", config.issue);
    }

    return args;
};

const runNode = (args, extraEnv) => {
    execFileSync("node", args, {
        stdio: "inherit",
        env: { ...process.env, ...(extraEnv || {}) }
    });
};

function main() {
    const config = resolveLocalScreenshotConfig(parseArgs(process.argv));

    if (!config.skipCapture) {
        runNode(buildCaptureArgs(config), buildCaptureEnv(config));
    }

    if (config.publish) {
        if (!existsSync(config.manifest)) {
            throw new Error(
                `Cannot publish screenshots because manifest is missing: ${config.manifest}`
            );
        }
        runNode(buildPublishArgs(config));
    }
}

if (import.meta.url === `file://${process.argv[1]}`) {
    try {
        main();
    } catch (error) {
        console.error(error.message || error);
        process.exit(1);
    }
}
