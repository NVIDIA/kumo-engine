#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from "node:child_process";
import { randomInt } from "node:crypto";
import { basename, join } from "node:path";
import {
    existsSync,
    mkdtempSync,
    readFileSync,
    rmSync,
    writeFileSync
} from "node:fs";
import { tmpdir } from "node:os";
import process from "node:process";

import { appendScreenshotSectionToPrBody } from "./build-ui-pr-body.mjs";

const DEFAULT_IMAGES_BRANCH = "images";
const DEFAULT_BASE_BRANCH = "master";
const ensuredBranches = new Set();

const readJsonIfPresent = (filePath) => {
    if (!filePath || !existsSync(filePath)) {
        return {};
    }
    return JSON.parse(readFileSync(filePath, "utf8"));
};

const fileStem = (filePath) => basename(filePath, ".png");

export const coverageSummary = (result) => {
    const coverage = result.coverage || [];
    if (!coverage.length) {
        return "";
    }
    return coverage
        .map((item) => {
            const status = item.skipped
                ? "skipped"
                : item.passed
                  ? "passed"
                  : "missing";
            return `${status}:${item.id || "unknown"}`;
        })
        .join(", ");
};

const coverageIds = (result) =>
    new Set((result.coverage || []).map((item) => item.id).filter(Boolean));

export const resultTier = ({
    result,
    resolution = {},
    override = {},
    successes = []
}) => {
    if (["primary", "regression"].includes(result.presentation)) {
        return result.presentation;
    }

    const selectedRecipe = resolution.selectedRecipe;
    const explicitCoverage = new Set(override.coverageIds || []);
    const hasExplicitScreenshotIntent = Boolean(
        override.recipeKey ||
        override.route ||
        explicitCoverage.size > 0 ||
        override.fixtures
    );

    if (hasExplicitScreenshotIntent) {
        if (explicitCoverage.size > 0) {
            const resultCoverage = coverageIds(result);
            return [...explicitCoverage].some((id) => resultCoverage.has(id))
                ? "primary"
                : "regression";
        }
        return "primary";
    }

    if (selectedRecipe && selectedRecipe !== "multi") {
        return "primary";
    }

    return successes.length <= 3 ? "primary" : "regression";
};

export const buildResolutionSummary = (resolution = {}) => {
    if (!Object.keys(resolution).length) {
        return "";
    }

    const matched =
        (resolution.matchedRecipes || [])
            .map((recipe) => recipe.key)
            .filter(Boolean)
            .join(", ") || "none";
    return (
        `Plan resolution: action \`${resolution.action || "unknown"}\`, ` +
        `selected recipe \`${resolution.selectedRecipe || "none"}\`, ` +
        `matched recipes \`${matched}\`. ` +
        "See `screenshot-plan-resolution.json` in the build-ui artifact bundle."
    );
};

const markdownLine = (value) => `${value}\n`;

export async function buildScreenshotSection({
    manifest,
    resolution = {},
    override = {},
    repo,
    issue,
    runUrl,
    uploadImage,
    timestamp = Math.floor(Date.now() / 1000),
    random = () => randomInt(10000, 99999)
}) {
    const successes = (manifest.results || []).filter(
        (result) => result.success && !result.skipped
    );
    const primary = [];
    const regression = [];
    const skipped = [];
    const failures = [];

    for (const result of manifest.results || []) {
        const coverage = coverageSummary(result);
        if (result.success && result.skipped) {
            skipped.push({
                route: result.route,
                name: result.scenarioName || result.route,
                reason: result.skipReason || "",
                coverage
            });
            continue;
        }

        if (result.success) {
            const tier = resultTier({
                result,
                resolution,
                override,
                successes
            });
            if (tier === "primary") {
                const name = fileStem(result.filePath);
                const filename = `${name}-${timestamp}-${random()}.png`;
                const url = await uploadImage({
                    filePath: result.filePath,
                    filename,
                    repo,
                    issue
                });
                primary.push({
                    name,
                    route: result.route,
                    selector: result.matchedSelector || "",
                    finalUrl: result.finalUrl || "",
                    coverage,
                    url
                });
            } else {
                regression.push({
                    name: fileStem(result.filePath),
                    route: result.route,
                    finalUrl: result.finalUrl || "",
                    coverage
                });
            }
            continue;
        }

        failures.push({
            route: result.route,
            error: result.error || "",
            metadataPath: result.metadataPath || "",
            finalUrl: result.finalUrl || "",
            coverage,
            failureType: result.failureType || "unknown"
        });
    }

    if (
        primary.length === 0 &&
        regression.length === 0 &&
        skipped.length === 0 &&
        failures.length === 0
    ) {
        return "";
    }

    let body = "";
    body += markdownLine("## Screenshots");
    body += markdownLine("");
    body += markdownLine(
        `Visual evidence captured by build-ui${
            runUrl ? ` ([workflow run](${runUrl}))` : ""
        }.`
    );
    body += markdownLine("");
    body += markdownLine("### Primary Evidence");

    if (primary.length === 0) {
        body += markdownLine(
            "_No primary evidence screenshots were selected for inline publication. Review the uploaded screenshot artifact bundle for regression captures._"
        );
        body += markdownLine("");
    }

    for (const item of primary) {
        body += markdownLine(`### ${item.name}`);
        body += markdownLine(`Route: \`${item.route}\``);
        body += markdownLine(`Matched selector: \`${item.selector}\``);
        body += markdownLine(`Final URL: \`${item.finalUrl}\``);
        if (item.coverage) {
            body += markdownLine(`Coverage: \`${item.coverage}\``);
        }
        body += markdownLine(
            `<img width="1280" alt="${item.name}" src="${item.url}" />`
        );
        body += markdownLine("");
    }

    if (regression.length > 0 || skipped.length > 0) {
        body += markdownLine("<details>");
        body += markdownLine(
            `<summary>Regression screenshot coverage (${regression.length} captured, ${skipped.length} skipped)</summary>`
        );
        body += markdownLine("");
        for (const item of regression) {
            const coverage = item.coverage
                ? `, coverage: \`${item.coverage}\``
                : "";
            body += markdownLine(
                `- \`${item.name}\` on \`${item.route}\` (final URL: \`${item.finalUrl || "unknown"}\`${coverage})`
            );
        }
        if (skipped.length > 0) {
            body += markdownLine("");
            body += markdownLine("Skipped conditional scenarios:");
            for (const item of skipped) {
                const coverage = item.coverage
                    ? `, coverage: \`${item.coverage}\``
                    : "";
                body += markdownLine(
                    `- \`${item.name}\` on \`${item.route}\`: ${item.reason}${coverage}`
                );
            }
        }
        body += markdownLine("</details>");
        body += markdownLine("");
    }

    const resolutionSummary = buildResolutionSummary(resolution);
    if (resolutionSummary) {
        body += markdownLine("<details>");
        body += markdownLine("<summary>Screenshot plan details</summary>");
        body += markdownLine("");
        body += markdownLine(resolutionSummary);
        body += markdownLine("");
        body += markdownLine("</details>");
        body += markdownLine("");
    }

    if (failures.length > 0) {
        body += markdownLine("### Failed Routes");
        for (const item of failures) {
            const coverage = item.coverage
                ? `, coverage: \`${item.coverage}\``
                : "";
            const failureType = item.failureType
                ? `, type: \`${item.failureType}\``
                : "";
            body += markdownLine(
                `- \`${item.route}\`: ${item.error} (final URL: \`${item.finalUrl || "unknown"}\`, metadata: \`${item.metadataPath || "n/a"}\`${failureType}${coverage})`
            );
        }
        body += markdownLine("");
    }

    body += markdownLine("---");
    body += "*build-ui agent*";
    return body;
}

const gh = (args, options = {}) =>
    execFileSync("gh", args, {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
        ...options
    }).trim();

const ensureImagesBranch = ({ repo, branch, baseBranch }) => {
    const cacheKey = `${repo}:${branch}:${baseBranch}`;
    if (ensuredBranches.has(cacheKey)) {
        return;
    }

    try {
        gh(["api", `repos/${repo}/git/ref/heads/${branch}`]);
        ensuredBranches.add(cacheKey);
        return;
    } catch {
        const baseSha = gh([
            "api",
            `repos/${repo}/git/ref/heads/${baseBranch}`,
            "--jq",
            ".object.sha"
        ]);
        gh([
            "api",
            "-X",
            "POST",
            `repos/${repo}/git/refs`,
            "-f",
            `ref=refs/heads/${branch}`,
            "-f",
            `sha=${baseSha}`
        ]);
        ensuredBranches.add(cacheKey);
    }
};

export const uploadImageToGitHub = ({
    filePath,
    filename,
    repo,
    issue,
    branch = DEFAULT_IMAGES_BRANCH,
    baseBranch = DEFAULT_BASE_BRANCH
}) => {
    ensureImagesBranch({ repo, branch, baseBranch });

    const tempDir = mkdtempSync(join(tmpdir(), "build-ui-screenshot-"));
    const payloadPath = join(tempDir, "upload-payload.json");
    try {
        const content = readFileSync(filePath).toString("base64");
        writeFileSync(
            payloadPath,
            JSON.stringify({
                message: `Screenshot from build-ui #${issue}`,
                content,
                branch
            }),
            "utf8"
        );

        const commitSha = gh([
            "api",
            "-X",
            "PUT",
            `repos/${repo}/contents/screenshots/${filename}`,
            "--input",
            payloadPath,
            "--jq",
            ".commit.sha"
        ]);
        return `https://github.com/${repo}/blob/${commitSha}/screenshots/${filename}?raw=true`;
    } finally {
        rmSync(tempDir, { recursive: true, force: true });
    }
};

export const mergeScreenshotSectionIntoPrBody = ({ body, section }) => {
    const pattern = /(^|\n)## Screenshots\n.*?(?=\n## |$)/s;
    const baseBody = pattern.test(body)
        ? body.replace(pattern, "$1").trimEnd()
        : body.trimEnd();

    return appendScreenshotSectionToPrBody({
        body: baseBody ? `${baseBody}\n` : "",
        section
    });
};

export const patchPrBodyWithSection = ({ repo, prNumber, section }) => {
    const currentBody = gh([
        "pr",
        "view",
        String(prNumber),
        "--repo",
        repo,
        "--json",
        "body",
        "--jq",
        ".body"
    ]);
    const updated = mergeScreenshotSectionIntoPrBody({
        body: currentBody,
        section
    });

    const tempDir = mkdtempSync(join(tmpdir(), "build-ui-pr-body-"));
    const payloadPath = join(tempDir, "payload.json");
    try {
        writeFileSync(payloadPath, JSON.stringify({ body: updated }), "utf8");
        gh([
            "api",
            "-X",
            "PATCH",
            `repos/${repo}/pulls/${prNumber}`,
            "--input",
            payloadPath
        ]);
    } finally {
        rmSync(tempDir, { recursive: true, force: true });
    }
};

export const commentWithSection = ({ repo, type, target, sectionPath }) => {
    execFileSync(
        "gh",
        [type, "comment", String(target), "--repo", repo, "-F", sectionPath],
        { stdio: "inherit" }
    );
};

const parseArgs = (argv) => {
    const args = {};
    for (let index = 2; index < argv.length; index += 1) {
        const key = argv[index];
        if (key === "--patch-pr") {
            args.patchPr = true;
            continue;
        }
        if (!key.startsWith("--")) {
            throw new Error(`Unexpected argument: ${key}`);
        }
        args[key.slice(2)] = argv[++index];
    }
    return args;
};

async function main() {
    const args = parseArgs(process.argv);
    const repo = args.repo;
    const issue = args.issue;
    const manifestPath = args.manifest;
    const outputSectionPath = args["output-section"];

    if (!repo || !issue || !manifestPath || !outputSectionPath) {
        throw new Error(
            "Usage: build-ui-publish-screenshots.mjs --repo <owner/repo> --issue <number> --manifest <results.json> --output-section <path> [--resolution <path>] [--override <path>] [--run-url <url>] [--pr-number <number> --patch-pr] [--comment-type pr|issue --comment-target <number>]"
        );
    }

    const branch = args.branch || DEFAULT_IMAGES_BRANCH;
    const baseBranch = args["base-branch"] || DEFAULT_BASE_BRANCH;
    const manifest = readJsonIfPresent(manifestPath);
    const resolution = readJsonIfPresent(args.resolution);
    const override = readJsonIfPresent(args.override);

    const section = await buildScreenshotSection({
        manifest,
        resolution,
        override,
        repo,
        issue,
        runUrl: args["run-url"],
        uploadImage: (input) =>
            uploadImageToGitHub({
                ...input,
                branch,
                baseBranch
            })
    });

    if (!section) {
        return;
    }

    writeFileSync(outputSectionPath, `${section}\n`, "utf8");

    if (args.patchPr && args["pr-number"]) {
        patchPrBodyWithSection({
            repo,
            prNumber: args["pr-number"],
            section
        });
    }

    if (args["comment-type"] && args["comment-target"]) {
        commentWithSection({
            repo,
            type: args["comment-type"],
            target: args["comment-target"],
            sectionPath: outputSectionPath
        });
    }
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch((err) => {
        console.error(err.message || err);
        process.exit(1);
    });
}
