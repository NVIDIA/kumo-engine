#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Buffer } from "node:buffer";
import { error } from "node:console";
import { readFileSync, writeFileSync } from "node:fs";
import process from "node:process";

export const DEFAULT_MAX_PR_BODY_BYTES = 60000;
export const OMITTED_SCREENSHOT_SECTION =
    "\n\n_Screenshot section omitted because the PR body is near GitHub's size limit. See screenshot artifacts._\n";
export const TRUNCATED_SCREENSHOT_SECTION_SUFFIX =
    "\n\n_Screenshot section truncated to keep the PR body under GitHub's size limit. See screenshot artifacts for full details._\n";

const byteLength = (value) => Buffer.byteLength(value, "utf8");

const VALIDATION_ITEMS = [
    ["type-check", "Type check"],
    ["lint-check", "Lint"],
    ["format-check", "Format"],
    ["targeted-tests", "Targeted tests"],
    ["related-tests", "Related tests"],
    ["build", "Build"]
];

const normalizeArtifactPayload = (value) => value?.payload ?? value ?? {};

const readJsonIfPresent = (path) => {
    try {
        return JSON.parse(readFileSync(path, "utf8"));
    } catch {
        return {};
    }
};

const isValidationCompletedItem = (item) =>
    /type-?check|lint-?check|format-?check|prettier|eslint|targeted tests|related tests|screenshot capture|workflow push|pr creation/i.test(
        item
    );

const cleanCompletedItem = (item) =>
    item
        .replace(/^\*\*([^*]+)\*\*\s*-\s*/, "$1: ")
        .replace(/^AC(\d+)\s*[-:]\s*/i, "AC$1: ")
        .trim();

const buildSummaryItems = (handoffPayload, issueTitle) => {
    const completed = Array.isArray(handoffPayload.completed)
        ? handoffPayload.completed
        : [];
    const items = completed
        .map((item) => String(item || "").trim())
        .filter(Boolean)
        .filter((item) => !isValidationCompletedItem(item))
        .filter((item) => !/data-testid/i.test(item))
        .map(cleanCompletedItem)
        .slice(0, 8);

    if (items.length > 0) {
        return items;
    }

    return issueTitle
        ? [`Implements the UI changes requested in "${issueTitle}".`]
        : ["Implements the requested UI changes."];
};

const formatStatus = (status) => {
    if (!status) {
        return "not reported";
    }
    return String(status).replaceAll("_", " ");
};

const buildValidationItems = (verificationPayload, screenshotStatus) => {
    const items = VALIDATION_ITEMS.map(([key, label]) => {
        const status = verificationPayload[key];
        if (!status) {
            return null;
        }
        return `${label}: ${formatStatus(status)}`;
    }).filter(Boolean);

    if (screenshotStatus) {
        items.push(`Headless screenshots: ${formatStatus(screenshotStatus)}`);
    }

    return items.length > 0 ? items : ["Validation status was not reported."];
};

export function buildReviewerPrBody({
    issueNumber,
    issueTitle,
    handoff,
    verification,
    runUrl,
    screenshotStatus
}) {
    const handoffPayload = normalizeArtifactPayload(handoff);
    const verificationPayload = normalizeArtifactPayload(verification);
    const summaryItems = buildSummaryItems(handoffPayload, issueTitle);
    const validationItems = buildValidationItems(
        verificationPayload,
        screenshotStatus
    );

    const body = [
        `Fixes #${issueNumber}`,
        "",
        "Design Spec: not applicable",
        "",
        "## Summary",
        "",
        ...summaryItems.map((item) => `- ${item}`),
        "",
        "## Testing Done",
        "",
        ...validationItems.map((item) => `- ${item}`),
        "",
        "## Hotfix Context",
        "",
        "Not applicable.",
        "",
        "## Breaking Changes",
        "",
        "None.",
        "",
        "## Documentation Update",
        "",
        "Not applicable unless noted in the summary.",
        "",
        "## Unit Tests",
        "",
        "See Testing Done.",
        "",
        "## Checklist",
        "",
        "- [x] I listed all the **breaking changes** introduced by the PR",
        "- [x] I made sure to **update examples and documentation** with my changes (if necessary)",
        "- [x] I have added and modified unit tests required to cover changes made by this PR.",
        "",
        "<details>",
        "<summary>Build-ui workflow details</summary>",
        "",
        runUrl ? `- Workflow run: ${runUrl}` : null,
        screenshotStatus
            ? `- Screenshot attempt status: ${formatStatus(screenshotStatus)}`
            : null,
        "- Artifact bundle: `build-ui-headless-artifacts`",
        "- Screenshot bundle: `build-ui-screenshots`",
        "- Useful files: `verification-status.json`, `handoff.json`, `screenshot-plan.json`, `screenshot-plan-resolution.json`, `screenshot-debug.md`",
        "",
        "</details>",
        ""
    ].filter((line) => line !== null);

    return `${body.join("\n")}\n`;
}

export function appendScreenshotSectionToPrBody({
    body,
    section,
    maxBytes = DEFAULT_MAX_PR_BODY_BYTES
}) {
    const prefixedSection = `\n${section}`;
    const available = maxBytes - byteLength(body);

    if (available <= 200) {
        return `${body}${OMITTED_SCREENSHOT_SECTION}`;
    }

    if (byteLength(prefixedSection) <= available) {
        return `${body}${prefixedSection}`;
    }

    const suffixBytes = byteLength(TRUNCATED_SCREENSHOT_SECTION_SUFFIX);
    const sectionBuffer = Buffer.from(prefixedSection, "utf8");
    const truncatedSection = sectionBuffer
        .subarray(0, Math.max(0, available - suffixBytes))
        .toString("utf8");

    return `${body}${truncatedSection}${TRUNCATED_SCREENSHOT_SECTION_SUFFIX}`;
}

function main() {
    const [, , commandOrBodyPath, ...args] = process.argv;
    if (commandOrBodyPath === "--base") {
        const [
            outputPath,
            issueNumber,
            issueJsonPath,
            handoffJsonPath,
            verificationJsonPath,
            runUrl,
            screenshotStatus
        ] = args;
        if (
            !outputPath ||
            !issueNumber ||
            !issueJsonPath ||
            !handoffJsonPath ||
            !verificationJsonPath
        ) {
            error(
                "Usage: build-ui-pr-body.mjs --base <output-path> <issue-number> <issue-json-path> <handoff-json-path> <verification-json-path> [run-url] [screenshot-status]"
            );
            process.exit(2);
        }

        const issue = readJsonIfPresent(issueJsonPath);
        writeFileSync(
            outputPath,
            buildReviewerPrBody({
                issueNumber,
                issueTitle: issue.title,
                handoff: readJsonIfPresent(handoffJsonPath),
                verification: readJsonIfPresent(verificationJsonPath),
                runUrl,
                screenshotStatus
            }),
            "utf8"
        );
        return;
    }

    const [bodyPath, sectionPath] = [commandOrBodyPath, ...args];
    if (!bodyPath || !sectionPath) {
        error(
            "Usage: build-ui-pr-body.mjs <body-path> <section-path>\n       build-ui-pr-body.mjs --base <output-path> <issue-number> <issue-json-path> <handoff-json-path> <verification-json-path> [run-url] [screenshot-status]"
        );
        process.exit(2);
    }

    const body = readFileSync(bodyPath, "utf8");
    const section = readFileSync(sectionPath, "utf8");
    writeFileSync(
        bodyPath,
        appendScreenshotSectionToPrBody({ body, section }),
        "utf8"
    );
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
