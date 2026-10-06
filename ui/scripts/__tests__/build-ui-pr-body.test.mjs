// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Buffer } from "node:buffer";

import { describe, expect, it } from "vitest";

import {
    appendScreenshotSectionToPrBody,
    buildReviewerPrBody,
    OMITTED_SCREENSHOT_SECTION,
    TRUNCATED_SCREENSHOT_SECTION_SUFFIX
} from "../build-ui-pr-body.mjs";

describe("build-ui PR body helpers", () => {
    it("appends the full screenshot section when it fits", () => {
        const body = "Closes #1\n";
        const section = "## Screenshot\n![screenshot](url)\n";

        expect(
            appendScreenshotSectionToPrBody({
                body,
                section,
                maxBytes: 1000
            })
        ).toBe(`${body}\n${section}`);
    });

    it("truncates the screenshot section when only part of it fits", () => {
        const body = "A".repeat(100);
        const section = "B".repeat(1000);
        const result = appendScreenshotSectionToPrBody({
            body,
            section,
            maxBytes: 500
        });

        expect(result).toContain(TRUNCATED_SCREENSHOT_SECTION_SUFFIX);
        expect(result).toMatch(/^A{100}\nB+/);
        expect(Buffer.byteLength(result, "utf8")).toBeLessThanOrEqual(500);
    });

    it("omits the screenshot section when the base body is near the limit", () => {
        const body = "A".repeat(350);
        const result = appendScreenshotSectionToPrBody({
            body,
            section: "B".repeat(1000),
            maxBytes: 500
        });

        expect(result).toBe(`${body}${OMITTED_SCREENSHOT_SECTION}`);
        expect(result).not.toContain("B");
    });

    it("builds a reviewer-facing PR body from handoff and verification artifacts", () => {
        const body = buildReviewerPrBody({
            issueNumber: "28440",
            issueTitle: "Align New Prediction Page Layout with New Model Page",
            handoff: {
                payload: {
                    completed: [
                        "AC1: Added viewport-aware layout to New Prediction",
                        "AC2: Moved overflow into Settings and Schema panels",
                        "Added data-testid attributes for screenshots",
                        "TypeScript type-check: passed"
                    ]
                }
            },
            verification: {
                payload: {
                    "type-check": "passed",
                    "lint-check": "passed",
                    "format-check": "passed",
                    build: "not-run",
                    "targeted-tests": "passed"
                }
            },
            runUrl: "https://github.com/kumo-ai/kumo/actions/runs/1",
            screenshotStatus: "captured"
        });

        expect(body).toContain("Fixes #28440");
        expect(body).toContain("Design Spec: not applicable");
        expect(body).toContain("## Summary");
        expect(body).toContain(
            "- AC1: Added viewport-aware layout to New Prediction"
        );
        expect(body).toContain(
            "- AC2: Moved overflow into Settings and Schema panels"
        );
        expect(body).not.toContain("data-testid");
        expect(body).not.toContain("TypeScript type-check");
        expect(body).toContain("## Testing Done");
        expect(body).toContain("- Type check: passed");
        expect(body).toContain("- Headless screenshots: captured");
        expect(body).toContain("## Hotfix Context");
        expect(body).toContain("## Breaking Changes");
        expect(body).toContain("## Documentation Update");
        expect(body).toContain("## Unit Tests");
        expect(body).toContain("## Checklist");
        expect(body).toContain("None.");
        expect(body).toContain("Not applicable unless noted in the summary.");
        expect(body).toContain("See Testing Done.");
        expect(body).toContain("<summary>Build-ui workflow details</summary>");
    });
});
