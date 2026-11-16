// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
    buildResolutionSummary,
    buildScreenshotSection,
    coverageSummary,
    mergeScreenshotSectionIntoPrBody,
    resultTier
} from "../build-ui-publish-screenshots.mjs";

const tempPng = (name) => {
    const dir = mkdtempSync(join(tmpdir(), "build-ui-publish-test-"));
    const filePath = join(dir, name);
    writeFileSync(filePath, "png");
    return filePath;
};

describe("build-ui screenshot publisher", () => {
    it("formats coverage statuses", () => {
        expect(
            coverageSummary({
                coverage: [
                    { id: "route_loaded", passed: true },
                    { id: "optional_panel", skipped: true },
                    { id: "missing_copy", passed: false }
                ]
            })
        ).toBe(
            "passed:route_loaded, skipped:optional_panel, missing:missing_copy"
        );
    });

    it("selects primary screenshots from explicit coverage overrides", () => {
        const result = {
            success: true,
            coverage: [{ id: "settings_panel", passed: true }]
        };

        expect(
            resultTier({
                result,
                override: { coverageIds: ["settings_panel"] },
                successes: [result]
            })
        ).toBe("primary");

        expect(
            resultTier({
                result,
                override: { coverageIds: ["schema_panel"] },
                successes: [result]
            })
        ).toBe("regression");
    });

    it("builds a screenshot section and uploads only primary screenshots", async () => {
        const uploaded = [];
        const primaryPath = tempPng("new-prediction-config-layout.png");
        const regressionPath = tempPng("new-training-page.png");
        const section = await buildScreenshotSection({
            repo: "kumo-ai/kumo",
            issue: "28440",
            runUrl: "https://github.com/kumo-ai/kumo/actions/runs/1",
            timestamp: 123,
            random: () => 456,
            resolution: {
                action: "merged-primary",
                selectedRecipe: "multi",
                matchedRecipes: [{ key: "new_prediction" }]
            },
            manifest: {
                results: [
                    {
                        success: true,
                        presentation: "primary",
                        route: "/newjob/prediction/",
                        filePath: primaryPath,
                        matchedSelector:
                            '[data-testid="new-prediction-config-layout"]',
                        finalUrl: "http://localhost:3000/newjob/prediction/",
                        coverage: [{ id: "config_visible", passed: true }]
                    },
                    {
                        success: true,
                        presentation: "regression",
                        route: "/newjob/training/",
                        filePath: regressionPath,
                        finalUrl: "http://localhost:3000/newjob/training/"
                    },
                    {
                        success: true,
                        skipped: true,
                        scenarioName: "optional-settings",
                        route: "/settings/",
                        skipReason: "feature disabled"
                    },
                    {
                        success: false,
                        route: "/graphs/",
                        error: "ready selector missing",
                        finalUrl: "http://localhost:3000/graphs/",
                        failureType: "selector-readiness"
                    }
                ]
            },
            uploadImage: async (input) => {
                uploaded.push(input);
                return `https://example.test/${input.filename}`;
            }
        });

        expect(uploaded).toHaveLength(1);
        expect(uploaded[0]).toMatchObject({
            filePath: primaryPath,
            filename: "new-prediction-config-layout-123-456.png",
            repo: "kumo-ai/kumo",
            issue: "28440"
        });
        expect(section).toContain("## Screenshots");
        expect(section).toContain("### new-prediction-config-layout");
        expect(section).toContain(
            '<img width="1280" alt="new-prediction-config-layout" src="https://example.test/new-prediction-config-layout-123-456.png" />'
        );
        expect(section).toContain(
            "Regression screenshot coverage (1 captured, 1 skipped)"
        );
        expect(section).toContain(
            "- `new-training-page` on `/newjob/training/`"
        );
        expect(section).toContain(
            "- `/graphs/`: ready selector missing (final URL: `http://localhost:3000/graphs/`, metadata: `n/a`, type: `selector-readiness`)"
        );
        expect(section).toContain("Plan resolution: action `merged-primary`");
    });

    it("summarizes catalog resolution", () => {
        expect(
            buildResolutionSummary({
                action: "wrote",
                selectedRecipe: "new_prediction",
                matchedRecipes: [
                    { key: "new_prediction" },
                    { key: "new_training" }
                ]
            })
        ).toBe(
            "Plan resolution: action `wrote`, selected recipe `new_prediction`, matched recipes `new_prediction, new_training`. See `screenshot-plan-resolution.json` in the build-ui artifact bundle."
        );
    });

    it("replaces an existing screenshot section and preserves the rest of the PR body", () => {
        const body =
            "## Screenshots\nold\n## Summary\n- Changed prediction page\n";
        const result = mergeScreenshotSectionIntoPrBody({
            body,
            section: "## Screenshots\nnew\n"
        });

        expect(result).toContain("## Screenshots\nnew\n");
        expect(result).toContain("## Summary\n- Changed prediction page");
        expect(result).not.toContain("old");
    });
});
