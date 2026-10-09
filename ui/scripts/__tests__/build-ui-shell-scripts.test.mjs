// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import process from "node:process";

import { describe, expect, it } from "vitest";

const repoRoot = resolve(import.meta.dirname, "../../..");
const skillScript = (name) =>
    resolve(repoRoot, ".agents/skills/build-ui/scripts", name);
const workflowPath = resolve(repoRoot, ".github/workflows/build-ui-agent.yml");

describe("build-ui shell scripts", () => {
    it("writes valid JSON artifacts for escaped targets", () => {
        const output = join(
            mkdtempSync(join(tmpdir(), "build-ui-artifact-test-")),
            "artifact.json"
        );
        const target = 'issue-"quoted"\\slash\nnext-line';

        execFileSync("bash", [
            skillScript("write-headless-artifact.sh"),
            "--type",
            "blocker",
            "--target",
            target,
            "--output",
            output
        ]);

        const artifact = JSON.parse(readFileSync(output, "utf8"));
        expect(artifact).toMatchObject({
            artifactType: "blocker",
            headless: true,
            target,
            payload: {}
        });
    });

    it("checks UI runtime config from the repo root", () => {
        const output = execFileSync("bash", [
            skillScript("check-preflight.sh")
        ]).toString();

        expect(output).toContain("runtime-config\tunavailable");
    });

    it("can emit not-needed preflight statuses", () => {
        const output = execFileSync(
            "bash",
            [skillScript("check-preflight.sh")],
            {
                env: {
                    ...process.env,
                    BUILD_UI_BROWSER_NOT_NEEDED: "true"
                }
            }
        ).toString();

        expect(output).toContain("browser-tooling\tnot-needed");
    });

    it("uses normalized issue number for summarize REST fallback", () => {
        const source = readFileSync(skillScript("summarize-issue.sh"), "utf8");

        expect(source).toContain('issue_number="$issue"');
        expect(source).toContain("/issues/([0-9]+)");
        expect(source).toContain("issues/${issue_number}/comments");
        expect(source).not.toContain("issues/${issue}/comments");
        expect(source).not.toContain('comments="[]"');
    });

    it("keeps validators usable without ripgrep", () => {
        for (const script of [
            "validate-skill-structure.sh",
            "validate-workflow-behavior.sh",
            "validate-output-schemas.sh"
        ]) {
            const source = readFileSync(skillScript(script), "utf8");

            expect(source).toContain("command -v rg");
            expect(source).toContain("command -v grep");
        }
    });

    it("keeps build-ui helper scripts executable", () => {
        for (const script of [
            "check-preflight.sh",
            "ensure-screenshot-plan.py",
            "extract-figma-context.py",
            "summarize-issue.sh",
            "update-headless-artifact.sh",
            "validate-output-schemas.sh",
            "validate-skill-structure.sh",
            "validate-workflow-behavior.sh",
            "write-headless-artifact.sh"
        ]) {
            expect(statSync(skillScript(script)).mode & 0o111, script).not.toBe(
                0
            );
        }
    });

    it("keeps headless dry-run and skip-screenshots wired into workflow gates", () => {
        const workflow = readFileSync(workflowPath, "utf8");

        expect(workflow).toContain("--dry-run");
        expect(workflow).toContain("DRY_RUN=true");
        expect(workflow).toContain('"$DRY_RUN" != "true"');
        expect(workflow).toContain('"$DRY_RUN" == "true"');
        expect(workflow).toContain("dry_run=$DRY_RUN");
        expect(workflow).toContain("--skip-screenshots");
        expect(workflow).toContain("--mode(=|[[:space:]]+)");
        expect(workflow).toContain("SKIP_SCREENSHOTS=true");
        expect(workflow).toContain('"$SKIP_SCREENSHOTS" != "true"');
        expect(workflow).toContain(
            "steps.screenshot_mode.outputs.skip_screenshots != 'true'"
        );
    });

    it("prevents runner tooling overlays from being staged into ticket PRs", () => {
        const workflow = readFileSync(workflowPath, "utf8");

        expect(workflow).toContain("BUILD_UI_RUNNER_TOOLING_FILES");
        expect(workflow).toContain('git restore --staged -- "$file"');
        expect(workflow).toContain(
            "Only workflow runner tooling changed; skipping implementation commit."
        );
    });

    it("persists complete phase zero evidence in headless workflow", () => {
        const workflow = readFileSync(workflowPath, "utf8");

        expect(workflow).toContain("gh-auth-status.txt");
        expect(workflow).toContain("git-status-porcelain.txt");
        expect(workflow).toContain("gh auth status");
        expect(workflow).toContain("git status --porcelain=v1");
        expect(workflow).toContain("ghAuthStatus");
        expect(workflow).toContain("gitStatusPorcelain");
    });

    it("preserves plan-level screenshot feature flags during workflow normalization", () => {
        const workflow = readFileSync(workflowPath, "utf8");

        expect(workflow).toContain("_requiredFeatureFlags");
        expect(workflow).toContain(
            "Screenshot plan _requiredFeatureFlags must be an object"
        );
        expect(workflow).toContain('metadata["_requiredFeatureFlags"]');
    });
});
