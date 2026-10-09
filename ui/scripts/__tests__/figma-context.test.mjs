// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import process from "node:process";

import { describe, expect, it } from "vitest";

const repoRoot = resolve(import.meta.dirname, "../../..");
const scriptPath = resolve(
    repoRoot,
    ".agents/skills/build-ui/scripts/extract-figma-context.py"
);

describe("extract-figma-context.py", () => {
    it("extracts Figma links from issue context without requiring a token", () => {
        const dir = mkdtempSync(resolve(tmpdir(), "build-ui-figma-"));
        const inputPath = resolve(dir, "issue.json");
        const outputPath = resolve(dir, "figma-context.json");

        writeFileSync(
            inputPath,
            `${JSON.stringify(
                {
                    issue: {
                        title: "Update model planner",
                        body: "Design: https://www.figma.com/design/abcDEF12345/Kumo?node-id=3997-3802&t=test"
                    },
                    comments: [
                        {
                            body: "Related file https://www.figma.com/file/Zyx98765432/Spec"
                        }
                    ]
                },
                null,
                2
            )}\n`
        );

        const output = execFileSync(
            "python3",
            [scriptPath, "--input-file", inputPath, "--output", outputPath],
            {
                env: {
                    ...process.env,
                    FIGMA_ACCESS_TOKEN: ""
                }
            }
        ).toString();
        const context = JSON.parse(readFileSync(outputPath, "utf8"));
        const summary = readFileSync(resolve(dir, "figma-summary.md"), "utf8");

        expect(output).toContain("skipped:");
        expect(context.artifactType).toBe("figma-context");
        expect(context.source).toBe("figma-rest");
        expect(context.status).toBe("skipped");
        expect(context.reason).toContain("FIGMA_ACCESS_TOKEN");
        expect(context.links).toEqual([
            expect.objectContaining({
                fileKey: "abcDEF12345",
                nodeId: "3997:3802"
            }),
            expect.objectContaining({
                fileKey: "Zyx98765432",
                nodeId: null
            })
        ]);
        expect(summary).toContain("# Figma Context");
        expect(summary).toContain("3997:3802");
    });

    it("normalizes markdown punctuation around Figma links", () => {
        const dir = mkdtempSync(resolve(tmpdir(), "build-ui-figma-"));
        const inputPath = resolve(dir, "issue.json");
        const outputPath = resolve(dir, "figma-context.json");

        writeFileSync(
            inputPath,
            `${JSON.stringify({
                issue: {
                    title: "Design",
                    body: "See (https://www.figma.com/design/abcDEF12345/Kumo?node-id=3997-3802)."
                },
                comments: []
            })}\n`
        );

        execFileSync(
            "python3",
            [scriptPath, "--input-file", inputPath, "--output", outputPath],
            {
                env: {
                    ...process.env,
                    FIGMA_ACCESS_TOKEN: ""
                }
            }
        );
        const context = JSON.parse(readFileSync(outputPath, "utf8"));

        expect(context.links).toEqual([
            expect.objectContaining({
                url: "https://www.figma.com/design/abcDEF12345/Kumo?node-id=3997-3802",
                fileKey: "abcDEF12345",
                nodeId: "3997:3802"
            })
        ]);
    });

    it("honors Figma file allowlists", () => {
        const dir = mkdtempSync(resolve(tmpdir(), "build-ui-figma-"));
        const inputPath = resolve(dir, "issue.json");
        const outputPath = resolve(dir, "figma-context.json");

        writeFileSync(
            inputPath,
            `${JSON.stringify({
                issue: {
                    title: "Design",
                    body: "Keep https://www.figma.com/design/allowed123/Kumo and drop https://www.figma.com/design/blocked456/Kumo"
                },
                comments: []
            })}\n`
        );

        execFileSync(
            "python3",
            [scriptPath, "--input-file", inputPath, "--output", outputPath],
            {
                env: {
                    ...process.env,
                    FIGMA_ACCESS_TOKEN: "",
                    FIGMA_FILE_ALLOWLIST: "allowed123"
                }
            }
        );
        const context = JSON.parse(readFileSync(outputPath, "utf8"));

        expect(context.links).toEqual([
            expect.objectContaining({
                fileKey: "allowed123"
            })
        ]);
    });

    it("writes a skipped context when explicitly skipped", () => {
        const dir = mkdtempSync(resolve(tmpdir(), "build-ui-figma-"));
        const outputPath = resolve(dir, "figma-context.json");

        execFileSync("python3", [
            scriptPath,
            "--skip",
            "--input-file",
            resolve(dir, "missing.json"),
            "--output",
            outputPath
        ]);
        const context = JSON.parse(readFileSync(outputPath, "utf8"));

        expect(context.status).toBe("skipped");
        expect(context.reason).toBe("skipped by --skip-figma");
        expect(context.links).toEqual([]);
    });

    it("treats missing node payloads as partial extraction errors without rendering images", () => {
        const repoRootForPython = JSON.stringify(repoRoot);
        const script = `
import importlib.util
import pathlib

repo_root = pathlib.Path(${repoRootForPython})
module_path = repo_root / ".agents/skills/build-ui/scripts/extract-figma-context.py"
spec = importlib.util.spec_from_file_location("extract_figma_context", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def fake_figma_api_json(path, token):
    if path.startswith("files/FILE123/nodes"):
        return {"nodes": {"1:2": None}}
    if path.startswith("images/FILE123"):
        raise AssertionError("image rendering should be skipped for missing nodes")
    raise AssertionError(path)

module.figma_api_json = fake_figma_api_json
_, nodes, images, errors = module.fetch_figma_artifacts(
    [{"fileKey": "FILE123", "nodeId": "1:2"}],
    pathlib.Path("/tmp/build-ui-figma-test/figma-context.json"),
    "token",
)
assert nodes == []
assert images == []
assert any(error.get("stage") == "node-fetch" for error in errors)
assert not any(error.get("stage") == "image-render" for error in errors)
`;

        execFileSync("python3", ["-c", script]);
    });

    it("uses branch keys as Figma API file identifiers for branch URLs", () => {
        const repoRootForPython = JSON.stringify(repoRoot);
        const script = `
import importlib.util
import pathlib

repo_root = pathlib.Path(${repoRootForPython})
module_path = repo_root / ".agents/skills/build-ui/scripts/extract-figma-context.py"
spec = importlib.util.spec_from_file_location("extract_figma_context", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

paths = []

def fake_figma_api_json(path, token):
    paths.append(path)
    if path.startswith("files/BRANCH123/nodes"):
        return {"nodes": {"1:2": {"document": {"name": "Frame", "type": "FRAME"}}}}
    if path.startswith("images/BRANCH123"):
        return {"images": {"1:2": None}}
    raise AssertionError(path)

module.figma_api_json = fake_figma_api_json
_, nodes, _, errors = module.fetch_figma_artifacts(
    [{"fileKey": "FILE123", "branchKey": "BRANCH123", "nodeId": "1:2"}],
    pathlib.Path("/tmp/build-ui-figma-test/figma-context.json"),
    "token",
)
assert any(path.startswith("files/BRANCH123/nodes") for path in paths)
assert any(path.startswith("images/BRANCH123") for path in paths)
assert nodes[0]["fileKey"] == "FILE123"
assert nodes[0]["branchKey"] == "BRANCH123"
assert nodes[0]["apiFileKey"] == "BRANCH123"
assert any(error.get("stage") == "image-render" for error in errors)
`;

        execFileSync("python3", ["-c", script]);
    });

    it("returns non-zero when extraction records failed status", () => {
        const dir = mkdtempSync(resolve(tmpdir(), "build-ui-figma-"));
        const outputPath = resolve(dir, "figma-context.json");

        const result = spawnSync(
            "python3",
            [
                scriptPath,
                "--input-file",
                resolve(dir, "missing.json"),
                "--output",
                outputPath
            ],
            {
                env: {
                    ...process.env,
                    FIGMA_ACCESS_TOKEN: ""
                },
                encoding: "utf8"
            }
        );
        const context = JSON.parse(readFileSync(outputPath, "utf8"));

        expect(result.status).toBe(1);
        expect(context.status).toBe("failed");
    });
});
