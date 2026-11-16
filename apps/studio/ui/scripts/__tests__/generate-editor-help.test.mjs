// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawnSync } from "node:child_process";
import {
    cpSync,
    mkdtempSync,
    readFileSync,
    rmSync,
    symlinkSync,
    writeFileSync
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { expect, it } from "vitest";

it("regenerates offline and rejects stale output, invalid source and arguments without writing", () => {
    const ui = process.cwd();
    const root = mkdtempSync(path.join(tmpdir(), "editor-help-"));
    try {
        for (const file of [
            "scripts/generate-editor-help.mjs",
            "editor-hints/editor-hints.json",
            "src/components/CodeEditor/editorHintRegistry.ts",
            "src/components/CodeEditor/tooltipKeys.ts",
            "src/types/codeEditor.ts",
            "src/utils/PQLParsing/pqDocs.json",
            ".prettierrc"
        ])
            cpSync(path.join(ui, file), path.join(root, file), {
                recursive: true
            });
        symlinkSync(
            path.join(ui, "node_modules"),
            path.join(root, "node_modules")
        );
        const run = (...args) =>
            spawnSync(
                process.execPath,
                [path.join(root, "scripts/generate-editor-help.mjs"), ...args],
                {
                    encoding: "utf8",
                    env: {
                        ...process.env,
                        HTTP_PROXY: "http://127.0.0.1:1",
                        HTTPS_PROXY: "http://127.0.0.1:1"
                    }
                }
            );
        const output = path.join(root, "src/utils/PQLParsing/pqDocs.json");
        const original = readFileSync(output, "utf8");
        expect(run("--check").status).toBe(0);
        writeFileSync(output, "stale\n");
        expect(run("--check").stderr).toContain("Editor help is stale");
        expect(readFileSync(output, "utf8")).toBe("stale\n");
        expect(run().status).toBe(0);
        expect(readFileSync(output, "utf8")).toBe(original);
        expect(run().status).toBe(0);
        expect(readFileSync(output, "utf8")).toBe(original);
        expect(run("--unknown").stderr).toContain("Usage:");
        writeFileSync(
            path.join(root, "editor-hints/editor-hints.json"),
            '{"schemaVersion":99}'
        );
        expect(run().status).not.toBe(0);
        expect(readFileSync(output, "utf8")).toBe(original);
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
}, 20000);
