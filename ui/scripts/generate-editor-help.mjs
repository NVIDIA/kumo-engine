// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { createJiti } from "jiti";
import { format, resolveConfig } from "prettier";

const jiti = createJiti(import.meta.url);
const { parseRegistry } = await jiti.import(
    "../src/components/CodeEditor/editorHintRegistry.ts"
);
const source = new URL("../editor-hints/editor-hints.json", import.meta.url);
const output = new URL("../src/utils/PQLParsing/pqDocs.json", import.meta.url);
const args = process.argv.slice(2);
if (args.length > 1 || (args.length === 1 && args[0] !== "--check")) {
    throw new Error("Usage: generate-editor-help.mjs [--check]");
}
const registry = parseRegistry(JSON.parse(await readFile(source, "utf8")));
for (const entries of [registry.hints, registry.noHints]) {
    entries.sort((a, b) => {
        const left = `${a.language}:${a.key}`;
        const right = `${b.language}:${b.key}`;
        return left < right ? -1 : left > right ? 1 : 0;
    });
}
const content = await format(JSON.stringify(registry), {
    ...(await resolveConfig(fileURLToPath(output))),
    parser: "json"
});
if (args.includes("--check")) {
    if ((await readFile(output, "utf8")) !== content) {
        throw new Error(
            "Editor help is stale. Run npm --prefix ui run generate:editor-help."
        );
    }
    console.log("Editor help matches its authored registry.");
} else {
    await writeFile(output, content);
    console.log("Generated editor help from the local registry.");
}
