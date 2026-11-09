// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { HighlightTypes } from "types/codeEditor";

import data from "utils/PQLParsing/pqDocs.json";
import { yamlDict } from "utils/YAML/yamlDictionary";

import {
    EDITOR_HINT_SCHEMA_VERSION,
    parseHintsFromMintlify
} from "../parseHints";
import { modelPlannerKeys, predictiveQueryKeys } from "../tooltipKeys";

describe("bundled editor help", () => {
    it("contains explicit language, key, content, highlight and configured-key dispositions", () => {
        expect(data.schemaVersion).toBe(EDITOR_HINT_SCHEMA_VERSION);
        expect(data.hints.length).toBeGreaterThan(0);
        for (const entry of data.hints) {
            expect(entry.title.trim()).not.toBe("");
            expect(typeof entry.syntax).toBe("string");
            expect(typeof entry.description).toBe("string");
            expect(Object.values(HighlightTypes)).toContain(
                entry.highlightType
            );
        }
        for (const [language, keys] of [
            ["PQLANG_V2", predictiveQueryKeys],
            ["YAML", modelPlannerKeys]
        ] as const) {
            const configured = data.hints
                .filter((entry) => entry.language === language)
                .flatMap((entry) => entry.configuredKeys);
            expect(configured.sort()).toEqual([...keys].sort());
        }
    });

    it.each([
        ["PQLANG_V2", "and"],
        ["YAML", "activation"],
        ["PQLANG_V2", "predict"],
        ["PQLANG_V2", "for each"],
        ["YAML", "num_layers"]
    ])("renders the bundled %s help for %s", (language, key) => {
        const hints = parseHintsFromMintlify(language, {});
        const entry = data.hints.find(
            (entry) => entry.language === language && entry.key === key
        )!;
        expect(hints[key].title).toBe(entry.title);
        expect(hints[key].subtitle).not.toBe("");
        expect(hints[key].hint).not.toBe("");
    });

    it("preserves literal metrics, explicit no-link help and unknown-key behavior", () => {
        const hints = parseHintsFromMintlify("YAML", structuredClone(yamlDict));
        expect(hints["recall@K"].hint).toContain("<code>Recall@K</code>");
        expect(hints.ndcg.hint).toContain(
            '<img src="https://files.readme.io/ffa14c4-image.png" alt="NDCG formula">'
        );
        expect(hints.prediction_time_encodings.link).toBe("");
        expect(hints.prediction_time_encodings.hint).toBe(
            "Adds the prediction anchor timestamp to the model input."
        );
        expect(hints.ap.hint).toBe("");
        expect(hints.mape.hint).toBe("");
        expect(hints.training_job.hint).toBe("");
        expect(hints.unknown_key).toBeUndefined();
        expect(parseHintsFromMintlify("UNKNOWN", {})).toEqual({});
    });

    it("emits explicit aliases without inventing configured-key aliases", () => {
        const registry = structuredClone(data);
        Reflect.set(registry.hints[0], "aliases", ["explicit_alias"]);
        const hints = parseHintsFromMintlify(
            registry.hints[0].language,
            {},
            registry
        );
        expect(hints.explicit_alias).toEqual(hints[registry.hints[0].key]);
        expect(
            parseHintsFromMintlify("YAML", {}).gt_num_layers
        ).toBeUndefined();
    });

    it.each([
        "duplicate",
        "alias",
        "missing",
        "fallback",
        "empty",
        "language",
        "format",
        "version",
        "markup",
        "image",
        "title",
        "url",
        "field",
        "blankDescription",
        "emptyBody",
        "credentials",
        "backslash",
        "extraField",
        "extraConfigured",
        "control",
        "noHintConflict",
        "duplicateConfigured"
    ])("rejects malformed registry data: %s", (kind) => {
        const registry = structuredClone(data);
        const first = registry.hints[0];
        if (kind === "duplicate") registry.hints.push(first);
        if (kind === "alias") Reflect.set(first, "aliases", [first.key]);
        if (kind === "missing") registry.hints.splice(0, 1);
        if (kind === "fallback")
            registry.hints = registry.hints.filter(
                (entry) => entry.key !== "ap"
            );
        if (kind === "empty") first.key = "";
        if (kind === "language") first.language = "HTML";
        if (kind === "format") first.format = "html";
        if (kind === "version")
            Reflect.deleteProperty(registry, "schemaVersion");
        if (kind === "markup") {
            first.format = "literal";
            first.description = '<code onclick="alert(1)">bad</code>';
        }
        if (kind === "image") {
            first.format = "literal";
            first.description = '<img src="https://evil.example/x" alt="bad">';
        }
        if (kind === "title") first.title = "<img src=x onerror=alert(1)>";
        if (kind === "url") first.documentationUrl = "javascript:alert(1)";
        if (kind === "field") Reflect.deleteProperty(first, "description");
        if (kind === "blankDescription") first.description = "";
        if (kind === "emptyBody") {
            first.description = "";
            first.syntax = "";
        }
        if (kind === "credentials")
            first.documentationUrl =
                "https://user:password@docs.kumo.ai/example";
        if (kind === "backslash")
            first.documentationUrl = "https://docs.kumo.ai/\\example";
        if (kind === "extraField") Reflect.set(first, "unsupported", true);
        if (kind === "extraConfigured") first.configuredKeys.push("unexpected");
        if (kind === "control") first.description = "bad\u0000value";
        if (kind === "noHintConflict")
            Reflect.set(registry, "noHints", [
                {
                    language: first.language,
                    key: first.key,
                    reason: "Not emitted"
                }
            ]);
        if (kind === "duplicateConfigured")
            first.configuredKeys.push(first.configuredKeys[0]);

        expect(() =>
            parseHintsFromMintlify("PQLANG_V2", {}, registry)
        ).toThrow();
    });
});
