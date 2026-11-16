// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HighlightTypes, HoverHintsType } from "types/codeEditor";

import { keywords, operators, type_name } from "utils/Constants";
import data from "utils/PQLParsing/pqDocs.json";

import { LANGUAGE_KEYS, parseRegistry } from "./editorHintRegistry";

export { EDITOR_HINT_SCHEMA_VERSION } from "./editorHintRegistry";

/**
 * Escapes HTML characters and converts markdown syntax to HTML tags
 * @param text - The text to sanitize and format
 * @param convertBold - Whether to convert **text** to <strong> tags
 * @returns Sanitized and formatted text
 */
function sanitizeAndFormatText(text: string, convertBold = false): string {
    if (!text) return text;

    let result = text
        .replaceAll("<", "&lt")
        .replaceAll(">", "&gt")
        .replaceAll(/`([^`]*)`/g, "<code>$1</code>");

    if (convertBold) {
        result = result.replaceAll(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    }

    return result;
}

// Strip the gt_ / gnn_ prefix from a key (those keys only).
function stripKeyPrefix(key: string): string {
    if (key.startsWith("gt_") || key.startsWith("gnn_")) {
        return key.startsWith("gt_")
            ? key.split("gt_")[1]
            : key.split("gnn_")[1];
    }
    return key;
}

// Filter empty lines and " Description", then sanitize with bold conversion.
function formatHintBody(raw: string): string {
    const body = raw
        ?.split(/\n/)
        ?.filter((word) => word !== "" && word !== " Description")
        ?.join("\r\n");
    return sanitizeAndFormatText(body, true);
}

// Resolve the highlight type, possibly rewriting keyval/title (operators, rank).
function resolveHighlight(
    language: string,
    key: string,
    keyval: string,
    title: string
): { keyval: string; title: string; highlightType: HighlightTypes } {
    let highlightType: HighlightTypes = HighlightTypes.KEYWORD;
    if (language === "PQLANG_V2") {
        if (keywords?.indexOf(key) !== -1) {
            if (key === "rank") {
                keyval = "rank";
            }
            highlightType = HighlightTypes.KEYWORD;
        } else if (operators?.indexOf(key) !== -1) {
            keyval = keyval?.replaceAll("_", " ");
            title = title?.replaceAll("_", " ");
            highlightType = HighlightTypes.OPERATOR;
        } else if (type_name?.indexOf(key) !== -1) {
            highlightType = HighlightTypes.TYPE_NAME;
        }
    } else if (language === "YAML") {
        highlightType = HighlightTypes.KEYWORD;
    }
    return { keyval, title, highlightType };
}

export function parseHintsFromMintlify(
    language: string,
    hintsObj: HoverHintsType,
    source: unknown = data
) {
    if (
        source &&
        typeof source === "object" &&
        ("schemaVersion" in source || "hints" in source || "noHints" in source)
    ) {
        const registry = parseRegistry(source);
        for (const entry of registry.hints) {
            if (entry.language !== language) continue;
            const hint = {
                title: entry.title,
                subtitle:
                    entry.format === "literal"
                        ? entry.syntax
                        : sanitizeAndFormatText(entry.syntax),
                hint:
                    entry.format === "literal"
                        ? entry.description
                        : formatHintBody(entry.description),
                link: entry.documentationUrl ?? "",
                type: entry.highlightType
            };
            for (const key of [entry.key, ...entry.aliases])
                hintsObj[key] = hint;
        }
        return hintsObj;
    }
    // Legacy tuple data remains readable during reviewed synchronization updates.
    const legacy = source as Record<string, string[]>;
    const keys: string[] = LANGUAGE_KEYS[language] ?? [];

    try {
        for (const rawKey of keys) {
            const link = `https://kumo.ai/docs/reference/${rawKey}`;
            const key = stripKeyPrefix(rawKey);

            const value = legacy[key];
            if (value === undefined) {
                continue;
            }
            const title0 = value[0];
            let keyval: string = String(title0).toLowerCase();
            if (key === "column_processing") {
                keyval = "encoder_overrides";
            }
            const subtitle = sanitizeAndFormatText(value[1]);
            const body = formatHintBody(value[2]);
            const resolved = resolveHighlight(language, key, keyval, title0);
            hintsObj[resolved.keyval] = {
                title: resolved.title,
                subtitle: subtitle,
                hint: body,
                link: link,
                type: resolved.highlightType as HighlightTypes
            };
        }
    } catch (error) {
        console.log(error);
    }
    return hintsObj;
}
