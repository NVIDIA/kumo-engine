// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from "zod";

import { HighlightTypes, HoverHintsType } from "types/codeEditor";

import { keywords, operators, type_name } from "utils/Constants";
import data from "utils/PQLParsing/pqDocs.json";

import {
    fallbackHintKeys,
    modelPlannerKeys,
    predictiveQueryKeys
} from "./tooltipKeys";

export const EDITOR_HINT_SCHEMA_VERSION = 1;

const languageSchema = z.enum(["PQLANG_V2", "YAML"]);
const keySchema = z
    .string()
    .min(1)
    .refine(
        (key) =>
            key.trim() === key &&
            !/[\p{Cc}]/u.test(key) &&
            !["__proto__", "constructor", "prototype"].includes(key)
    );
const contentSchema = z.string().refine(
    (text) =>
        !Array.from(text).some((char) => {
            const code = char.charCodeAt(0);
            return (code < 32 && !"\t\r\n".includes(char)) || code === 127;
        })
);
const registrySchema = z
    .object({
        schemaVersion: z.literal(EDITOR_HINT_SCHEMA_VERSION),
        hints: z
            .array(
                z
                    .object({
                        language: languageSchema,
                        key: keySchema,
                        aliases: z.array(keySchema),
                        configuredKeys: z.array(keySchema),
                        title: contentSchema.refine((title) => !!title.trim()),
                        syntax: contentSchema,
                        description: contentSchema,
                        highlightType: z.enum(HighlightTypes),
                        format: z.enum(["markdown", "literal"]),
                        documentationUrl: z
                            .string()
                            .url()
                            .regex(/^https:\/\/[^\s\p{Cc}"'`<>\\]+$/u)
                            .refine(
                                (url) =>
                                    !new URL(url).username &&
                                    !new URL(url).password
                            )
                            .optional()
                    })
                    .strict()
                    .refine(
                        (entry) =>
                            !!entry.syntax.trim() || !!entry.description.trim()
                    )
            )
            .min(1),
        noHints: z.array(
            z
                .object({
                    language: languageSchema,
                    key: keySchema,
                    reason: z.string().trim().min(1)
                })
                .strict()
        )
    })
    .strict();

function validateMarkup(text: string, literal = false) {
    const plain = literal
        ? text.replace(
              /<\/?code>|<img src="https:\/\/files\.readme\.io\/[a-zA-Z0-9_-]+\.png" alt="[^"<>\p{Cc}]*">/gu,
              ""
          )
        : text;
    if (literal ? plain.includes("<") : /<\s*[a-z!/?]/i.test(plain)) {
        throw new Error("Unsafe editor hint markup");
    }
}

function parseRegistry(source: unknown) {
    const registry = registrySchema.parse(source);
    const emitted = new Set<string>();
    const configured = new Set<string>();
    const claim = (set: Set<string>, language: string, key: string) => {
        const identity = `${language}:${key}`;
        if (set.has(identity))
            throw new Error(`Duplicate editor hint key: ${identity}`);
        set.add(identity);
    };
    for (const entry of registry.hints) {
        for (const key of [entry.key, ...entry.aliases])
            claim(emitted, entry.language, key);
        for (const key of entry.configuredKeys)
            claim(configured, entry.language, key);
        // ap/mape preserve existing metric metadata without visible hover bodies.
        const emptyBody =
            entry.language === "YAML" && ["ap", "mape"].includes(entry.key);
        if (emptyBody ? entry.description !== "" : !entry.description.trim())
            throw new Error(
                `Invalid editor hint description: ${entry.language}:${entry.key}`
            );
        validateMarkup(entry.title);
        if (entry.format === "literal") {
            validateMarkup(entry.syntax, true);
            validateMarkup(entry.description, true);
        }
    }
    for (const entry of registry.noHints) {
        if (emitted.has(`${entry.language}:${entry.key}`))
            throw new Error(
                `Conflicting no-hint disposition: ${entry.language}:${entry.key}`
            );
        claim(configured, entry.language, entry.key);
    }
    if (configured.size !== Object.values(LANGUAGE_KEYS).flat().length)
        throw new Error("Unexpected configured editor hint key");
    for (const [language, keys] of Object.entries(LANGUAGE_KEYS)) {
        for (const key of keys) {
            if (!configured.has(`${language}:${key}`))
                throw new Error(
                    `Missing editor hint disposition: ${language}:${key}`
                );
        }
        for (const key of fallbackHintKeys[language] ?? []) {
            if (!emitted.has(`${language}:${key}`))
                throw new Error(
                    `Missing fallback editor hint: ${language}:${key}`
                );
        }
    }
    return registry;
}

// Language to keys mapping for extensibility
const LANGUAGE_KEYS: Record<string, string[]> = {
    PQLANG_V2: predictiveQueryKeys,
    YAML: modelPlannerKeys
};

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
