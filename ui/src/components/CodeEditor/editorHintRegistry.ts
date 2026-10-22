// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from "zod";

import { HighlightTypes } from "../../types/codeEditor";

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

export function parseRegistry(source: unknown) {
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
export const LANGUAGE_KEYS: Record<string, string[]> = {
    PQLANG_V2: predictiveQueryKeys,
    YAML: modelPlannerKeys
};
