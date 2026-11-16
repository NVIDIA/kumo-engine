// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StringStream } from "@codemirror/language";
import { describe, expect, it } from "vitest";

import { highlights, yaml } from "../yaml";
import { yamlDictionary } from "../yamlDictionary";

describe("yaml", () => {
    it("Should have highlight specs with tag property", () => {
        expect(highlights.specs.length).toBeGreaterThan(0);
        highlights.specs.forEach((spec) => {
            expect(spec).toHaveProperty("tag");
        });
    });

    it("Should return LanguageSupport instance with language and support", () => {
        const yamlSupport = yaml();

        expect(yamlSupport.language).toBeDefined();
        expect(yamlSupport.support).toBeDefined();
    });

    describe("YAML tokenizer", () => {
        const keys: Array<string> = Object.keys(yamlDictionary);

        function tokenize(text: string, column = 0): string | null {
            const stream = new StringStream(text, 4, column);

            for (const keyword of keys) {
                if (stream.match(keyword, true, true)) {
                    const peek = stream.peek();

                    if (!(peek === undefined)) {
                        if (
                            !(
                                peek.trim() === "" ||
                                /[(),'_:[\].e-]+/.test(peek) ||
                                keys.includes(peek)
                            )
                        ) {
                            return null;
                        } else if (stream.column() !== 0) {
                            const currentStream = stream.string;
                            const shouldBeSpace = currentStream.charAt(
                                stream.column() - 1
                            );
                            if (
                                shouldBeSpace !== " " &&
                                !keys.includes(shouldBeSpace) &&
                                !/[(),'_:[\].e-]+/.test(shouldBeSpace)
                            ) {
                                return null;
                            }
                        }
                        return yamlDictionary[keyword]?.type ?? null;
                    }
                    return yamlDictionary[keyword]?.type ?? null;
                }
            }
            stream.next();
            return null;
        }

        it.each([
            ["predict ", "predict"],
            ["assuming:", "assuming"],
            ["where(", "where"],
            ["activation,", "activation"],
            ["channels ", "channels"],
            ["max_epochs:", "max_epochs"],
            ["loss[", "loss"],
            ["module.", "module"],
            ["name_", "name"],
            ["1e-", "e-"],
            ["0", "0"],
            [".", "."],
            ["-", "-"]
        ])("Should tokenize '%s' as %s", (input, keyword) => {
            const token = tokenize(input);
            expect(token).toBe(yamlDictionary[keyword]?.type ?? null);
        });

        it.each([["predictable"], ["predict123"]])(
            "Should reject '%s'",
            (input) => {
                const token = tokenize(input);
                expect(token).toBeNull();
            }
        );

        it.each([["PREDICT "], ["Predict "], ["predict "]])(
            "Should match '%s' case-insensitively",
            (input) => {
                const token = tokenize(input);
                expect(token).toBe(yamlDictionary["predict"]?.type ?? null);
            }
        );

        it.each([
            [" "],
            ["("],
            [")"],
            [","],
            ["'"],
            ["_"],
            [":"],
            ["["],
            ["]"],
            ["."],
            ["e"],
            ["-"]
        ])("Should allow peek character '%s'", (char) => {
            const token = tokenize(`metrics${char}`);
            expect(token).toBe(yamlDictionary["metrics"]?.type ?? null);
        });

        it("Should return null for non-matching text", () => {
            const token = tokenize("random_text_that_does_not_match ");
            expect(token).toBeNull();
        });
    });

    it.each([
        ["encoder_overrides"],
        ["activation"],
        ["channels"],
        ["metrics"],
        ["loss"],
        ["auroc"],
        ["mae"]
    ])("Should have valid yamlDictionary entry for '%s'", (key) => {
        expect(yamlDictionary[key]).toBeDefined();
        expect(yamlDictionary[key].type).toBeDefined();
    });
});
