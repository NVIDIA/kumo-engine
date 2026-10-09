// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import data from "utils/PQLParsing/pqDocs.json";

import { parseHintsFromMintlify } from "../parseHints";

describe("bundled editor help", () => {
    it("contains title, syntax and description for each entry", () => {
        expect(Object.keys(data).length).toBeGreaterThan(0);
        for (const entry of Object.values(data)) {
            expect(entry).toHaveLength(3);
            for (const text of entry) {
                expect(typeof text).toBe("string");
                expect(text.trim()).not.toBe("");
            }
        }
    });

    it.each([
        ["PQLANG_V2", "and"],
        ["YAML", "activation"]
    ])("renders the bundled %s help", (language, key) => {
        const hints = parseHintsFromMintlify(language, {});
        expect(hints[key].title).toBe(data[key as keyof typeof data][0]);
        expect(hints[key].subtitle).not.toBe("");
        expect(hints[key].hint).not.toBe("");
    });
});
