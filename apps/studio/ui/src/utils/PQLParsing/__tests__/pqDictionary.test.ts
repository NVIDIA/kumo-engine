// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { HighlightTypes } from "types/codeEditor";

import { pqObj } from "../pqDictionary";

describe("pqDictionary", () => {
    describe("pqObj", () => {
        it("Should export hover hints object", () => {
            expect(pqObj).toBeDefined();
            expect(typeof pqObj).toBe("object");
        });

        it("Should have entries for all digits 0-9", () => {
            const digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

            digits.forEach((digit) => {
                expect(pqObj[digit]).toBeDefined();
            });
        });

        it("Should have entries for numeric symbols", () => {
            expect(pqObj["-"]).toBeDefined();
            expect(pqObj["."]).toBeDefined();
        });

        it("Should have all entries with NUMBER type", () => {
            Object.values(pqObj).forEach((entry) => {
                expect(entry.type).toBe(HighlightTypes.NUMBER);
            });
        });

        it("Should have all entries with required properties", () => {
            Object.values(pqObj).forEach((entry) => {
                expect(entry).toHaveProperty("title");
                expect(entry).toHaveProperty("subtitle");
                expect(entry).toHaveProperty("hint");
                expect(entry).toHaveProperty("link");
                expect(entry).toHaveProperty("type");
            });
        });

        it("Should have 12 total entries", () => {
            const keys = Object.keys(pqObj);
            expect(keys.length).toBe(12);
        });
    });
});
