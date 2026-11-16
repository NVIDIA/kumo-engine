// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { pql1 } from "../PQL1";

describe("PQL1", () => {
    const createPQL1 = () => pql1();

    describe("pql1 function", () => {
        it("Should return LanguageSupport instance", () => {
            const result = createPQL1();

            expect(result).toBeDefined();
            expect(result.language).toBeDefined();
        });

        it("Should create language with correct name", () => {
            const result = createPQL1();

            expect(result.language.name).toBe("PQL1");
        });

        it("Should include syntax highlighting support", () => {
            const result = createPQL1();

            expect(result.support).toBeDefined();
        });

        it("Should create language with parser", () => {
            const result = createPQL1();

            expect(result.language.parser).toBeDefined();
        });
    });
});
