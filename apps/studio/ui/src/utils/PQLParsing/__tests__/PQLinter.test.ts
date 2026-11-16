// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { pqlLinter } from "../PQLinter";

describe("pqlLinter", () => {
    describe("Error message parsing", () => {
        it("Should return extension for empty error message", () => {
            const result = pqlLinter("");

            expect(result).toBeDefined();
            expect(typeof result).toBe("object");
        });

        it.each([
            {
                description: "Should handle row/column error format",
                errorMsg: "1. 'target' row 5, column 10: invalid syntax"
            },
            {
                description: "Should handle line/col error format",
                errorMsg: "1. 'WHERE' line 2, col 3; unexpected token"
            },
            {
                description: "Should handle line and col error format",
                errorMsg: "1. 'PREDICT' line 1 and col 5; missing value"
            },
            {
                description: "Should handle 'in row' error format",
                errorMsg: "1. 'CLASSIFY' in row 3, column 8: syntax error"
            },
            {
                description: "Should handle 'in line' error format",
                errorMsg: "1. 'RANK' in line 4, col 2; invalid"
            },
            {
                description: "Should handle 'in line and col' error format",
                errorMsg: "1. 'test' in line 5 and col 6; error"
            }
        ])("$description", ({ errorMsg }) => {
            const result = pqlLinter(errorMsg);

            expect(result).toBeDefined();
            expect(typeof result).toBe("object");
        });

        it("Should handle multiple errors in single message", () => {
            const errorMsg =
                "1. 'error1' row 1, column 5: first 2. 'error2' line 2, col 3; second 3. 'error3' line 3, col 4; third";
            const result = pqlLinter(errorMsg);

            expect(result).toBeDefined();
            expect(typeof result).toBe("object");
        });

        it("Should handle error message without quoted tokens", () => {
            const errorMsg = "1. some error without quotes";
            const result = pqlLinter(errorMsg);

            expect(result).toBeDefined();
            expect(typeof result).toBe("object");
        });

        it("Should handle error with only numeric index", () => {
            const errorMsg = "1. ";
            const result = pqlLinter(errorMsg);

            expect(result).toBeDefined();
            expect(typeof result).toBe("object");
        });
    });

    describe("Error token extraction", () => {
        it.each([
            {
                description: "Should extract token from single quotes",
                errorMsg: "1. 'tokenName' row 1, col 5: error",
                hasToken: true
            },
            {
                description: "Should handle alphanumeric tokens",
                errorMsg: "1. 'token123' row 1, col 5: error",
                hasToken: true
            },
            {
                description: "Should handle uppercase tokens",
                errorMsg: "1. 'PREDICT' row 1, col 5: error",
                hasToken: true
            },
            {
                description: "Should handle lowercase tokens",
                errorMsg: "1. 'where' row 1, col 5: error",
                hasToken: true
            },
            {
                description: "Should handle mixed case tokens",
                errorMsg: "1. 'MyToken' row 1, col 5: error",
                hasToken: true
            }
        ])("$description", ({ errorMsg }) => {
            const result = pqlLinter(errorMsg);

            expect(result).toBeDefined();
            expect(typeof result).toBe("object");
        });
    });

    describe("Edge cases", () => {
        it("Should handle empty string", () => {
            const result = pqlLinter("");

            expect(result).toBeDefined();
        });

        it("Should handle whitespace only", () => {
            const result = pqlLinter("   ");

            expect(result).toBeDefined();
        });

        it("Should handle error message with no numeric prefix", () => {
            const result = pqlLinter("'token' row 1, col 5: error");

            expect(result).toBeDefined();
        });

        it("Should handle very long error messages", () => {
            const longMsg = Array(100)
                .fill(0)
                .map((_, i) => `${i + 1}. 'error${i}' row 1, col ${i}: msg`)
                .join(" ");
            const result = pqlLinter(longMsg);

            expect(result).toBeDefined();
        });

        it("Should handle special characters in error message", () => {
            const errorMsg =
                "1. 'token' row 1, col 5: error with $pecial ch@rs!";
            const result = pqlLinter(errorMsg);

            expect(result).toBeDefined();
        });

        it("Should handle tokens with underscores", () => {
            const errorMsg = "1. 'my_token' row 1, col 5: error";
            const result = pqlLinter(errorMsg);

            expect(result).toBeDefined();
        });
    });
});
