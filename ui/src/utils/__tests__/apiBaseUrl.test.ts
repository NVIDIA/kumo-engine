// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { resolveApiBaseUrl } from "../apiBaseUrl";

describe("resolveApiBaseUrl", () => {
    it("uses the relative same-origin base for a local Vite UI", () => {
        expect(
            resolveApiBaseUrl({
                mode: "development",
                developmentBase: "/",
                productionBase: "https://api.example.com"
            })
        ).toBe("/");
    });

    it("leaves deployed runtime configuration unchanged", () => {
        expect(
            resolveApiBaseUrl({
                mode: "production",
                developmentBase: "/",
                productionBase: "/api/"
            })
        ).toBe("/api/");
    });

    it("retains explicit remote and test bases", () => {
        expect(
            resolveApiBaseUrl({
                apiTarget: "https://tenant.example.com/internal",
                mode: "development",
                developmentBase: "/"
            })
        ).toBe("https://tenant.example.com/internal");
        expect(resolveApiBaseUrl({ mode: "test" })).toBe(
            "http://localhost:10001"
        );
    });
});
