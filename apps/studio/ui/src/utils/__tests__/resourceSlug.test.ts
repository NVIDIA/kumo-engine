// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    deriveResourceSlug,
    resourceSlugError,
    validateResourceSlug
} from "utils/resourceSlug";

describe("resourceSlug", () => {
    it.each([
        ["Fraud Detection", "fraud-detection"],
        ["  Risk__Team! ", "risk-team"],
        ["ＡＢＣ Data", "abc-data"]
    ])("derives %s as %s", (name, expected) => {
        expect(deriveResourceSlug(name)).toBe(expected);
    });

    it("leaves non-ASCII letters for canonical validation to reject", () => {
        const derived = deriveResourceSlug("Café Data");
        expect(derived).toBe("café-data");
        expect(validateResourceSlug(derived)).not.toBe("");
    });

    it.each(["ab", "Upper", "two--hyphens", "ends-", "équipe"])(
        "rejects invalid ID %s",
        (slug) => expect(validateResourceSlug(slug)).not.toBe("")
    );

    it.each([
        "ambient-s3",
        "api",
        "by-id",
        "by-slug",
        "default",
        "new",
        "settings",
        "system"
    ])("rejects reserved ID %s", (slug) =>
        expect(validateResourceSlug(slug)).toContain("reserved")
    );

    it("validates the derived ID when no explicit ID was chosen", () => {
        expect(resourceSlugError("", "Default")).toContain("reserved");
        expect(resourceSlugError("prod-aws", "Default")).toBe("");
    });
});
