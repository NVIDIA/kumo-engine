// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import transform from "../inputTransformer.cjs";

describe("inputTransformer", () => {
    it("keeps top-level and nested by-slug operation IDs distinct", () => {
        const spec = {
            paths: {
                "/v0/projects/by-slug/{slug}": {
                    get: {
                        operationId: "get_project_by_slug",
                        tags: ["Projects"]
                    }
                },
                "/v0/projects/{project_id}/identities/by-slug/{slug}": {
                    get: {
                        operationId: "get_identity_by_slug",
                        tags: ["Identities"]
                    }
                }
            }
        };

        const result = transform(spec);

        expect(
            result.paths["/v0/projects/by-slug/{slug}"].get.operationId
        ).toBe("get_projects_by-slug_by_slug");
        expect(
            result.paths["/v0/projects/{project_id}/identities/by-slug/{slug}"]
                .get.operationId
        ).toBe("get_projects_identities_by-slug_by_slug");
    });
});
