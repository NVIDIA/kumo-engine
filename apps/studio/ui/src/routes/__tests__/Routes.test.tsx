// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { routes } from "../Routes";

const flattenPaths = (items: typeof routes): string[] =>
    items.flatMap((route) => [
        typeof route.path === "string" ? route.path : "",
        ...(route.children ? flattenPaths(route.children) : [])
    ]);

describe("routes", () => {
    it("keeps the Direct Connect session entry points", () => {
        const paths = flattenPaths(routes);

        expect(paths).toContain("/");
        expect(paths).toContain("/login");
        expect(paths).toContain("/logout");
        expect(paths).toContain("/clear");
    });

    it("registers resources only beneath an explicit project path", () => {
        const paths = flattenPaths(routes);

        expect(paths).toContain("/project/:projectId");
        expect(paths).toContain("connectors/:id?");
        expect(paths).toContain("tables/:id?");
        expect(paths).toContain("graphs/:id?");
        expect(paths).toContain("settings");
        expect(paths).not.toContain("/connectors/");
        expect(paths).not.toContain("/tables/");
        expect(paths).not.toContain("/graphs/");
        expect(paths).not.toContain("/models/");
        expect(paths).not.toContain("/predictions/");
    });

    it("does not register compatibility routes", () => {
        const paths = flattenPaths(routes);

        expect(paths.some((path) => path.includes("connectorsV2"))).toBe(false);
        expect(paths.some((path) => path.includes("tablesV2"))).toBe(false);
        expect(paths.some((path) => path.includes("graphsV2"))).toBe(false);
        expect(paths.some((path) => path.startsWith("/jobs"))).toBe(false);
    });
});
