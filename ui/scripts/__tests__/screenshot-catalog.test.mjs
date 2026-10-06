// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync, readdirSync, statSync } from "node:fs";
import { relative, resolve } from "node:path";

import { describe, expect, it } from "vitest";

const weakSelectors = new Set([
    "main",
    "h1",
    '[data-testid="main-layout"]',
    "[data-testid='main-layout']",
    '[role="tablist"]',
    "[role='tablist']"
]);

const isWeakSelector = (selector) => {
    if (typeof selector !== "string") {
        return false;
    }
    const normalized = selector.trim();
    const lowered = normalized.toLowerCase();

    return (
        weakSelectors.has(normalized) ||
        normalized.startsWith("h1") ||
        lowered.includes("main-layout") ||
        lowered.includes("tablist") ||
        lowered.includes("tabs-list")
    );
};

const getCovers = (item) => {
    if (Array.isArray(item.covers)) {
        return item.covers;
    }

    if (typeof item.covers === "string") {
        return [item.covers];
    }

    return [];
};

const coverageIdPattern = /^[a-z][a-z0-9_]*$/;

const collectSourceFiles = (dir) => {
    return readdirSync(dir).flatMap((name) => {
        const filePath = resolve(dir, name);
        const stat = statSync(filePath);

        if (stat.isDirectory()) {
            return collectSourceFiles(filePath);
        }

        return [filePath];
    });
};

const selectorAttributePattern =
    /\[data-(testid|screenshot-ready)=["']([^"']+)["']\]/g;

const collectSelectorAttributes = (selector) => {
    if (typeof selector !== "string") {
        return [];
    }

    return [...selector.matchAll(selectorAttributePattern)].map((match) => ({
        attribute: `data-${match[1]}`,
        value: match[2]
    }));
};

describe("screenshot catalog", () => {
    const catalogPath = resolve(
        import.meta.dirname,
        "../../src/screenshot/catalog.json"
    );
    const routeMapPath = resolve(
        import.meta.dirname,
        "../../src/screenshot/route-map.json"
    );
    const catalog = JSON.parse(readFileSync(catalogPath, "utf8"));
    const routeMap = JSON.parse(readFileSync(routeMapPath, "utf8"));
    const srcRoot = resolve(import.meta.dirname, "../../src");
    const absoluteSourceFiles = collectSourceFiles(srcRoot);
    const sourceFiles = absoluteSourceFiles.map(
        (filePath) =>
            `ui/src/${relative(srcRoot, filePath).replaceAll("\\", "/")}`
    );
    const implementationSource = absoluteSourceFiles
        .filter((filePath) => !filePath.includes("/src/screenshot/"))
        .map((filePath) => readFileSync(filePath, "utf8"))
        .join("\n");

    it("contains valid recipe entries", () => {
        expect(Array.isArray(catalog)).toBe(true);
        expect(catalog.length).toBeGreaterThan(0);

        for (const entry of catalog) {
            expect(typeof entry.key).toBe("string");
            expect(entry.key.length).toBeGreaterThan(0);
            expect(Array.isArray(entry.patterns)).toBe(true);
            expect(entry.patterns.length).toBeGreaterThan(0);
            expect(Array.isArray(entry.canonicalRoutes)).toBe(true);
            expect(entry.canonicalRoutes.length).toBeGreaterThan(0);
            expect(Array.isArray(entry.visualCoverage)).toBe(true);
            expect(entry.visualCoverage.length).toBeGreaterThan(0);
            for (const coverage of entry.visualCoverage) {
                expect(typeof coverage).toBe("object");
                expect(Array.isArray(coverage)).toBe(false);
                expect(coverage.id).toMatch(coverageIdPattern);
                expect(typeof coverage.label).toBe("string");
                expect(coverage.label.length).toBeGreaterThan(10);
            }
            expect(entry.plan).toBeTruthy();
            expect(Array.isArray(entry.plan.scenarios)).toBe(true);
            expect(entry.plan.scenarios.length).toBeGreaterThan(0);
        }
    });

    it("requires assertions for every scenario", () => {
        for (const entry of catalog) {
            for (const scenario of entry.plan.scenarios) {
                expect(typeof scenario.route).toBe("string");
                expect(scenario.route.startsWith("/")).toBe(true);
                expect(Array.isArray(scenario.assertions)).toBe(true);
                expect(scenario.assertions.length).toBeGreaterThan(0);
            }
        }
    });

    it("requires every visual coverage item to be backed by setup or assertions", () => {
        for (const entry of catalog) {
            const expectedCoverage = new Set(
                entry.visualCoverage.map((coverage) => coverage.id)
            );
            const covered = new Set();

            for (const scenario of entry.plan.scenarios) {
                for (const action of scenario.setup ?? []) {
                    for (const coverage of getCovers(action)) {
                        covered.add(coverage);
                    }
                }

                for (const assertion of scenario.assertions) {
                    for (const coverage of getCovers(assertion)) {
                        covered.add(coverage);
                    }
                }
            }

            expect([...covered].sort()).toEqual([...expectedCoverage].sort());
        }
    });

    it("uses only declared coverage ids in covers fields", () => {
        for (const entry of catalog) {
            const expectedCoverage = new Set(
                entry.visualCoverage.map((coverage) => coverage.id)
            );

            for (const scenario of entry.plan.scenarios) {
                for (const action of scenario.setup ?? []) {
                    for (const coverage of getCovers(action)) {
                        expect(coverage).toMatch(coverageIdPattern);
                        expect(expectedCoverage.has(coverage)).toBe(true);
                    }
                }

                for (const assertion of scenario.assertions) {
                    for (const coverage of getCovers(assertion)) {
                        expect(coverage).toMatch(coverageIdPattern);
                        expect(expectedCoverage.has(coverage)).toBe(true);
                    }
                }
            }
        }
    });

    it("prevents generic main-layout-only recipes", () => {
        for (const entry of catalog) {
            for (const scenario of entry.plan.scenarios) {
                const readySelectors = Array.isArray(scenario.readySelector)
                    ? scenario.readySelector
                    : scenario.readySelector
                      ? [scenario.readySelector]
                      : [];
                const assertionSelectors = scenario.assertions.map(
                    (assertion) => assertion.selector
                );
                const hasSeededState = Boolean(
                    scenario.historyState ||
                    scenario.localStorage ||
                    scenario.sessionStorage
                );
                const hasSetup =
                    Array.isArray(scenario.setup) && scenario.setup.length > 0;
                const allReadyWeak =
                    readySelectors.length > 0 &&
                    readySelectors.every((selector) =>
                        isWeakSelector(selector)
                    );
                const allAssertionsWeak =
                    assertionSelectors.length > 0 &&
                    assertionSelectors.every((selector) =>
                        isWeakSelector(selector)
                    );

                expect(
                    !(
                        !hasSeededState &&
                        !hasSetup &&
                        allReadyWeak &&
                        allAssertionsWeak
                    )
                ).toBe(true);
            }
        }
    });

    it("rejects text assertions on broad container selectors", () => {
        for (const entry of catalog) {
            for (const scenario of entry.plan.scenarios) {
                for (const assertion of scenario.assertions) {
                    if (assertion.type === "text") {
                        expect(isWeakSelector(assertion.selector)).toBe(false);
                    }
                }
            }
        }
    });

    it("requires detail and output recipes to assert on their own surface", () => {
        const routeHeadingPattern =
            /\[data-testid="(?:models|predictions|jobs)-list-page-heading"\]/;

        for (const entry of catalog) {
            if (
                !entry.key.includes("details") &&
                !entry.key.includes("outputs")
            ) {
                continue;
            }

            const selectors = entry.plan.scenarios.flatMap((scenario) =>
                scenario.assertions.map((assertion) => assertion.selector)
            );

            expect(
                selectors.some(
                    (selector) => !routeHeadingPattern.test(selector)
                ),
                `${entry.key} must assert on a detail/output surface, not only a list heading`
            ).toBe(true);
        }
    });

    it("uses clickable job id cells for job detail recipe setup", () => {
        const expectedSetupSelectorByRecipe = new Map([
            ["model_details", ':nth-match([id^="trainingjob-"], 1)'],
            ["prediction_details", ':nth-match([id^="bp-job-"], 1)'],
            ["prediction_outputs", ':nth-match([id^="bp-job-"], 1)']
        ]);

        for (const [
            recipeKey,
            expectedSelector
        ] of expectedSetupSelectorByRecipe) {
            const recipe = catalog.find((entry) => entry.key === recipeKey);
            const setupSelectors = recipe.plan.scenarios.flatMap((scenario) =>
                (scenario.setup ?? []).map((action) => action.selector)
            );

            expect(setupSelectors).toContain(expectedSelector);
            expect(
                setupSelectors.some((selector) =>
                    selector.includes('a[href*="')
                )
            ).toBe(false);
        }
    });

    it("keeps route-map recipes tied to catalog coverage", () => {
        const catalogKeys = new Set(catalog.map((entry) => entry.key));

        expect(Array.isArray(routeMap)).toBe(true);
        expect(routeMap.length).toBeGreaterThan(0);

        for (const mapping of routeMap) {
            expect(typeof mapping.key).toBe("string");
            expect(mapping.key.length).toBeGreaterThan(0);
            expect(Array.isArray(mapping.patterns)).toBe(true);
            expect(mapping.patterns.length).toBeGreaterThan(0);
            expect(Array.isArray(mapping.routes)).toBe(true);
            expect(mapping.routes.length).toBeGreaterThan(0);
            expect(Array.isArray(mapping.recipeKeys)).toBe(true);
            expect(mapping.recipeKeys.length).toBeGreaterThan(0);
            for (const route of mapping.routes) {
                expect(route.startsWith("/")).toBe(true);
            }
            for (const recipeKey of mapping.recipeKeys) {
                expect(catalogKeys.has(recipeKey)).toBe(true);
            }
        }
    });

    it("keeps every catalog and route-map pattern tied to a real source file", () => {
        for (const entry of catalog) {
            for (const pattern of entry.patterns) {
                const regex = new RegExp(pattern);
                expect(
                    sourceFiles.some((filePath) => regex.test(filePath)),
                    `${entry.key} catalog pattern did not match a real file: ${pattern}`
                ).toBe(true);
            }
        }

        for (const mapping of routeMap) {
            for (const pattern of mapping.patterns) {
                const regex = new RegExp(pattern);
                expect(
                    sourceFiles.some((filePath) => regex.test(filePath)),
                    `${mapping.key} route-map pattern did not match a real file: ${pattern}`
                ).toBe(true);
            }
        }
    });

    it("keeps readiness selector values tied to implementation source", () => {
        for (const entry of catalog) {
            for (const scenario of entry.plan.scenarios) {
                const readySelectors = [
                    ...(Array.isArray(scenario.readySelector)
                        ? scenario.readySelector
                        : scenario.readySelector
                          ? [scenario.readySelector]
                          : []),
                    ...(Array.isArray(scenario.readySelectors)
                        ? scenario.readySelectors
                        : [])
                ];

                for (const selector of readySelectors) {
                    for (const {
                        attribute,
                        value
                    } of collectSelectorAttributes(selector)) {
                        const escapedValue = value.replace(
                            /[.*+?^${}()|[\]\\]/g,
                            "\\$&"
                        );
                        const literalAttribute = new RegExp(
                            `${attribute}[\\s\\S]{0,160}["']${escapedValue}["']`
                        );
                        const pageHeaderProxy =
                            (attribute === "data-screenshot-ready" ||
                                attribute === "data-testid") &&
                            new RegExp(
                                `screenshotTestId=["']${escapedValue}["']`
                            ).test(implementationSource);
                        const listOverviewProxy =
                            attribute === "data-screenshot-ready" &&
                            value.endsWith("-list-page") &&
                            implementationSource.includes(
                                "loadingStatus ? undefined : listOverviewTestId"
                            );

                        expect(
                            literalAttribute.test(implementationSource) ||
                                pageHeaderProxy ||
                                listOverviewProxy,
                            `${entry.key} readiness selector is not tied to implementation source: ${selector}`
                        ).toBe(true);
                    }
                }
            }
        }
    });
});
