// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest";

import {
    assertExpectedRoute,
    buildScreenshotDebugMarkdown,
    classifyScreenshotFailure,
    getManifestPath,
    getMissingCoverage,
    getRouteFileStem,
    getRouteReadySelectors,
    getScenarioFileStem,
    fetchFeatureFlags,
    isAuthDisabledMode,
    normalizePath,
    normalizeScreenshotPlan,
    parseArgs,
    parseReadySelectors,
    resolveAuthStrategy,
    shouldSkipScenarioForFeatureFlags
} from "../ci-screenshot.mjs";

describe("ci-screenshot helpers", () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("parses ready selector JSON", () => {
        expect(
            parseReadySelectors('{"*":"main","/graphs/":["#graph","main"]}')
        ).toEqual({
            "*": "main",
            "/graphs/": ["#graph", "main"]
        });
    });

    it("normalizes routes consistently", () => {
        expect(normalizePath("/graphs")).toBe("/graphs/");
        expect(normalizePath("/graphs/")).toBe("/graphs/");
        expect(normalizePath("/")).toBe("/");
    });

    it("builds stable route file stems", () => {
        expect(getRouteFileStem("/graphs/overview/")).toBe("graphs-overview");
        expect(getRouteFileStem("/")).toBe("home");
    });

    it("builds stable scenario file stems", () => {
        expect(
            getScenarioFileStem(
                "Training Task Type Sentence Case",
                "/newjob/training/"
            )
        ).toBe("training-task-type-sentence-case");
        expect(getScenarioFileStem("", "/newjob/training/")).toBe(
            "newjob-training"
        );
    });

    it("resolves exact and wildcard ready selectors", () => {
        expect(
            getRouteReadySelectors("/graphs/", {
                "/graphs/": ["[data-testid='graph-page']", "main"],
                "*": "[data-testid='main-layout']"
            })
        ).toEqual(["[data-testid='graph-page']", "main"]);

        expect(
            getRouteReadySelectors("/tables/", {
                "*": "[data-testid='main-layout']"
            })
        ).toEqual(["[data-testid='main-layout']"]);
    });

    it("returns the manifest path for an output directory", () => {
        expect(getManifestPath("/tmp/build-ui-screenshots/123")).toBe(
            "/tmp/build-ui-screenshots/123/results.json"
        );
    });

    it("normalizes route plans into scenario records", () => {
        expect(
            normalizeScreenshotPlan({
                routes: ["/newjob/training/"],
                readySelectors: {
                    "*": '[data-testid="main-layout"]',
                    "/newjob/training/": '[data-testid="infer-task-type"]'
                }
            })
        ).toEqual([
            {
                name: "newjob-training",
                route: "/newjob/training/",
                historyState: null,
                localStorage: null,
                sessionStorage: null,
                fixtures: [],
                setup: [],
                assertions: [],
                presentation: null,
                recipeKey: null,
                requiredFeatureFlags: {},
                visualCoverage: [],
                readySelectors: ['[data-testid="infer-task-type"]']
            }
        ]);
    });

    it("normalizes scenario plans with setup and assertions", () => {
        expect(
            normalizeScreenshotPlan({
                scenarios: [
                    {
                        name: "training-task-type-sentence-case",
                        route: "/newjob/training/",
                        historyState: {
                            taskType: "binary_classification"
                        },
                        localStorage: {
                            featureFlag: true
                        },
                        sessionStorage: {
                            currentTab: "planner"
                        },
                        setup: [
                            {
                                action: "paste",
                                selector: '[data-testid="query-editor"]',
                                value: "select * from my_table"
                            },
                            {
                                action: "click",
                                selector: '[data-testid="graph-selector"]'
                            }
                        ],
                        readySelector: '[data-testid="infer-task-type"]',
                        assertions: [
                            {
                                type: "text",
                                selector: '[data-testid="infer-task-type"]',
                                matches: "^Task type: [A-Z][a-z]+(?: [a-z]+)*$"
                            }
                        ]
                    }
                ]
            })
        ).toEqual([
            {
                name: "training-task-type-sentence-case",
                route: "/newjob/training/",
                historyState: {
                    taskType: "binary_classification"
                },
                localStorage: {
                    featureFlag: true
                },
                sessionStorage: {
                    currentTab: "planner"
                },
                fixtures: [],
                setup: [
                    {
                        action: "paste",
                        selector: '[data-testid="query-editor"]',
                        value: "select * from my_table"
                    },
                    {
                        action: "click",
                        selector: '[data-testid="graph-selector"]'
                    }
                ],
                assertions: [
                    {
                        type: "text",
                        selector: '[data-testid="infer-task-type"]',
                        matches: "^Task type: [A-Z][a-z]+(?: [a-z]+)*$"
                    }
                ],
                presentation: null,
                recipeKey: null,
                requiredFeatureFlags: {},
                visualCoverage: [],
                readySelectors: ['[data-testid="infer-task-type"]']
            }
        ]);
    });

    it("normalizes recipe metadata and structured visual coverage", () => {
        expect(
            normalizeScreenshotPlan({
                _recipeKey: "new_training",
                _requiredFeatureFlags: {
                    EXPERIMENTAL_TRAINING_UI: true
                },
                _visualCoverage: [
                    {
                        id: "new_training_task_type_visible",
                        label: "New Training page renders the inferred task type text"
                    }
                ],
                scenarios: [
                    {
                        name: "training-task-type",
                        route: "/newjob/training/",
                        assertions: [
                            {
                                type: "visible",
                                selector: '[data-testid="infer-task-type"]',
                                covers: ["new_training_task_type_visible"]
                            }
                        ]
                    }
                ]
            })
        ).toEqual([
            expect.objectContaining({
                recipeKey: "new_training",
                requiredFeatureFlags: {
                    EXPERIMENTAL_TRAINING_UI: true
                },
                visualCoverage: [
                    {
                        id: "new_training_task_type_visible",
                        label: "New Training page renders the inferred task type text"
                    }
                ],
                assertions: [
                    {
                        type: "visible",
                        selector: '[data-testid="infer-task-type"]',
                        covers: ["new_training_task_type_visible"]
                    }
                ]
            })
        ]);
    });

    it("preserves valid screenshot presentation metadata", () => {
        expect(
            normalizeScreenshotPlan({
                scenarios: [
                    {
                        name: "primary-evidence",
                        route: "/newjob/prediction/",
                        presentation: "primary",
                        assertions: [
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-prediction-page-heading"]'
                            }
                        ]
                    },
                    {
                        name: "unknown-presentation",
                        route: "/newjob/training/",
                        presentation: "featured",
                        assertions: [
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-training-page-heading"]'
                            }
                        ]
                    }
                ]
            }).map((scenario) => scenario.presentation)
        ).toEqual(["primary", null]);
    });

    it("skips scenarios when required feature flags are not satisfied", () => {
        const [scenario] = normalizeScreenshotPlan({
            _recipeKey: "project_list",
            _requiredFeatureFlags: {
                PROJECT_SWITCHER_V2: true
            },
            scenarios: [
                {
                    name: "project-switcher-dialog",
                    route: "/connectors/",
                    assertions: [
                        {
                            type: "visible",
                            selector: '[data-testid="project-list-heading"]'
                        }
                    ]
                }
            ]
        });

        expect(
            shouldSkipScenarioForFeatureFlags(scenario, {
                PROJECT_SWITCHER_V2: false
            })
        ).toContain("PROJECT_SWITCHER_V2=true");
        expect(
            shouldSkipScenarioForFeatureFlags(scenario, {
                PROJECT_SWITCHER_V2: true
            })
        ).toBeNull();
    });

    it("fetches feature flags through an authenticated browser context when available", async () => {
        const get = vi.fn().mockResolvedValue({
            ok: () => true,
            json: async () => ({ ENABLE_FILE_UPLOAD: true }),
            status: () => 200,
            statusText: () => "OK"
        });
        const fetchSpy = vi
            .spyOn(globalThis, "fetch")
            .mockRejectedValue(new Error("bare fetch should not be called"));

        await expect(
            fetchFeatureFlags("https://app.example.test", {
                request: { get }
            })
        ).resolves.toEqual({ ENABLE_FILE_UPLOAD: true });

        expect(get).toHaveBeenCalledWith(
            "https://app.example.test/v0/constants/flags"
        );
        expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("falls back to unauthenticated feature flag fetch for auth-disabled captures", async () => {
        vi.spyOn(globalThis, "fetch").mockResolvedValue({
            ok: true,
            json: async () => ({ ENABLE_BIGQUERY: true })
        });

        await expect(
            fetchFeatureFlags("https://app.example.test")
        ).resolves.toEqual({ ENABLE_BIGQUERY: true });

        expect(globalThis.fetch).toHaveBeenCalledWith(
            "https://app.example.test/v0/constants/flags"
        );
    });

    it("merges named screenshot fixtures into scenarios", () => {
        expect(
            normalizeScreenshotPlan({
                fixtures: {
                    seededQuery: {
                        localStorage: {
                            savedQuery: "select * from transactions"
                        },
                        setup: [
                            {
                                action: "fill",
                                selector: '[data-testid="query-editor"]',
                                value: "select * from transactions"
                            }
                        ],
                        readySelector: '[data-testid="query-editor"]'
                    }
                },
                scenarios: [
                    {
                        name: "query-state",
                        route: "/newjob/training/",
                        fixtures: ["seededQuery"],
                        localStorage: {
                            selectedGraph: "transactions"
                        },
                        assertions: [
                            {
                                type: "visible",
                                selector: '[data-testid="query-editor"]',
                                covers: "query_editor_visible"
                            }
                        ]
                    }
                ]
            })
        ).toEqual([
            expect.objectContaining({
                fixtures: ["seededQuery"],
                localStorage: {
                    savedQuery: "select * from transactions",
                    selectedGraph: "transactions"
                },
                setup: [
                    {
                        action: "fill",
                        selector: '[data-testid="query-editor"]',
                        value: "select * from transactions"
                    }
                ],
                readySelectors: ['[data-testid="query-editor"]']
            })
        ]);
    });

    it("provides built-in fixtures for common screenshot states", () => {
        expect(
            normalizeScreenshotPlan({
                scenarios: [
                    {
                        name: "model-planner-tab",
                        route: "/newjob/training/",
                        fixtures: ["selected_tab_model_planner"],
                        assertions: [
                            {
                                type: "visible",
                                selector: '[data-testid="tab-model-planner"]',
                                covers: "model_planner_tab_visible"
                            }
                        ]
                    }
                ]
            })
        ).toEqual([
            expect.objectContaining({
                fixtures: ["selected_tab_model_planner"],
                setup: [
                    {
                        action: "click",
                        selector: '[data-testid="tab-model-planner"]'
                    }
                ],
                readySelectors: ['[data-testid="tab-model-planner"]']
            })
        ]);
    });

    it("classifies screenshot failures for workflow diagnostics", () => {
        expect(
            classifyScreenshotFailure("redirected to login instead of /graphs/")
        ).toBe("auth");
        expect(
            classifyScreenshotFailure(
                "loaded unexpected route /models/ instead of /newjob/training/"
            )
        ).toBe("route-mismatch");
        expect(
            classifyScreenshotFailure("ready selector not found for /graphs/")
        ).toBe("selector-readiness");
        expect(classifyScreenshotFailure("page rendered an error state")).toBe(
            "app-runtime"
        );
    });

    it("summarizes missing coverage across screenshot results", () => {
        expect(
            getMissingCoverage([
                {
                    route: "/newjob/training/",
                    scenarioName: "training",
                    coverage: [
                        {
                            id: "task_type_visible",
                            label: "Task type is visible",
                            passed: true
                        },
                        {
                            id: "tab_label_visible",
                            label: "Tab label is visible",
                            passed: false
                        }
                    ]
                }
            ])
        ).toEqual([
            {
                route: "/newjob/training/",
                scenarioName: "training",
                id: "tab_label_visible",
                label: "Tab label is visible"
            }
        ]);
    });

    it("builds a human-readable screenshot debug summary", () => {
        const markdown = buildScreenshotDebugMarkdown({
            generatedAt: "2026-05-06T00:00:00.000Z",
            outputDir: "/tmp/build-ui-screenshots/1",
            succeeded: 0,
            failed: 1,
            coverageFailed: 1,
            failureTypes: ["selector-readiness", "coverage-missing"],
            missingCoverage: [
                {
                    route: "/newjob/training/",
                    scenarioName: "training",
                    id: "task_type_visible",
                    label: "Task type is visible"
                }
            ],
            results: [
                {
                    route: "/newjob/training/",
                    scenarioName: "training",
                    success: false,
                    finalUrl: "http://localhost:3000/newjob/training/",
                    recipeKey: "new_training",
                    failureType: "selector-readiness",
                    error: "ready selector not found",
                    readySelectors: ['[data-testid="infer-task-type"]'],
                    pageSnapshot: {
                        title: "Kumo",
                        pathname: "/newjob/training/",
                        authState: { hasAccessToken: true }
                    },
                    coverage: [
                        {
                            id: "task_type_visible",
                            label: "Task type is visible",
                            passed: false
                        }
                    ],
                    debugEvents: [
                        {
                            type: "api-response",
                            status: 500,
                            url: "http://localhost:3000/v0/test"
                        }
                    ]
                }
            ]
        });

        expect(markdown).toContain("# Screenshot Debug Summary");
        expect(markdown).toContain("Failure types: selector-readiness");
        expect(markdown).toContain("task_type_visible");
        expect(markdown).toContain("Candidate ready selectors");
        expect(markdown).toContain("api-response:500");
    });

    it("treats auth-disabled mode as active from the flag or the backend", () => {
        expect(isAuthDisabledMode({ authDisabled: true }, {})).toBe(true);
        expect(isAuthDisabledMode({}, { authDisabled: true })).toBe(true);
        expect(isAuthDisabledMode({}, {})).toBe(false);
        expect(isAuthDisabledMode(undefined, undefined)).toBe(false);
    });

    it("resolves the auth strategy, preferring real credentials and guarding backend mismatches", () => {
        // localhost-style: flag set, no creds, backend confirms auth disabled
        expect(
            resolveAuthStrategy({ authDisabled: true }, { authDisabled: true })
        ).toMatchObject({ useSentinel: true, backendRequiresAuth: false });

        // unsafe: auth-disabled requested but backend enforces auth, no creds
        expect(
            resolveAuthStrategy({ authDisabled: true }, { authDisabled: false })
        ).toMatchObject({ useSentinel: true, backendRequiresAuth: true });

        // Direct Connect credentials always win -> browser login path, never
        // the sentinel.
        expect(
            resolveAuthStrategy(
                {
                    authDisabled: true,
                    authEmail: "shots@kumo.ai",
                    authPassword: "secret"
                },
                { authDisabled: false }
            )
        ).toMatchObject({ useSentinel: false, backendRequiresAuth: false });

        // backend reports auth disabled even without the flag, no creds:
        // auth-disabled mode is active, so main()'s auth-input gate is
        // satisfied without credentials.
        expect(resolveAuthStrategy({}, { authDisabled: true })).toMatchObject({
            authDisabledMode: true,
            hasRealCreds: false,
            useSentinel: true,
            backendRequiresAuth: false
        });

        // nothing requested -> authenticated path. authDisabledMode is false and
        // there are no real creds, which is exactly the combination main()
        // fast-fails on for missing auth input.
        expect(resolveAuthStrategy({}, { authDisabled: false })).toMatchObject({
            authDisabledMode: false,
            hasRealCreds: false,
            useSentinel: false,
            backendRequiresAuth: false
        });
    });

    it("accepts --auth-disabled without Direct Connect credentials", () => {
        const authEnvKeys = [
            "SCREENSHOT_AUTH_EMAIL",
            "TEST_EMAIL",
            "SCREENSHOT_AUTH_PASSWORD",
            "TEST_PASSWORD",
            "SCREENSHOT_AUTH_DISABLED"
        ];
        const saved = Object.fromEntries(
            authEnvKeys.map((key) => [key, process.env[key]])
        );
        authEnvKeys.forEach((key) => delete process.env[key]);
        try {
            const args = parseArgs([
                "node",
                "ci-screenshot.mjs",
                "--auth-disabled",
                "--plan",
                "plan.json",
                "--output",
                "/tmp/out"
            ]);
            expect(args.authDisabled).toBe(true);
            expect(args.authEmail).toBeUndefined();
            expect(args.authPassword).toBeUndefined();
        } finally {
            authEnvKeys.forEach((key) => {
                if (saved[key] === undefined) {
                    delete process.env[key];
                } else {
                    process.env[key] = saved[key];
                }
            });
        }
    });

    it("ignores removed Auth0 screenshot credential env vars", () => {
        const authEnvKeys = [
            "AUTH0_TEST_EMAIL",
            "AUTH0_TEST_PASSWORD",
            "SCREENSHOT_AUTH_EMAIL",
            "SCREENSHOT_AUTH_PASSWORD",
            "TEST_EMAIL",
            "TEST_PASSWORD"
        ];
        const saved = Object.fromEntries(
            authEnvKeys.map((key) => [key, process.env[key]])
        );
        authEnvKeys.forEach((key) => delete process.env[key]);
        process.env.AUTH0_TEST_EMAIL = "old-auth0@example.test";
        process.env.AUTH0_TEST_PASSWORD = "old-auth0-password";
        try {
            const args = parseArgs([
                "node",
                "ci-screenshot.mjs",
                "--plan",
                "plan.json",
                "--output",
                "/tmp/out"
            ]);
            expect(args.authEmail).toBeUndefined();
            expect(args.authPassword).toBeUndefined();
        } finally {
            authEnvKeys.forEach((key) => {
                if (saved[key] === undefined) {
                    delete process.env[key];
                } else {
                    process.env[key] = saved[key];
                }
            });
        }
    });

    it("rejects redirects to external auth origins", async () => {
        const page = {
            url: () => "https://auth.kumoai.cloud/u/login?state=test"
        };

        await expect(
            assertExpectedRoute(
                page,
                "/newjob/training/",
                "http://localhost:3000"
            )
        ).rejects.toThrow(
            "redirected to external origin https://auth.kumoai.cloud instead of http://localhost:3000/newjob/training/"
        );
    });

    it("matches routes by pathname when the requested route includes query params", async () => {
        const page = {
            url: () =>
                "http://localhost:3000/models/?page=1&job_type=TRAINING_JOB"
        };

        await expect(
            assertExpectedRoute(
                page,
                "/models/?page=1&job_type=TRAINING_JOB",
                "http://localhost:3000"
            )
        ).resolves.toBeUndefined();
    });
});
