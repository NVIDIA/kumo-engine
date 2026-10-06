#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable no-undef */

/**
 * CI Screenshot Script
 *
 * Captures Playwright screenshots of UI routes for the headless build-ui
 * workflow. Use --auth-disabled for local/dev captures, or provide
 * --auth-email and --auth-password for Direct Connect password-mode captures.
 *
 * Usage:
 *   node scripts/ci-screenshot.mjs \
 *     --routes /connectors,/tables,/graphs \
 *     --output /tmp/build-ui-screenshots/12345 \
 *     [--base-url http://localhost:3000]
 */

import { chromium } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const CI_SCREENSHOT_AUTH_KEY = "__ciScreenshotAuth";
// Sentinel token used by the app when the backend reports auth_disabled.
const AUTH_DISABLED_SENTINEL = "AUTH_DISABLED";
const ERROR_PATTERNS = [
    /redirecting to login page/i,
    /checking to see if you're logged in/i,
    /something went wrong/i,
    /service temporarily unavailable/i,
    /login required/i,
    /failed to get org_id/i
];
const READY_SELECTOR_TIMEOUT_MS = 30_000;
const LOADING_OVERLAY_SELECTOR = '[data-testid="animated-loading"]';
const DEFAULT_SCREENSHOT_FIXTURES = {
    seeded_query_editor: {
        setup: [
            {
                action: "fill",
                selector: '[data-testid="query-editor"]',
                value: "SELECT * FROM transactions"
            }
        ],
        readySelector: '[data-testid="query-editor"]'
    },
    selected_tab_model_planner: {
        setup: [
            {
                action: "click",
                selector: '[data-testid="tab-model-planner"]'
            }
        ],
        readySelector: '[data-testid="tab-model-planner"]'
    },
    open_graph_table_view: {
        setup: [
            {
                action: "click",
                selector: '[data-testid="transaction_show_icon"]'
            }
        ],
        readySelector: "text=Column Statistics"
    },
    opened_dialog: {
        readySelector: '[data-testid="dialog-content"]'
    },
    list_page_loaded: {
        readySelector: '[data-testid$="-list-page"]'
    },
    empty_state_loaded: {
        readySelector: '[data-testid="empty-state"]'
    },
    selected_connector_state: {
        historyState: {
            connectorId: "test",
            connectorsPresent: true
        },
        readySelector: '[data-testid="connector-list-page"]'
    },
    selected_graph_state: {
        historyState: {
            graphId: "customer_graph",
            graphName: "customer_graph"
        },
        readySelector: '[data-testid="graph-list-page"]'
    },
    selected_table_state: {
        historyState: {
            tableId: "transaction",
            tableName: "transaction"
        },
        readySelector: '[data-testid="table-list-page"]'
    }
};

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

export function parseArgs(argv) {
    const args = { baseUrl: "http://localhost:3000" };
    for (let i = 2; i < argv.length; i++) {
        switch (argv[i]) {
            case "--routes":
                args.routes = argv[++i].split(",").map((r) => r.trim());
                break;
            case "--plan":
                args.plan = argv[++i];
                break;
            case "--output":
                args.output = argv[++i];
                break;
            case "--base-url":
                args.baseUrl = argv[++i];
                break;
            case "--auth-email":
                args.authEmail = argv[++i];
                break;
            case "--auth-password":
                args.authPassword = argv[++i];
                break;
            case "--ready-selectors":
                args.readySelectors = parseReadySelectors(argv[++i]);
                break;
            case "--auth-disabled":
                args.authDisabled = true;
                break;
            case "--help":
                console.log(
                    "Usage: ci-screenshot.mjs (--plan plan.json | --routes /a,/b) --output <dir> [--base-url url] [--ready-selectors json] [--auth-disabled] [--auth-email email --auth-password password]"
                );
                console.log(
                    "  --auth-disabled (or SCREENSHOT_AUTH_DISABLED=1): use the local auth-disabled sentinel. Use against an auth-disabled backend (e.g. local dev)."
                );
                console.log(
                    '  Ready selectors JSON example: {"\\/graphs\\/":"[data-testid=\\"graph-page\\"]","*":"main"}'
                );
                process.exit(0);
        }
    }

    if (!args.authEmail) {
        args.authEmail =
            process.env.SCREENSHOT_AUTH_EMAIL ?? process.env.TEST_EMAIL;
    }
    if (!args.authPassword) {
        args.authPassword =
            process.env.SCREENSHOT_AUTH_PASSWORD ?? process.env.TEST_PASSWORD;
    }

    if (!args.readySelectors && process.env.CI_SCREENSHOT_READY_SELECTORS) {
        args.readySelectors = parseReadySelectors(
            process.env.CI_SCREENSHOT_READY_SELECTORS
        );
    }

    if (!args.authDisabled && process.env.SCREENSHOT_AUTH_DISABLED === "1") {
        args.authDisabled = true;
    }

    // Structural validation only. The auth-input requirement is enforced in
    // main() after getAuthClientConfig(), so a backend that reports auth_disabled:true
    // can satisfy it without the operator passing --auth-disabled (see
    // resolveAuthStrategy / isAuthDisabledMode). In auth-disabled mode the
    // backend issues no real session, so no credentials are required.
    if ((!args.routes && !args.plan) || !args.output) {
        console.error(
            "Missing required args. Provide a plan or routes plus an output directory. Run with --help for usage."
        );
        process.exit(1);
    }
    return args;
}

export function parseReadySelectors(rawSelectors) {
    if (!rawSelectors) {
        return {};
    }

    try {
        const parsed = JSON.parse(rawSelectors);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
            throw new Error("ready selectors must be a JSON object");
        }
        return parsed;
    } catch (error) {
        console.error(`Invalid ready selectors JSON: ${error.message}`);
        process.exit(1);
    }
}

// ---------------------------------------------------------------------------
// Fetch auth client config from the backend (via the Vite proxy)
// ---------------------------------------------------------------------------

async function getAuthClientConfig(baseUrl) {
    const url = `${baseUrl}/v0/authentication/client-config`;
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(
            `Failed to fetch auth client config: ${res.status} ${res.statusText}`
        );
    }
    const data = await res.json();
    return {
        authDisabled: Boolean(data.auth_disabled),
        directConnectLoginMethod: data.direct_connect_login_method || "oidc"
    };
}

// Auth-disabled mode is active when the operator opts in (--auth-disabled /
// SCREENSHOT_AUTH_DISABLED) or the backend itself reports auth_disabled.
export function isAuthDisabledMode(args, authConfig) {
    return Boolean(args?.authDisabled) || Boolean(authConfig?.authDisabled);
}

// Decide how to authenticate the capture. Direct Connect password credentials
// always take precedence and follow the browser login path, even on localhost.
// Auth-disabled mode only changes behavior when no credentials are available.
export function resolveAuthStrategy(args, authConfig) {
    const authDisabledMode = isAuthDisabledMode(args, authConfig);
    const hasRealCreds = Boolean(args?.authEmail && args?.authPassword);
    const useSentinel = authDisabledMode && !hasRealCreds;
    const backendRequiresAuth =
        useSentinel && authConfig?.authDisabled === false;
    return { authDisabledMode, hasRealCreds, useSentinel, backendRequiresAuth };
}

function responseOk(response) {
    return typeof response.ok === "function" ? response.ok() : response.ok;
}

function responseStatus(response) {
    return typeof response.status === "function"
        ? response.status()
        : response.status;
}

function responseStatusText(response) {
    return typeof response.statusText === "function"
        ? response.statusText()
        : response.statusText;
}

export async function fetchFeatureFlags(baseUrl, context) {
    const url = `${baseUrl}/v0/constants/flags`;
    const response = context
        ? await context.request.get(url)
        : await fetch(url);

    if (!responseOk(response)) {
        console.log(
            `[feature-flags] failed to fetch ${url}: ${responseStatus(response)} ${responseStatusText(response)}`
        );
        return {};
    }

    const payload = await response.json();
    return payload && typeof payload === "object" ? payload : {};
}

// ---------------------------------------------------------------------------
// Inject auth state into the browser's localStorage
// ---------------------------------------------------------------------------

async function installAuthBootstrap(context) {
    await context.addInitScript(
        ({ authKey, sentinel }) => {
            // Match the encoding used by storeToLocalStorage(key, value, encode=true)
            // in ui/src/utils/storage.ts — JSON.stringify(base64Encode(value)).
            localStorage.setItem("accessToken", JSON.stringify(btoa(sentinel)));
            localStorage.setItem(authKey, JSON.stringify(true));

            // userDetails is stored with encode=false (plain JSON.stringify)
            localStorage.setItem(
                "userDetails",
                JSON.stringify({
                    first_name: "CI",
                    last_name: "Bot",
                    email: "ci@kumo.ai",
                    isLoggedIn: true,
                    picture: "",
                    user_id: "ci-bot",
                    name: "CI Bot"
                })
            );
        },
        {
            authKey: CI_SCREENSHOT_AUTH_KEY,
            sentinel: AUTH_DISABLED_SENTINEL
        }
    );
}

async function loginWithBrowserFlow(page, baseUrl, authEmail, authPassword) {
    console.log("Authenticating via Direct Connect password login...");
    await page.goto(`${baseUrl}/login`, {
        waitUntil: "load",
        timeout: 30_000
    });

    const usernameInput = page.getByLabel("Email Address");
    const passwordInput = page.getByLabel("Password");
    const signInButton = page.getByRole("button", {
        name: "Sign in",
        exact: true
    });

    await usernameInput.waitFor({ state: "visible", timeout: 30_000 });
    await usernameInput.fill(authEmail);
    await passwordInput.fill(authPassword);
    await signInButton.click();

    await page.waitForURL(
        (url) => new URL(url).origin === new URL(baseUrl).origin,
        { timeout: 60_000 }
    );

    await page.waitForLoadState("networkidle").catch(() => {
        console.log("  Warning: networkidle timed out after login, continuing");
    });

    try {
        await page.waitForFunction(
            () => Boolean(localStorage.getItem("userDetails")),
            { timeout: 60_000 }
        );
    } catch (error) {
        const timeoutSnapshot = await page.evaluate(() => ({
            url: window.location.href,
            pathname: window.location.pathname,
            localStorageKeys: Object.keys(localStorage),
            accessToken: localStorage.getItem("accessToken"),
            userDetails: localStorage.getItem("userDetails")
        }));
        console.log(
            `[browser-auth-timeout] ${JSON.stringify(timeoutSnapshot)}`
        );
        throw error;
    }

    await page
        .getByTestId("alert-dialog-content")
        .waitFor({ state: "visible", timeout: 5_000 })
        .then(async () => {
            await page.getByTestId("cancel-icon").click();
            await page
                .getByTestId("alert-dialog-content")
                .waitFor({ state: "hidden", timeout: 10_000 });
        })
        .catch(() => {});

    await page.evaluate(
        ({ authKey }) => {
            localStorage.setItem(authKey, JSON.stringify(true));
            if (!localStorage.getItem("userDetails")) {
                localStorage.setItem(
                    "userDetails",
                    JSON.stringify({
                        first_name: "CI",
                        last_name: "Bot",
                        email: "ci@kumo.ai",
                        isLoggedIn: true,
                        picture: "",
                        user_id: "ci-bot",
                        name: "CI Bot"
                    })
                );
            }
        },
        { authKey: CI_SCREENSHOT_AUTH_KEY }
    );

    const authSnapshot = await page.evaluate(() => ({
        url: window.location.href,
        pathname: window.location.pathname,
        userDetails: localStorage.getItem("userDetails")
    }));
    console.log(`[browser-auth-snapshot] ${JSON.stringify(authSnapshot)}`);
}

export function normalizePath(route) {
    if (!route || route === "/") {
        return "/";
    }

    return `/${route.replace(/^\/+|\/+$/g, "")}/`;
}

export function getRouteFileStem(route) {
    return route.replace(/^\/|\/$/g, "").replace(/\//g, "-") || "home";
}

export function getRouteReadySelectors(route, readySelectors = {}) {
    const normalizedRoute = normalizePath(route);
    const exactMatch =
        readySelectors[route] ??
        readySelectors[normalizedRoute] ??
        readySelectors[normalizedRoute.replace(/\/$/, "")];
    const fallbackSelectors =
        exactMatch ?? readySelectors["*"] ?? '[data-testid="main-layout"]';

    return Array.isArray(fallbackSelectors)
        ? fallbackSelectors
        : [fallbackSelectors];
}

export function getScenarioFileStem(name, route) {
    const base = name || getRouteFileStem(route);
    return (
        base
            .trim()
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/^-+|-+$/g, "") || getRouteFileStem(route)
    );
}

function normalizeSelectorList(selectors) {
    if (!selectors) {
        return [];
    }
    return Array.isArray(selectors) ? selectors : [selectors];
}

function normalizeCoverageList(value) {
    if (typeof value === "string" && value.trim()) {
        return [value.trim()];
    }
    if (Array.isArray(value)) {
        return value
            .filter((item) => typeof item === "string" && item.trim())
            .map((item) => item.trim());
    }
    return [];
}

function normalizeVisualCoverage(value) {
    if (!Array.isArray(value)) {
        return [];
    }

    return value.filter(
        (item) =>
            item &&
            typeof item === "object" &&
            !Array.isArray(item) &&
            typeof item.id === "string" &&
            item.id.trim() &&
            typeof item.label === "string" &&
            item.label.trim()
    );
}

function normalizeRequiredFeatureFlags(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        return {};
    }

    return Object.fromEntries(
        Object.entries(value).filter(
            ([key, expected]) =>
                typeof key === "string" &&
                key.trim() &&
                typeof expected === "boolean"
        )
    );
}

function normalizeFixtureNames(value) {
    if (typeof value === "string" && value.trim()) {
        return [value.trim()];
    }
    if (Array.isArray(value)) {
        return value
            .filter((item) => typeof item === "string" && item.trim())
            .map((item) => item.trim());
    }
    return [];
}

function normalizePresentation(value) {
    return value === "primary" || value === "regression" ? value : null;
}

function normalizeFixtureMap(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        return {};
    }

    const fixtures = {};
    for (const [name, fixture] of Object.entries(value)) {
        if (
            typeof name === "string" &&
            name.trim() &&
            fixture &&
            typeof fixture === "object" &&
            !Array.isArray(fixture)
        ) {
            fixtures[name.trim()] = fixture;
        }
    }
    return fixtures;
}

function mergeBootstrapMaps(...maps) {
    const merged = {};
    for (const map of maps) {
        const normalized = normalizeBootstrapMap(map);
        if (normalized) {
            Object.assign(merged, normalized);
        }
    }
    return Object.keys(merged).length > 0 ? merged : null;
}

function getFixtureBundle(fixtureNames, fixtureMap, scenarioIndex) {
    const bundle = {
        historyState: null,
        localStorage: null,
        sessionStorage: null,
        setup: [],
        assertions: [],
        readySelectors: []
    };

    for (const fixtureName of fixtureNames) {
        const fixture = fixtureMap[fixtureName];
        if (!fixture) {
            throw new Error(
                `scenario ${scenarioIndex} references unknown fixture "${fixtureName}"`
            );
        }

        bundle.historyState = mergeBootstrapMaps(
            bundle.historyState,
            fixture.historyState
        );
        bundle.localStorage = mergeBootstrapMaps(
            bundle.localStorage,
            fixture.localStorage
        );
        bundle.sessionStorage = mergeBootstrapMaps(
            bundle.sessionStorage,
            fixture.sessionStorage
        );
        if (Array.isArray(fixture.setup)) {
            bundle.setup.push(...fixture.setup);
        }
        if (Array.isArray(fixture.assertions)) {
            bundle.assertions.push(...fixture.assertions);
        }
        bundle.readySelectors.push(
            ...normalizeSelectorList(
                fixture.readySelector ?? fixture.readySelectors
            )
        );
    }

    return bundle;
}

export function normalizeScreenshotPlan(plan) {
    if (!plan || typeof plan !== "object" || Array.isArray(plan)) {
        throw new Error("screenshot plan must be a JSON object");
    }

    const hasRoutes = Array.isArray(plan.routes) && plan.routes.length > 0;
    const hasScenarios =
        Array.isArray(plan.scenarios) && plan.scenarios.length > 0;

    if (!hasRoutes && !hasScenarios) {
        throw new Error(
            "screenshot plan must contain a non-empty routes or scenarios array"
        );
    }

    if (hasRoutes && hasScenarios) {
        throw new Error(
            "screenshot plan must not define both routes and scenarios"
        );
    }

    const readySelectors = plan.readySelectors ?? {};
    const fixtureMap = {
        ...DEFAULT_SCREENSHOT_FIXTURES,
        ...normalizeFixtureMap(plan.fixtures)
    };
    const visualCoverage = normalizeVisualCoverage(plan._visualCoverage);
    const requiredFeatureFlags = normalizeRequiredFeatureFlags(
        plan._requiredFeatureFlags
    );
    const recipeKey =
        typeof plan._recipeKey === "string" && plan._recipeKey.trim()
            ? plan._recipeKey.trim()
            : null;
    if (
        readySelectors === null ||
        typeof readySelectors !== "object" ||
        Array.isArray(readySelectors)
    ) {
        throw new Error(
            "screenshot plan readySelectors must be an object when present"
        );
    }

    if (hasScenarios) {
        return plan.scenarios.map((scenario, index) => {
            if (
                !scenario ||
                typeof scenario !== "object" ||
                Array.isArray(scenario)
            ) {
                throw new Error(`scenario ${index} must be an object`);
            }
            const route = scenario.route?.trim();
            if (!route || !route.startsWith("/")) {
                throw new Error(`scenario ${index} route must start with '/'`);
            }
            const name =
                typeof scenario.name === "string" && scenario.name.trim()
                    ? scenario.name.trim()
                    : getScenarioFileStem("", route);
            const fixtureNames = normalizeFixtureNames(scenario.fixtures);
            const fixtureBundle = getFixtureBundle(
                fixtureNames,
                fixtureMap,
                index
            );
            const scenarioReadySelectors = normalizeSelectorList(
                scenario.readySelector ?? scenario.readySelectors
            );
            return {
                name,
                route,
                fixtures: fixtureNames,
                historyState: mergeBootstrapMaps(
                    fixtureBundle.historyState,
                    scenario.historyState
                ),
                localStorage: mergeBootstrapMaps(
                    fixtureBundle.localStorage,
                    scenario.localStorage
                ),
                sessionStorage: mergeBootstrapMaps(
                    fixtureBundle.sessionStorage,
                    scenario.sessionStorage
                ),
                setup: [
                    ...fixtureBundle.setup,
                    ...(Array.isArray(scenario.setup) ? scenario.setup : [])
                ],
                assertions: [
                    ...fixtureBundle.assertions,
                    ...(Array.isArray(scenario.assertions)
                        ? scenario.assertions
                        : [])
                ],
                presentation: normalizePresentation(scenario.presentation),
                recipeKey,
                requiredFeatureFlags: {
                    ...requiredFeatureFlags,
                    ...normalizeRequiredFeatureFlags(
                        scenario.requiredFeatureFlags
                    )
                },
                visualCoverage:
                    normalizeVisualCoverage(scenario.visualCoverage).length > 0
                        ? normalizeVisualCoverage(scenario.visualCoverage)
                        : visualCoverage,
                readySelectors:
                    scenarioReadySelectors.length > 0
                        ? scenarioReadySelectors
                        : fixtureBundle.readySelectors.length > 0
                          ? fixtureBundle.readySelectors
                          : getRouteReadySelectors(route, readySelectors)
            };
        });
    }

    return plan.routes.map((route, index) => {
        if (typeof route !== "string" || !route.trim()) {
            throw new Error(`route ${index} must be a non-empty string`);
        }
        const normalizedRoute = route.trim();
        if (!normalizedRoute.startsWith("/")) {
            throw new Error(`route ${index} must start with '/'`);
        }

        return {
            name: getScenarioFileStem("", normalizedRoute),
            route: normalizedRoute,
            historyState: null,
            localStorage: null,
            sessionStorage: null,
            fixtures: [],
            setup: [],
            assertions: [],
            presentation: null,
            recipeKey,
            requiredFeatureFlags,
            visualCoverage,
            readySelectors: getRouteReadySelectors(
                normalizedRoute,
                readySelectors
            )
        };
    });
}

function readPlanFile(planPath) {
    const raw = fs.readFileSync(planPath, "utf8");
    return normalizeScreenshotPlan(JSON.parse(raw));
}

function normalizeBootstrapMap(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        return null;
    }

    const normalized = {};
    for (const [key, entry] of Object.entries(value)) {
        if (typeof key === "string" && key.trim()) {
            normalized[key] = entry;
        }
    }

    return Object.keys(normalized).length > 0 ? normalized : null;
}

export async function assertExpectedRoute(page, route, baseUrl) {
    const currentUrl = new URL(page.url());
    const expectedBaseUrl = new URL(baseUrl);
    const requestedUrl = new URL(route, expectedBaseUrl);
    const requestedPath = normalizePath(requestedUrl.pathname);
    const currentPath = normalizePath(currentUrl.pathname);

    if (currentUrl.origin !== expectedBaseUrl.origin) {
        throw new Error(
            `redirected to external origin ${currentUrl.origin} instead of ${expectedBaseUrl.origin}${requestedPath}`
        );
    }

    if (currentPath === "/login/") {
        throw new Error(`redirected to login instead of ${requestedPath}`);
    }

    if (requestedPath !== "/" && !currentPath.startsWith(requestedPath)) {
        throw new Error(
            `loaded unexpected route ${currentPath} instead of ${requestedPath}`
        );
    }
}

async function assertNoKnownErrorState(page) {
    const bodyText = (await page.textContent("body")) || "";
    const matchedPattern = ERROR_PATTERNS.find((pattern) =>
        pattern.test(bodyText)
    );

    if (matchedPattern) {
        throw new Error(
            `page rendered an error state matching ${matchedPattern}`
        );
    }
}

async function waitForReadySelectors(page, route, readySelectors) {
    let lastError;
    for (const selector of readySelectors) {
        try {
            await page.waitForSelector(selector, {
                timeout: READY_SELECTOR_TIMEOUT_MS
            });
            return selector;
        } catch (error) {
            lastError = error;
        }
    }

    throw new Error(
        `ready selector not found for ${route}: ${readySelectors.join(", ")}${
            lastError ? ` (${lastError.message})` : ""
        }`
    );
}

async function applyHistoryStateAndReload(page, historyState) {
    if (!historyState) {
        return;
    }

    await page.evaluate((state) => {
        const current = window.history.state ?? {};
        const currentUsr =
            current.usr && typeof current.usr === "object" ? current.usr : {};
        window.history.replaceState(
            {
                ...current,
                usr: {
                    ...currentUsr,
                    ...state
                }
            },
            "",
            window.location.href
        );
    }, historyState);
    await page.reload({ waitUntil: "domcontentloaded" });
}

async function seedStorageAndReload(page, storageKind, values) {
    if (!values || typeof values !== "object") {
        return;
    }

    await page.evaluate(
        ({ storageKind, values }) => {
            const storage =
                storageKind === "sessionStorage"
                    ? window.sessionStorage
                    : window.localStorage;
            for (const [key, value] of Object.entries(values)) {
                storage.setItem(key, JSON.stringify(value));
            }
        },
        { storageKind, values }
    );
    await page.reload({ waitUntil: "domcontentloaded" });
}

async function runSetupActions(page, scenario) {
    const covered = [];
    for (const [index, action] of scenario.setup.entries()) {
        const label = `${scenario.name} setup[${index}]`;
        if (!action || typeof action !== "object" || Array.isArray(action)) {
            throw new Error(`${label}: action must be an object`);
        }

        const selector = action.selector;
        const kind = action.action;
        if (typeof selector !== "string" || !selector.trim()) {
            throw new Error(`${label}: selector is required`);
        }

        console.log(`  Running ${label}: ${kind} ${selector}`);
        const locator = page.locator(selector);
        const timeout = action.timeoutMs ?? READY_SELECTOR_TIMEOUT_MS;

        switch (kind) {
            case "click":
                await locator.click({ timeout });
                break;
            case "fill":
                if (typeof action.value !== "string") {
                    throw new Error(`${label}: fill requires string value`);
                }
                await locator.fill(action.value, { timeout });
                break;
            case "type":
                if (typeof action.value !== "string") {
                    throw new Error(`${label}: type requires string value`);
                }
                await locator.click({ timeout });
                await page.keyboard.insertText(action.value);
                break;
            case "paste":
                if (typeof action.value !== "string") {
                    throw new Error(`${label}: paste requires string value`);
                }
                await locator.evaluate((element, value) => {
                    element.focus();
                    const data = new DataTransfer();
                    data.setData("text/plain", value);
                    const event = new ClipboardEvent("paste", {
                        clipboardData: data,
                        bubbles: true,
                        cancelable: true
                    });
                    element.dispatchEvent(event);
                }, action.value);
                break;
            case "press":
                if (typeof action.key !== "string" || !action.key.trim()) {
                    throw new Error(`${label}: press requires key`);
                }
                await locator.press(action.key, { timeout });
                break;
            case "select":
                if (
                    typeof action.value !== "string" &&
                    !(
                        Array.isArray(action.value) &&
                        action.value.every(
                            (value) => typeof value === "string" && value.trim()
                        )
                    )
                ) {
                    throw new Error(
                        `${label}: select requires string or string[] value`
                    );
                }
                await locator.selectOption(action.value, { timeout });
                break;
            case "waitFor":
                await locator.waitFor({
                    state: action.state ?? "visible",
                    timeout
                });
                break;
            default:
                throw new Error(`${label}: unsupported action ${kind}`);
        }
        covered.push(...normalizeCoverageList(action.covers));
    }
    return covered;
}

async function runAssertions(page, scenario) {
    const covered = [];
    for (const [index, assertion] of scenario.assertions.entries()) {
        const label = `${scenario.name} assertion[${index}]`;
        if (
            !assertion ||
            typeof assertion !== "object" ||
            Array.isArray(assertion)
        ) {
            throw new Error(`${label}: assertion must be an object`);
        }

        const selector = assertion.selector;
        if (typeof selector !== "string" || !selector.trim()) {
            throw new Error(`${label}: selector is required`);
        }

        const locator = page.locator(selector);
        switch (assertion.type) {
            case "visible":
                await locator.waitFor({
                    state: "visible",
                    timeout: assertion.timeoutMs ?? READY_SELECTOR_TIMEOUT_MS
                });
                break;
            case "text": {
                const text = ((await locator.textContent()) || "").trim();
                if (typeof assertion.contains === "string") {
                    if (!text.includes(assertion.contains)) {
                        throw new Error(
                            `${label}: expected text containing "${assertion.contains}" but got "${text}"`
                        );
                    }
                } else if (typeof assertion.equals === "string") {
                    if (text !== assertion.equals) {
                        throw new Error(
                            `${label}: expected text "${assertion.equals}" but got "${text}"`
                        );
                    }
                } else if (typeof assertion.matches === "string") {
                    const pattern = new RegExp(assertion.matches);
                    if (!pattern.test(text)) {
                        throw new Error(
                            `${label}: expected text matching /${assertion.matches}/ but got "${text}"`
                        );
                    }
                } else {
                    throw new Error(
                        `${label}: text assertion requires contains, equals, or matches`
                    );
                }
                break;
            }
            default:
                throw new Error(
                    `${label}: unsupported assertion type ${assertion.type}`
                );
        }
        covered.push(...normalizeCoverageList(assertion.covers));
    }
    return covered;
}

function getCoverageStatus(scenario, coveredIds) {
    const covered = new Set(coveredIds);
    return scenario.visualCoverage.map((coverage) => ({
        id: coverage.id,
        label: coverage.label,
        passed: covered.has(coverage.id)
    }));
}

function getSkippedCoverageStatus(scenario, reason) {
    return scenario.visualCoverage.map((coverage) => ({
        id: coverage.id,
        label: coverage.label,
        passed: true,
        skipped: true,
        reason
    }));
}

export function getMissingCoverage(results) {
    return results.flatMap((result) =>
        (result.coverage || [])
            .filter((coverage) => coverage.passed !== true)
            .map((coverage) => ({
                route: result.route,
                scenarioName: result.scenarioName,
                id: coverage.id,
                label: coverage.label
            }))
    );
}

export function shouldSkipScenarioForFeatureFlags(scenario, featureFlags = {}) {
    const requiredFeatureFlags = scenario.requiredFeatureFlags || {};
    const mismatches = Object.entries(requiredFeatureFlags).filter(
        ([flag, expected]) => featureFlags?.[flag] !== expected
    );

    if (mismatches.length === 0) {
        return null;
    }

    return `required feature flags not satisfied: ${mismatches
        .map(
            ([flag, expected]) =>
                `${flag}=${expected} (actual ${featureFlags?.[flag]})`
        )
        .join(", ")}`;
}

export function classifyScreenshotFailure(errorMessage) {
    const message = `${errorMessage || ""}`;
    const lowered = message.toLowerCase();

    if (
        lowered.includes("missing required permissions") ||
        lowered.includes("authenticated api preflight failed") ||
        lowered.includes("browser login") ||
        lowered.includes("access token") ||
        lowered.includes("accesstoken") ||
        lowered.includes("redirected to login") ||
        lowered.includes("redirected to external origin") ||
        lowered.includes("login required")
    ) {
        return "auth";
    }

    if (
        lowered.includes("loaded unexpected route") ||
        lowered.includes("instead of")
    ) {
        return "route-mismatch";
    }

    if (
        lowered.includes("ready selector not found") ||
        lowered.includes("waiting for selector") ||
        lowered.includes("waiting for locator") ||
        lowered.includes("timeout")
    ) {
        return "selector-readiness";
    }

    if (
        lowered.includes("page rendered an error state") ||
        lowered.includes("something went wrong") ||
        lowered.includes("service temporarily unavailable") ||
        lowered.includes("api-response") ||
        lowered.includes("request-failed")
    ) {
        return "app-runtime";
    }

    if (lowered.includes("coverage")) {
        return "coverage-missing";
    }

    return "unknown";
}

async function waitForLoadingOverlayToClear(page) {
    const overlay = page.locator(LOADING_OVERLAY_SELECTOR);
    try {
        await overlay.waitFor({
            state: "hidden",
            timeout: READY_SELECTOR_TIMEOUT_MS
        });
        console.log("  Loading overlay cleared");
    } catch {
        if ((await overlay.count()) > 0) {
            console.log(
                "  Warning: loading overlay still visible after timeout, continuing"
            );
        }
    }
}

async function logAuthSnapshot(page, label) {
    const snapshot = await page.evaluate(() => ({
        url: window.location.href,
        pathname: window.location.pathname,
        ciScreenshotAuth: localStorage.getItem("__ciScreenshotAuth"),
        accessToken: localStorage.getItem("accessToken")
    }));
    console.log(`[auth-snapshot:${label}] ${JSON.stringify(snapshot)}`);
}

function recordDebugEvent(events, event) {
    events.push({
        ...event,
        timestamp: new Date().toISOString()
    });
    if (events.length > 100) {
        events.shift();
    }
}

function attachPageDebugLogging(page, baseUrl, events) {
    const apiOrigin = new URL(baseUrl).origin;

    page.on("console", (message) => {
        const type = message.type();
        if (type === "error" || type === "warning") {
            const text = message.text();
            recordDebugEvent(events, {
                type: "console",
                level: type,
                text
            });
            console.log(`[page-console:${type}] ${text}`);
        }
    });

    page.on("pageerror", (error) => {
        recordDebugEvent(events, {
            type: "pageerror",
            text: error.message
        });
        console.log(`[page-error] ${error.message}`);
    });

    page.on("response", async (response) => {
        const url = response.url();
        if (
            (url.startsWith(`${apiOrigin}/v0/`) ||
                url.startsWith(
                    `${apiOrigin.replace("localhost", "127.0.0.1")}/v0/`
                ) ||
                url.startsWith(`${apiOrigin}/v1/`) ||
                url.startsWith(
                    `${apiOrigin.replace("localhost", "127.0.0.1")}/v1/`
                )) &&
            response.status() >= 400
        ) {
            recordDebugEvent(events, {
                type: "api-response",
                status: response.status(),
                url
            });
            console.log(`[api-response:${response.status()}] ${url}`);
        }
    });

    page.on("requestfailed", (request) => {
        const failure = request.failure();
        recordDebugEvent(events, {
            type: "request-failed",
            method: request.method(),
            url: request.url(),
            errorText: failure?.errorText || "unknown"
        });
        console.log(
            `[request-failed] ${request.method()} ${request.url()} :: ${failure?.errorText || "unknown"}`
        );
    });
}

async function getPageDebugSnapshot(page) {
    try {
        return await page.evaluate(() => ({
            title: document.title,
            url: window.location.href,
            pathname: window.location.pathname,
            bodyText: document.body?.innerText?.slice(0, 2000) ?? "",
            authState: {
                ciScreenshotAuth: localStorage.getItem("__ciScreenshotAuth"),
                hasAccessToken: Boolean(localStorage.getItem("accessToken"))
            }
        }));
    } catch (snapshotError) {
        return {
            snapshotError: snapshotError.message
        };
    }
}

async function writeFailureArtifacts(
    page,
    scenario,
    config,
    error,
    debugEvents
) {
    const routeStem = getScenarioFileStem(scenario.name, scenario.route);
    const failureDir = path.join(config.output, "failed");
    const htmlPath = path.join(failureDir, `${routeStem}.html`);
    const screenshotPath = path.join(failureDir, `${routeStem}.png`);
    const metadataPath = path.join(failureDir, `${routeStem}.json`);

    fs.mkdirSync(failureDir, { recursive: true });

    const metadata = {
        route: scenario.route,
        scenarioName: scenario.name,
        finalUrl: page.url(),
        error: error.message,
        failureType: classifyScreenshotFailure(error.message),
        recipeKey: scenario.recipeKey,
        readySelectors: scenario.readySelectors,
        assertions: scenario.assertions,
        setup: scenario.setup,
        pageSnapshot: await getPageDebugSnapshot(page),
        debugEvents,
        timestamp: new Date().toISOString()
    };

    try {
        fs.writeFileSync(htmlPath, await page.content(), "utf8");
    } catch (writeError) {
        metadata.htmlWriteError = writeError.message;
    }

    try {
        await page.screenshot({ fullPage: true, path: screenshotPath });
    } catch (writeError) {
        metadata.screenshotWriteError = writeError.message;
    }

    fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
    return {
        htmlPath,
        screenshotPath,
        metadataPath,
        finalUrl: metadata.finalUrl
    };
}

export function getManifestPath(outputDir) {
    return path.join(outputDir, "results.json");
}

function writeManifest(outputDir, manifest) {
    fs.writeFileSync(
        getManifestPath(outputDir),
        JSON.stringify(manifest, null, 2)
    );
}

export function buildScreenshotDebugMarkdown(manifest) {
    const lines = [
        "# Screenshot Debug Summary",
        "",
        `Generated: ${manifest.generatedAt ?? "unknown"}`,
        `Output directory: ${manifest.outputDir ?? "unknown"}`,
        `Succeeded: ${manifest.succeeded ?? 0}`,
        `Failed: ${manifest.failed ?? 0}`,
        `Coverage failed: ${manifest.coverageFailed ?? 0}`,
        `Failure types: ${(manifest.failureTypes ?? []).join(", ") || "none"}`,
        ""
    ];

    if (
        Array.isArray(manifest.missingCoverage) &&
        manifest.missingCoverage.length > 0
    ) {
        lines.push("## Missing Coverage", "");
        for (const coverage of manifest.missingCoverage) {
            lines.push(
                `- ${coverage.scenarioName ?? "unknown"} ${coverage.route ?? "unknown"}: ${coverage.id} - ${coverage.label}`
            );
        }
        lines.push("");
    }

    for (const result of manifest.results ?? []) {
        const resultStatus = result.skipped
            ? "Skipped"
            : result.success
              ? "Captured"
              : "Failed";
        lines.push(
            `## ${resultStatus}: ${result.scenarioName ?? result.route ?? "unknown"}`
        );
        lines.push("");
        lines.push(`- Route: ${result.route ?? "unknown"}`);
        lines.push(`- Final URL: ${result.finalUrl ?? "unknown"}`);
        lines.push(`- Recipe: ${result.recipeKey ?? "none"}`);
        if (result.failureType) {
            lines.push(`- Failure type: ${result.failureType}`);
        }
        if (result.error) {
            lines.push(`- Error: ${result.error}`);
        }
        if (result.skipReason) {
            lines.push(`- Skip reason: ${result.skipReason}`);
        }
        if (result.matchedSelector) {
            lines.push(`- Ready selector: ${result.matchedSelector}`);
        }
        if (Array.isArray(result.readySelectors)) {
            lines.push(
                `- Candidate ready selectors: ${result.readySelectors.join(", ")}`
            );
        }
        if (Array.isArray(result.coverage) && result.coverage.length > 0) {
            const coverageSummary = result.coverage
                .map(
                    (coverage) =>
                        `${coverage.skipped ? "skipped" : coverage.passed ? "pass" : "missing"}:${coverage.id}`
                )
                .join(", ");
            lines.push(`- Coverage: ${coverageSummary}`);
        }
        if (result.pageSnapshot) {
            lines.push(
                `- DOM title: ${result.pageSnapshot.title ?? "unknown"}`
            );
            lines.push(
                `- DOM path: ${result.pageSnapshot.pathname ?? "unknown"}`
            );
            if (result.pageSnapshot.authState) {
                lines.push(
                    `- Auth state: local=${result.pageSnapshot.authState.hasAccessToken ? "present" : "missing"}`
                );
            }
        }
        const debugEvents = Array.isArray(result.debugEvents)
            ? result.debugEvents.slice(-10)
            : [];
        if (debugEvents.length > 0) {
            lines.push("- Recent page events:");
            for (const event of debugEvents) {
                const label =
                    event.type === "api-response"
                        ? `${event.type}:${event.status} ${event.url}`
                        : event.type === "request-failed"
                          ? `${event.type} ${event.method} ${event.url} ${event.errorText}`
                          : `${event.type}:${event.level ?? ""} ${event.text}`;
                lines.push(`  - ${label.trim()}`);
            }
        }
        lines.push("");
    }

    return `${lines.join("\n")}\n`;
}

function writeScreenshotDebugSummary(outputDir, manifest) {
    fs.writeFileSync(
        path.join(outputDir, "screenshot-debug.md"),
        buildScreenshotDebugMarkdown(manifest)
    );
}

// ---------------------------------------------------------------------------
// Capture a single scenario
// ---------------------------------------------------------------------------

async function captureScenario(browser, scenario, config) {
    const context = await browser.newContext({
        viewport: { width: 1400, height: 900 },
        baseURL: config.baseUrl,
        ...(config.storageState ? { storageState: config.storageState } : {})
    });
    if (!config.storageState) {
        await installAuthBootstrap(context);
    }
    const page = await context.newPage();
    const debugEvents = [];
    attachPageDebugLogging(page, config.baseUrl, debugEvents);
    const pendingCoveredIds = [];

    try {
        // Navigate to the target route
        const targetUrl = `${config.baseUrl}${scenario.route}`;
        console.log(`  Navigating to ${targetUrl}`);
        await page.goto(targetUrl, {
            waitUntil: "load",
            timeout: 30_000
        });
        if (scenario.historyState) {
            await applyHistoryStateAndReload(page, scenario.historyState);
        }
        if (scenario.localStorage) {
            await seedStorageAndReload(
                page,
                "localStorage",
                scenario.localStorage
            );
        }
        if (scenario.sessionStorage) {
            await seedStorageAndReload(
                page,
                "sessionStorage",
                scenario.sessionStorage
            );
        }
        await logAuthSnapshot(page, "post-goto");

        await assertExpectedRoute(page, scenario.route, config.baseUrl);
        await waitForLoadingOverlayToClear(page);

        if (scenario.setup.length > 0) {
            pendingCoveredIds.push(...(await runSetupActions(page, scenario)));
            await waitForLoadingOverlayToClear(page);
            await page.waitForLoadState("networkidle").catch(() => {
                console.log(
                    "  Warning: networkidle timed out after setup, continuing"
                );
            });
        }

        const matchedSelector = await waitForReadySelectors(
            page,
            scenario.route,
            scenario.readySelectors
        );
        console.log(`  Ready selector matched: ${matchedSelector}`);
        await logAuthSnapshot(page, "post-selector");
        await assertExpectedRoute(page, scenario.route, config.baseUrl);
        pendingCoveredIds.push(...(await runAssertions(page, scenario)));
        await page.waitForLoadState("networkidle").catch(() => {
            console.log("  Warning: networkidle timed out, continuing");
        });
        await waitForLoadingOverlayToClear(page);
        await logAuthSnapshot(page, "pre-screenshot");
        await assertExpectedRoute(page, scenario.route, config.baseUrl);
        await assertNoKnownErrorState(page);

        const name = getScenarioFileStem(scenario.name, scenario.route);
        const filePath = path.join(config.output, `${name}.png`);
        await page.screenshot({ fullPage: true, path: filePath });
        const coveredIds = pendingCoveredIds;
        console.log(`  Saved ${filePath}`);
        return {
            route: scenario.route,
            scenarioName: scenario.name,
            success: true,
            filePath,
            finalUrl: page.url(),
            matchedSelector,
            presentation: scenario.presentation,
            recipeKey: scenario.recipeKey,
            readySelectors: scenario.readySelectors,
            coverage: getCoverageStatus(scenario, coveredIds),
            debugEvents
        };
    } catch (err) {
        const failureArtifacts = await writeFailureArtifacts(
            page,
            scenario,
            config,
            err,
            debugEvents
        );
        console.error(`  Failed to capture ${scenario.route}: ${err.message}`);
        return {
            route: scenario.route,
            scenarioName: scenario.name,
            success: false,
            error: err.message,
            failureType: classifyScreenshotFailure(err.message),
            presentation: scenario.presentation,
            recipeKey: scenario.recipeKey,
            readySelectors: scenario.readySelectors,
            coverage: getCoverageStatus(scenario, []),
            debugEvents,
            ...failureArtifacts
        };
    } finally {
        await context.close();
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
    const args = parseArgs(process.argv);

    // Create output directory
    fs.mkdirSync(args.output, { recursive: true });

    // Fetch auth client config from the backend (through the Vite proxy)
    console.log("Fetching auth client config...");
    let authConfig;
    try {
        authConfig = await getAuthClientConfig(args.baseUrl);
        console.log(
            `  authDisabled=${authConfig.authDisabled} loginMethod=${authConfig.directConnectLoginMethod}`
        );
    } catch (err) {
        console.error(`Failed to fetch auth client config: ${err.message}`);
        process.exit(1);
    }

    const { authDisabledMode, hasRealCreds, useSentinel, backendRequiresAuth } =
        resolveAuthStrategy(args, authConfig);

    // Enforce the auth-input requirement now that the backend's auth_disabled
    // flag is known. Credentials are only required when we are not in
    // auth-disabled mode (neither the operator opt-in nor the backend report)
    // and no real credentials were supplied.
    if (!authDisabledMode && !hasRealCreds) {
        console.error(
            "Missing required args. Provide Direct Connect email/password credentials (or --auth-disabled for an auth-disabled backend). Run with --help for usage."
        );
        process.exit(1);
    }

    // Guard the unsafe combination: auth-disabled mode was requested (often via
    // localhost auto-detection) but the backend actually enforces auth and no
    // real credentials were provided. Injecting the sentinel would only capture
    // a login/empty screen, so fail fast with an actionable message instead.
    if (backendRequiresAuth) {
        console.error(
            "Auth-disabled mode was requested, but the backend reports auth is enabled (auth_disabled=false). " +
                "Provide Direct Connect password credentials to capture against an authenticated backend."
        );
        process.exit(1);
    }

    if (useSentinel) {
        console.log("[auth] auth-disabled mode: using local sentinel session");
    }

    let storageState;
    let featureFlags;
    if (hasRealCreds) {
        if (authConfig.directConnectLoginMethod !== "password") {
            console.error(
                "Direct Connect browser credential capture requires direct_connect_login_method=password."
            );
            process.exit(1);
        }
        console.log(`Launching Chromium...`);
        const authBrowser = await chromium.launch({ headless: true });
        const authContext = await authBrowser.newContext({
            viewport: { width: 1400, height: 900 },
            baseURL: args.baseUrl
        });
        const authPage = await authContext.newPage();
        try {
            await loginWithBrowserFlow(
                authPage,
                args.baseUrl,
                args.authEmail,
                args.authPassword
            );
            featureFlags = await fetchFeatureFlags(args.baseUrl, authContext);
            storageState = await authContext.storageState();
        } finally {
            await authContext.close();
            await authBrowser.close();
        }
    }

    featureFlags ??= await fetchFeatureFlags(args.baseUrl);
    console.log(`[feature-flags] ${JSON.stringify(featureFlags)}`);

    const scenarios = args.plan
        ? readPlanFile(args.plan)
        : normalizeScreenshotPlan({
              routes: args.routes,
              readySelectors: args.readySelectors
          });

    const config = {
        baseUrl: args.baseUrl,
        output: args.output,
        authConfig,
        featureFlags,
        storageState
    };

    console.log(`Launching Chromium...`);
    const browser = await chromium.launch({ headless: true });

    const results = [];
    for (const scenario of scenarios) {
        const skipReason = shouldSkipScenarioForFeatureFlags(
            scenario,
            config.featureFlags
        );
        if (skipReason) {
            console.log(
                `Skipping ${scenario.name} on ${scenario.route}: ${skipReason}`
            );
            results.push({
                route: scenario.route,
                scenarioName: scenario.name,
                success: true,
                skipped: true,
                skipReason,
                presentation: scenario.presentation,
                recipeKey: scenario.recipeKey,
                readySelectors: scenario.readySelectors,
                coverage: getSkippedCoverageStatus(scenario, skipReason),
                debugEvents: []
            });
            continue;
        }
        console.log(`Capturing ${scenario.route}...`);
        results.push(await captureScenario(browser, scenario, config));
    }

    await browser.close();

    const succeeded = results.filter((result) => result.success).length;
    const failedRoutes = results
        .filter((result) => !result.success)
        .map((result) => result.route);
    const missingCoverage = getMissingCoverage(results);
    const failureTypes = [
        ...new Set(
            results
                .map((result) => result.failureType)
                .filter((failureType) => typeof failureType === "string")
        )
    ];
    if (missingCoverage.length > 0) {
        failureTypes.push("coverage-missing");
    }
    const manifest = {
        generatedAt: new Date().toISOString(),
        outputDir: config.output,
        requestedRoutes: scenarios.map((scenario) => scenario.route),
        succeeded,
        failed: failedRoutes.length,
        coverageFailed: missingCoverage.length,
        missingCoverage,
        failureTypes: [...new Set(failureTypes)],
        results
    };
    writeManifest(config.output, manifest);
    writeScreenshotDebugSummary(config.output, manifest);

    console.log(
        `\nDone: ${succeeded}/${scenarios.length} screenshots captured.`
    );
    if (failedRoutes.length > 0) {
        console.error(`Failed routes: ${failedRoutes.join(", ")}`);
        process.exit(1);
    }
    if (missingCoverage.length > 0) {
        console.error(
            `Missing screenshot coverage: ${missingCoverage
                .map((coverage) => `${coverage.route}:${coverage.id}`)
                .join(", ")}`
        );
        process.exit(1);
    }
}

const isMainModule =
    process.argv[1] &&
    path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMainModule) {
    main();
}
