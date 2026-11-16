#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Dead-test detector.
 *
 * Finds test files that exercise a component which no longer lives in this repo
 * (typically because it was promoted to `@kumo-ai/kumo-ui-catalog`). knip cannot
 * surface these: a `*.test.tsx` is always an entry point, and importing a real
 * external package is not an "unused" signal — so the orphaned test slips through.
 *
 * Three complementary heuristics, all regex-based (no AST):
 *
 *   1. Orphan test directory — a `__tests__/` whose owning component directory
 *      contains no local source file (only the tests). e.g.
 *      `components/@ui/ValidationCard/` holds only `__tests__/ValidationCard.test.tsx`
 *      because ValidationCard moved to the catalog.
 *
 *   2. Catalog-only test — a test file that imports from
 *      `@kumo-ai/kumo-ui-catalog` and has NO local import (neither a relative
 *      path nor a `baseUrl: "src"` absolute path that resolves under src/, and
 *      counting bare side-effect imports like `import "tests/mockAxios"`).
 *      A test that legitimately covers catalog↔app integration can opt out of
 *      this heuristic with a `dead-tests-ignore` comment anywhere in the file.
 *
 *   3. Unreachable lazy-route component — a component wired into `Routes.tsx`
 *      via `createLazyModule(() => import(...))` whose route path
 *      (`navRoutes.X`) is never referenced anywhere except the route registration
 *      and the `navRoutes` enum definition. Nothing navigates to it, so the route
 *      is dead and the component is rendered only by its own test. knip cannot see
 *      this: the lazy `import()` counts as "used", so the file never shows up in
 *      `knip --production`. Both the component file and its test are dead, plus the
 *      route registration + enum member should be removed.
 *
 * A file flagged by any heuristic is a strong delete candidate. Review before
 * removing — a test may legitimately cover catalog↔app integration, and a route
 * could (rarely) be reached by a non-`navRoutes` path.
 *
 * Usage:
 *   node scripts/find-dead-tests.mjs           # report; exit 1 if any found
 *   node scripts/find-dead-tests.mjs --warn     # report but always exit 0 (non-gating)
 */

import { readdirSync, readFileSync, statSync } from "fs";
import { dirname, join, relative } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const UI_ROOT = dirname(dirname(__filename));
const SRC = join(UI_ROOT, "src");
const CATALOG = "@kumo-ai/kumo-ui-catalog";
const ROUTES_FILE = join(SRC, "routes/Routes.tsx");
const ROUTES_ENUM_FILE = join(SRC, "utils/Constants/routes.ts");

const SKIP_DIRS = new Set(["node_modules", "gen", "grammar"]);
const isTest = (name) => /\.test\.(ts|tsx)$/.test(name);
const isSource = (name) => /\.(ts|tsx|js)$/.test(name) && !isTest(name);

/** Recursively collect all `*.test.{ts,tsx}` files under a directory. */
function collectTests(dir, out = []) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.isDirectory()) {
            if (SKIP_DIRS.has(entry.name)) continue;
            collectTests(join(dir, entry.name), out);
        } else if (isTest(entry.name)) {
            out.push(join(dir, entry.name));
        }
    }
    return out;
}

/**
 * True if `dir` (recursively) contains at least one non-test source file,
 * ignoring anything under a `__tests__` segment. Short-circuits on first hit.
 */
function hasLocalSource(dir) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.isDirectory()) {
            if (entry.name === "__tests__" || SKIP_DIRS.has(entry.name))
                continue;
            if (hasLocalSource(join(dir, entry.name))) return true;
        } else if (isSource(entry.name)) {
            return true;
        }
    }
    return false;
}

/** The component directory owning a test file = path before the LAST `/__tests__/`. */
function componentDirOf(testPath) {
    const parts = testPath.split(/[/\\]/);
    const idx = parts.lastIndexOf("__tests__");
    if (idx <= 0) return null;
    return parts.slice(0, idx).join("/");
}

const SRC_EXTS = ["ts", "tsx", "js"];

/** A specifier is "local" if it is relative OR resolves to a file under `src/`. */
function isLocalSpecifier(spec) {
    if (spec.startsWith("./") || spec.startsWith("../")) return true;
    // Absolute-from-src imports (baseUrl: "src"), e.g. "hooks/utils/debounce", "App".
    const candidates = [];
    for (const ext of SRC_EXTS) {
        candidates.push(join(SRC, `${spec}.${ext}`));
        candidates.push(join(SRC, spec, `index.${ext}`));
    }
    return candidates.some((p) => {
        try {
            return statSync(p).isFile();
        } catch {
            return false;
        }
    });
}

function analyzeImports(src) {
    let importsCatalog = false;
    let hasLocalImport = false;
    // Match both `... from "x"` and bare side-effect `import "x"` so a setup
    // import like `import "tests/mockAxios"` counts as a local dependency.
    const re = /(?:from|import)\s+["']([^"']+)["']/g;
    let m;
    while ((m = re.exec(src)) !== null) {
        const spec = m[1];
        if (spec === CATALOG) importsCatalog = true;
        else if (isLocalSpecifier(spec)) hasLocalImport = true;
    }
    return { importsCatalog, hasLocalImport };
}

/** Collect every `.ts`/`.tsx` file under `src/` (used to scan for route refs). */
function collectSourceFiles(dir, out = []) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.isDirectory()) {
            if (SKIP_DIRS.has(entry.name)) continue;
            collectSourceFiles(join(dir, entry.name), out);
        } else if (/\.(ts|tsx)$/.test(entry.name)) {
            out.push(join(dir, entry.name));
        }
    }
    return out;
}

/** Resolve a `baseUrl: "src"` import path to a real file under src/, or null. */
function resolveSrcModule(importPath) {
    for (const ext of SRC_EXTS) {
        for (const p of [
            join(SRC, `${importPath}.${ext}`),
            join(SRC, importPath, `index.${ext}`)
        ]) {
            try {
                if (statSync(p).isFile()) return p;
            } catch {
                /* keep looking */
            }
        }
    }
    return null;
}

/** Find the co-located test for a component file, or null. */
function testFileFor(componentFile) {
    const dir = dirname(componentFile);
    const base = componentFile
        .split(/[/\\]/)
        .pop()
        .replace(/\.(ts|tsx)$/, "");
    for (const cand of [
        join(dir, "__tests__", `${base}.test.tsx`),
        join(dir, "__tests__", `${base}.test.ts`),
        join(dir, `${base}.test.tsx`),
        join(dir, `${base}.test.ts`)
    ]) {
        try {
            if (statSync(cand).isFile()) return cand;
        } catch {
            /* not here */
        }
    }
    return null;
}

/**
 * Heuristic 3: lazy-route components whose `navRoutes.X` path is never referenced
 * outside the route registration and the enum definition — i.e. nothing navigates
 * to them, so the route (and the component it lazy-loads) is dead.
 */
function findDeadLazyRoutes() {
    let routesSrc;
    try {
        routesSrc = readFileSync(ROUTES_FILE, "utf8");
    } catch {
        return []; // Routes.tsx not found — skip this heuristic
    }

    // lazyName -> imported module path
    const lazyMap = new Map();
    const lazyRe =
        /export\s+const\s+(\w+)\s*=\s*createLazyModule\(\s*\(\)\s*=>\s*import\(\s*["']([^"']+)["']/g;
    let m;
    while ((m = lazyRe.exec(routesSrc)) !== null) lazyMap.set(m[1], m[2]);

    // pair `path: navRoutes.X ... lazy: Name.lazy` within a single route object
    const routes = [];
    const pairRe = /path:\s*navRoutes\.(\w+)[^{}]*?lazy:\s*(\w+)\.lazy/gs;
    while ((m = pairRe.exec(routesSrc)) !== null) {
        routes.push({ member: m[1], lazyName: m[2] });
    }

    const sources = collectSourceFiles(SRC);
    const dead = [];
    for (const { member, lazyName } of routes) {
        const importPath = lazyMap.get(lazyName);
        if (!importPath) continue;

        // Reachable if `navRoutes.<member>` appears anywhere other than the route
        // registration and the enum definition (tests count — conservative).
        const refRe = new RegExp(`navRoutes\\.${member}\\b`);
        const reachable = sources.some(
            (f) =>
                f !== ROUTES_FILE &&
                f !== ROUTES_ENUM_FILE &&
                refRe.test(readFileSync(f, "utf8"))
        );
        if (reachable) continue;

        const componentFile = resolveSrcModule(importPath);
        if (!componentFile) continue; // already gone
        const test = testFileFor(componentFile);
        dead.push({
            member,
            component: relative(UI_ROOT, componentFile),
            test: test ? relative(UI_ROOT, test) : null
        });
    }
    return dead;
}

const tests = collectTests(SRC);
const orphans = [];
const catalogOnly = [];

for (const testPath of tests) {
    const compDir = componentDirOf(testPath);
    const rel = relative(UI_ROOT, testPath);
    const src = readFileSync(testPath, "utf8");

    // Explicit opt-out: a test that legitimately covers catalog↔app integration
    // (importing only the catalog + test libs) can declare itself intentional with
    // a `dead-tests-ignore` comment, so this gate never hard-fails CI on it.
    if (/dead-tests-ignore/.test(src)) continue;

    // Heuristic 1: orphan test directory (skip the broad `src` / `src/tests` roots).
    if (compDir && compDir !== SRC && !compDir.endsWith("/tests")) {
        let exists;
        try {
            exists = statSync(compDir).isDirectory();
        } catch {
            exists = false;
        }
        if (exists && !hasLocalSource(compDir)) {
            orphans.push(rel);
            continue; // already a delete candidate; don't double-report
        }
    }

    // Heuristic 2: subject imported from the catalog, nothing local to test.
    const { importsCatalog, hasLocalImport } = analyzeImports(src);
    if (importsCatalog && !hasLocalImport) {
        catalogOnly.push(rel);
    }
}

const deadRoutes = findDeadLazyRoutes();

const total = orphans.length + catalogOnly.length + deadRoutes.length;

if (orphans.length) {
    console.log(
        "\nDead test files — orphan __tests__ directory (no local component source):"
    );
    for (const f of orphans.sort()) console.log(`  ${f}`);
}
if (catalogOnly.length) {
    console.log(
        `\nTest files whose subject is imported only from ${CATALOG} (no local import):`
    );
    for (const f of catalogOnly.sort()) console.log(`  ${f}`);
}
if (deadRoutes.length) {
    console.log(
        "\nDead lazy-route components — route never navigated to (rendered only by their own test):"
    );
    for (const r of deadRoutes.sort((a, b) =>
        a.component.localeCompare(b.component)
    )) {
        console.log(`  ${r.component}`);
        if (r.test) console.log(`    test:  ${r.test}`);
        console.log(
            `    route: remove navRoutes.${r.member} registration in src/routes/Routes.tsx + enum member in src/utils/Constants/routes.ts`
        );
    }
}

if (total === 0) {
    console.log("No dead test files found.");
    process.exit(0);
}

console.log(
    `\n${total} dead item(s) found. Review and delete if no longer used by local code.`
);
process.exit(process.argv.includes("--warn") ? 0 : 1);
