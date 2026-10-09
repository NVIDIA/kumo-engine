#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * UI baseline audit script.
 * Measures styling-debt metrics and outputs JSON for the baseline ratchet.
 *
 * Usage:
 *   node scripts/ui-baseline-audit.mjs              # print baseline JSON to stdout
 *   node scripts/ui-baseline-audit.mjs --write      # overwrite .ui-baseline.json
 *   node scripts/ui-baseline-audit.mjs --report     # also write .ui-baseline-audit-report.json (enumerated)
 *   node scripts/ui-baseline-audit.mjs --check      # CI ratchet — fail if any numeric metric increased
 *
 * Accuracy caveat: complexity/branch metrics are regex-based approximations,
 * not a real AST analysis. The ternary pattern (/\?[^.:=]/) and other branch
 * regexes will miscount in edge cases (e.g. `?` inside strings/comments,
 * unusual whitespace). Treat these numbers as trend indicators for the
 * ratchet, not ground-truth cyclomatic complexity. For authoritative
 * per-file enforcement, rely on the eslint `complexity` rule.
 */

import { spawnSync } from "child_process";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "fs";
import { dirname, extname, join, relative } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = join(__dirname, "..");
const SRC = join(ROOT, "src");

const WRITE = process.argv.includes("--write");
const REPORT = process.argv.includes("--report");
const CHECK = process.argv.includes("--check");

// ---------------------------------------------------------------------------
// File walker
// ---------------------------------------------------------------------------
function walk(dir, extensions = [".ts", ".tsx"], ignorePaths = []) {
    const results = [];
    let entries;
    try {
        entries = readdirSync(dir, { withFileTypes: true });
    } catch {
        return results;
    }
    for (const entry of entries) {
        const full = join(dir, entry.name);
        if (ignorePaths.some((p) => full.startsWith(p))) continue;
        if (entry.isDirectory()) {
            results.push(...walk(full, extensions, ignorePaths));
        } else if (extensions.includes(extname(entry.name))) {
            results.push(full);
        }
    }
    return results;
}

const IGNORE = [join(SRC, "gen"), join(SRC, "utils", "PQLParsing", "grammar")];

const allTsxTs = walk(SRC, [".ts", ".tsx"], IGNORE);
const allTsx = allTsxTs.filter((f) => f.endsWith(".tsx"));

// Style-debt metrics exclude tests: inline styles / arbitrary classes inside test files
// usually assert on rendered DOM, not real migration debt. `complexity()` also
// excludes tests so its scope mirrors the eslint `complexity` rule (the real gate).
const isTest = (f) => f.includes("__tests__") || f.includes(".test.");
const allTsxNonTest = allTsx.filter((f) => !isTest(f));
// Files in scope for the complexity pass: mirror the eslint complexity rule's
// ignores (**/__tests__/**, **/*.test.{ts,tsx}, src/tests/**; gen/* already
// excluded from allTsxTs). Test branches/LoC are fixture noise, not health signal.
const inComplexityScope = (f) =>
    !isTest(f) && !f.startsWith(join(SRC, "tests"));

function readText(path) {
    try {
        return readFileSync(path, "utf8");
    } catch {
        return "";
    }
}

const rel = (p) => relative(ROOT, p);

// Enumerated details accumulated when running with --report.
const details = {
    toHaveClassCallers: {},
    bidirectionalUiImports: {},
    uiFilesWithHooksOrStores: {},
    indirectClassNameFiles: [],
    templateLiteralClassNameFiles: [],
    presentationCandidates: [],
    filesOver400Loc: [],
    filesOver20Imports: [],
    layerViolations: {}
};

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/** 1. Count import statements that pull from src/css/ */
function appCssImports() {
    let n = 0;
    for (const f of allTsxTs) {
        const t = readText(f);
        const hits = t.match(/import\s+['"](?:\.\.?\/)*css\/[^'"]+['"]/g) ?? [];
        n += hits.length;
    }
    return n;
}

/** 2. Total LOC across all src/css/*.css files */
function appCssLoc() {
    const cssDir = join(SRC, "css");
    if (!existsSync(cssDir)) return 0;
    let total = 0;
    for (const name of readdirSync(cssDir)) {
        if (!name.endsWith(".css")) continue;
        total += readText(join(cssDir, name)).split("\n").length;
    }
    return total;
}

/** 3. LOC of src/css/kumo.css */
function kumoCssLoc() {
    const t = readText(join(SRC, "css", "kumo.css"));
    return t ? t.split("\n").length : 0;
}

/** 4. Total style={{ occurrences across .tsx files (excludes tests). */
function inlineStyleOccurrences() {
    let n = 0;
    for (const f of allTsxNonTest) {
        n += (readText(f).match(/style=\{\{/g) ?? []).length;
    }
    return n;
}

/** 5. Number of .tsx files that contain at least one style={{ (excludes tests). */
function filesWithInlineStyles() {
    return allTsxNonTest.filter((f) => /style=\{\{/.test(readText(f))).length;
}

/**
 * 6. Arbitrary-value Tailwind class occurrences, bucketed (excludes tests).
 * TODO: scope the bracket scan to className strings (className="…", cn()/clsx()/cx() args,
 * template literals). Today the regex scans every `[…]` in the file — array destructuring,
 * indexing, regex character classes etc. The bucket filters keep the count correct, but
 * it's doing far more work than needed.
 */
function arbitraryTailwindClasses() {
    const ARB = /\[([^\]]+)\]/g;
    let dynamic = 0,
        viewport = 0,
        hardcodedPixels = 0;
    for (const f of allTsxNonTest) {
        const t = readText(f);
        let m;
        ARB.lastIndex = 0;
        while ((m = ARB.exec(t)) !== null) {
            const inner = m[1].trim();
            if (inner.includes("var(--")) dynamic++;
            else if (/\d(vh|vw|svh|dvh|svw|dvw|vmin|vmax)/.test(inner))
                viewport++;
            else if (
                /^\d+(\.\d+)?(px|rem|em)$/.test(inner) ||
                /^#[0-9a-fA-F]{3,8}$/.test(inner)
            )
                hardcodedPixels++;
        }
    }
    return { dynamic, viewport, hardcodedPixels };
}

/** 7. Files that use generic neutral Tailwind palette classes (excludes tests). */
function neutralPaletteFiles() {
    const NEUTRAL =
        /\b(?:text|bg|border|ring|outline|fill|stroke)-(slate|gray|zinc|neutral)-\d+/;
    return allTsxNonTest.filter((f) => NEUTRAL.test(readText(f))).length;
}

/** 8. Catalog adoption — share of components/ and pages/ that import the catalog */
function catalogAdoption() {
    const jsxFiles = allTsx.filter((f) => {
        const r = relative(SRC, f);
        return r.startsWith("components") || r.startsWith("pages");
    });
    const adopters = jsxFiles.filter((f) =>
        readText(f).includes("@kumo-ai/kumo-ui-catalog")
    );
    const ratio =
        jsxFiles.length > 0
            ? Math.round((adopters.length / jsxFiles.length) * 100) / 100
            : 0;
    return { ratio, adopters: adopters.length, total: jsxFiles.length };
}

/** 9. JSX-rendering files with no catalog import, excluding documented adapters. */
function filesRenderingUiWithoutCatalogImport() {
    const ADAPTERS = [
        join("components", "@hoc"),
        // App-shell adapters allowed to render without a catalog import. Route
        // guards, Logout, and Intro were lifted out of @ui/ into the feature
        // layer (layer-violation cleanup) but remain app adapters, so their new
        // paths stay excluded here.
        join("components", "Route"),
        join("components", "@ui", "Layout", "MainLayout"),
        join("components", "Logout"),
        join("components", "Intro"),
        join("components", "@ui", "RouteErrorBoundary")
    ];
    const candidates = allTsx.filter((f) => {
        const r = relative(SRC, f);
        return (
            (r.startsWith("components") || r.startsWith("pages")) &&
            !ADAPTERS.some((a) => r.startsWith(a))
        );
    });
    return candidates.filter((f) => {
        const t = readText(f);
        return !t.includes("@kumo-ai/kumo-ui-catalog") && /<[A-Z]/.test(t);
    }).length;
}

/** 10. Test files that call toHaveClass(); enumerate asserted classes if --report. */
function classAssertingTestFiles() {
    const tests = allTsxTs.filter(
        (f) => f.includes("__tests__") || f.includes(".test.")
    );
    let count = 0;
    for (const f of tests) {
        const t = readText(f);
        const calls = t.match(/toHaveClass\(\s*([^)]+)\)/g) ?? [];
        if (calls.length === 0) continue;
        count++;
        if (REPORT) {
            const classes = [];
            for (const c of calls) {
                // Pull every quoted string out of the args
                const inner = c.match(/['"]([^'"]+)['"]/g) ?? [];
                for (const q of inner) classes.push(q.slice(1, -1));
            }
            details.toHaveClassCallers[rel(f)] = [...new Set(classes)].sort();
        }
    }
    return count;
}

/** 11. Files using className as an OBJECT property key (indirect, not a JSX attribute). */
function indirectClassNameFiles() {
    // Require the value to be a JS expression (string/template literal, class-name helper call,
    // object, or array). This avoids matching TS type contexts like `className: string` or
    // `className: string | undefined` in interface/type declarations.
    const INDIRECT =
        /(?<![<\w])className\s*:\s*(?:["'`]|(?:cn|clsx|cx|classNames)\s*\(|\{|\[)/;
    let count = 0;
    for (const f of allTsxTs) {
        if (INDIRECT.test(readText(f))) {
            count++;
            if (REPORT) details.indirectClassNameFiles.push(rel(f));
        }
    }
    return count;
}

/** 12. @ui files importing from feature folders (bidirectional coupling).
 *  Tests are excluded: shared test fixtures/mocks live in their feature's
 *  __tests__ folder and importing them is a test-organization concern, not
 *  runtime presentation coupling. Layer enforcement (boundaries) ignores tests
 *  too, so the metric tracks shipped @ui source only. */
function bidirectionalUiToFeatureImports() {
    const uiFiles = allTsxTs.filter(
        (f) =>
            relative(SRC, f).startsWith(join("components", "@ui")) && !isTest(f)
    );
    let count = 0;
    const FEATURE = /from\s+['"](components\/(?!@ui|@hoc)[^'"]+)['"]/g;
    for (const f of uiFiles) {
        const t = readText(f);
        FEATURE.lastIndex = 0;
        const matches = [];
        let m;
        while ((m = FEATURE.exec(t)) !== null) matches.push(m[1]);
        if (matches.length > 0) {
            count++;
            if (REPORT) details.bidirectionalUiImports[rel(f)] = matches;
        }
    }
    return count;
}

/** 13. Files that import cn() from the catalog and call it. */
function cnAdopters() {
    return allTsxTs.filter((f) => {
        const t = readText(f);
        return t.includes("@kumo-ai/kumo-ui-catalog") && /\bcn\s*\(/.test(t);
    }).length;
}

/** 13b. Files concatenating className via template literals (cn() candidates) — excludes tests. */
function templateLiteralClassNameFiles() {
    // className={`…${cond ? "a" : "b"}…`}  OR  className={"a" + (x ? " b" : "")}
    const TMPL = /className=\{\s*`/;
    const CONCAT = /className=\{\s*["'][^"']*["']\s*\+/;
    let count = 0;
    for (const f of allTsxNonTest) {
        const t = readText(f);
        if (TMPL.test(t) || CONCAT.test(t)) {
            count++;
            if (REPORT) details.templateLiteralClassNameFiles.push(rel(f));
        }
    }
    return count;
}

/** 14. @ui files that runtime-import hooks, stores, or generated query hooks. */
function uiFilesImportingHooksOrStores() {
    const uiFiles = allTsxTs.filter((f) =>
        relative(SRC, f).startsWith(join("components", "@ui"))
    );
    const RUNTIME =
        /^import\s+(?!type\s+).*?from\s+['"](?:hooks\/|stores\/|gen\/v[01]\/|@tanstack\/react-query)[^'"]*['"]/m;
    let count = 0;
    for (const f of uiFiles) {
        const t = readText(f);
        if (RUNTIME.test(t)) {
            count++;
            if (REPORT) {
                const hits =
                    t.match(
                        /^import\s+(?!type\s+).+from\s+['"](?:hooks\/|stores\/|gen\/v[01]\/|@tanstack\/react-query)[^'"]*['"]/gm
                    ) ?? [];
                details.uiFilesWithHooksOrStores[rel(f)] = hits;
            }
        }
    }
    return count;
}

/** 14b. @ui files that are presentation-only (no runtime hooks/stores) — promotion candidates. */
function presentationCandidates() {
    const uiFiles = allTsxTs.filter((f) =>
        relative(SRC, f).startsWith(join("components", "@ui"))
    );
    const RUNTIME =
        /^import\s+(?!type\s+).*?from\s+['"](?:hooks\/|stores\/|gen\/v[01]\/|@tanstack\/react-query)[^'"]*['"]/m;
    let count = 0;
    for (const f of uiFiles) {
        const t = readText(f);
        if (!RUNTIME.test(t)) {
            count++;
            if (REPORT) details.presentationCandidates.push(rel(f));
        }
    }
    return count;
}

/** 15. Circular dependency count via madge (null when madge unavailable). */
function circularDeps() {
    // Exclude generated code so the count tracks only hand-written cycles. Matches the
    // walker's IGNORE list (src/gen, src/utils/PQLParsing/grammar) — madge's --exclude
    // takes a regex applied to relative paths from cwd.
    const result = spawnSync(
        "npx",
        [
            "madge",
            "--circular",
            "--extensions",
            "ts,tsx",
            "--exclude",
            "^(gen/|utils/PQLParsing/grammar/)",
            "src/"
        ],
        { cwd: ROOT, encoding: "utf8", timeout: 90_000 }
    );
    if (result.error) return null;
    // madge writes the "Found N circular" / "No circular dependency found" summary to stderr.
    const combined = (result.stdout ?? "") + (result.stderr ?? "");
    const header = combined.match(/Found (\d+) circular/);
    if (header) return parseInt(header[1], 10);
    if (/No circular dependency found/.test(combined)) return 0;
    return null;
}

/**
 * 15b. Layer-rule violations — total count of runtime imports that cross a
 * forbidden layer boundary. Mirrors the matrix enforced by eslint-plugin-boundaries
 * and `@typescript-eslint/no-restricted-imports` (see eslint.config.ts). Type-only
 * imports (`import type …`) are excluded — they're erased at compile time and the
 * ESLint config explicitly allows them across some boundaries (e.g. gen/v* for @ui).
 *
 * Each forbidden import counts once (not files), so the number ratchets down as
 * individual offenders are fixed.
 *
 * Test files (**\/__tests__/**, *.test.{ts,tsx}) are exempt — a test importing a
 * mock/fixture from another layer is a test-organization concern, not a runtime
 * layer dependency. Only production-code imports are counted.
 */
function layerViolations() {
    const RULES = [
        {
            name: "ui-shared",
            from: join("components", "@ui") + "/",
            forbidden: [
                /^css\//,
                /^@tanstack\/react-query/,
                /^gen\/v[01]/,
                /^hooks\//,
                /^stores\//,
                /^pages\//,
                /^components\/(?!@ui|@hoc)/
            ]
        },
        {
            name: "hoc",
            from: join("components", "@hoc") + "/",
            forbidden: [
                /^pages\//,
                /^components\/(?!@ui|@hoc)/,
                /^hooks\//,
                /^stores\//
            ]
        },
        {
            name: "hooks",
            from: "hooks/",
            forbidden: [/^components\//, /^pages\//]
        },
        {
            name: "stores",
            from: "stores/",
            forbidden: [/^components\//, /^pages\//, /^hooks\//]
        },
        {
            name: "services",
            from: "services/",
            forbidden: [/^components\//, /^pages\//, /^hooks\//, /^stores\//]
        },
        {
            name: "utils",
            from: "utils/",
            forbidden: [
                /^components\//,
                /^pages\//,
                /^hooks\//,
                /^stores\//,
                /^services\//
            ]
        }
    ];

    // Matches `import [...] from "X"` and side-effect `import "X"`.
    // `(?!type\s)` skips `import type …` statements.
    const IMPORT_RE =
        /^\s*import\s+(?!type\s)(?:[\s\S]*?from\s+)?['"]([^'"]+)['"]/gm;

    let total = 0;
    for (const f of allTsxTs) {
        // Test files are exempt: a test importing a mock/fixture that happens to
        // live under pages/ or components/ is a test-organization concern, not a
        // runtime layer dependency. Mirrors the boundaries-plugin ignore list
        // (**/__tests__/**, **/*.test.{ts,tsx}, src/tests/**) re-enabled in Phase 4.
        if (isTest(f)) continue;
        const r = relative(SRC, f);
        const rule = RULES.find((rl) => r.startsWith(rl.from));
        if (!rule) continue;
        const t = readText(f);
        IMPORT_RE.lastIndex = 0;
        const hits = [];
        let m;
        while ((m = IMPORT_RE.exec(t)) !== null) {
            const path = m[1];
            if (rule.forbidden.some((re) => re.test(path)))
                hits.push({ layer: rule.name, import: path });
        }
        if (hits.length > 0) {
            total += hits.length;
            if (REPORT) details.layerViolations[rel(f)] = hits;
        }
    }
    return total;
}

/** 16. Storybook story files */
function storybookStories() {
    return walk(SRC, [".tsx", ".ts"], IGNORE).filter((f) =>
        f.includes(".stories.")
    ).length;
}

/**
 * 17. Complexity / branching / coupling / cognitive-load buckets.
 * Approximate cyclomatic complexity via regex (counted per file, then bucketed).
 *
 * Scope mirrors the eslint `complexity` rule (production source only — tests
 * excluded via inComplexityScope). The eslint rule is the authoritative gate
 * (cap 20, existing offenders grandfathered in eslint-suppressions.json); these
 * regex metrics are a localized dashboard. We deliberately do NOT emit a global
 * branch-point sum: a codebase-wide total measures size, not quality (it can't
 * tell 1000 simple functions from 10 monsters), so it's useless for assessment
 * and hostile to growth as a ratchet. Quality lives in the per-file buckets
 * (highComplexityFiles, filesOver400Loc, …) and the eslint per-function gate.
 */
function complexity() {
    const BRANCH = [
        /\bif\s*\(/g,
        /\?[^.:=]/g, // ternary (exclude ?. ?: ?=)
        /&&|\|\|/g,
        /\bcase\s+/g,
        /\bcatch\s*\(/g,
        /\bfor\s*\(/g,
        /\bwhile\s*\(/g,
        /\?\?(?!=)/g // nullish coalescing
    ];
    let highComplexityFiles = 0; // ≥ 40
    let mediumComplexityFiles = 0; // 20–39
    let filesOver400Loc = 0;
    let filesOver20Imports = 0;
    let totalImports = 0;
    let totalLoc = 0;

    const files = allTsxTs.filter(inComplexityScope);
    for (const f of files) {
        const t = readText(f);
        const loc = t.split("\n").length;
        totalLoc += loc;
        if (loc > 400) {
            filesOver400Loc++;
            if (REPORT) details.filesOver400Loc.push({ file: rel(f), loc });
        }

        const importCount = (t.match(/^\s*import\s+/gm) ?? []).length;
        totalImports += importCount;
        if (importCount > 20) {
            filesOver20Imports++;
            if (REPORT)
                details.filesOver20Imports.push({
                    file: rel(f),
                    imports: importCount
                });
        }

        let branches = 0;
        for (const p of BRANCH) branches += (t.match(p) ?? []).length;
        if (branches >= 40) highComplexityFiles++;
        else if (branches >= 20) mediumComplexityFiles++;
    }

    return {
        highComplexityFiles,
        mediumComplexityFiles,
        filesOver400Loc,
        filesOver20Imports,
        avgImportsPerFile:
            files.length > 0
                ? Math.round((totalImports / files.length) * 10) / 10
                : 0,
        avgLocPerFile:
            files.length > 0 ? Math.round(totalLoc / files.length) : 0
    };
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------
process.stderr.write("audit-ui-baseline: collecting metrics…\n");

const catalogInfo = catalogAdoption();

process.stderr.write("audit-ui-baseline: running madge (circular deps)…\n");
const circular = circularDeps();

const baseline = {
    appCssImports: appCssImports(),
    appCssLoc: appCssLoc(),
    kumoCssLoc: kumoCssLoc(),
    inlineStyleOccurrences: inlineStyleOccurrences(),
    filesWithInlineStyles: filesWithInlineStyles(),
    arbitraryTailwindClasses: arbitraryTailwindClasses(),
    neutralPaletteFiles: neutralPaletteFiles(),
    catalogAdoptionRatio: catalogInfo.ratio,
    filesRenderingUiWithoutCatalogImport:
        filesRenderingUiWithoutCatalogImport(),
    classAssertingTestFiles: classAssertingTestFiles(),
    indirectClassNameFiles: indirectClassNameFiles(),
    bidirectionalUiToFeatureImports: bidirectionalUiToFeatureImports(),
    cnAdopters: cnAdopters(),
    templateLiteralClassNameFiles: templateLiteralClassNameFiles(),
    uiFilesImportingHooksOrStores: uiFilesImportingHooksOrStores(),
    presentationCandidates: presentationCandidates(),
    complexity: complexity(),
    circularDeps: circular,
    layerViolations: layerViolations(),
    storybookStories: storybookStories(),
    axeViolationsCriticalSerious: null
};

if (WRITE) {
    const out = join(ROOT, ".ui-baseline.json");
    // Preserve any `_`-prefixed doc fields (e.g. `_comments`) already in the
    // committed baseline — they're documentation, not metrics, so a fresh
    // --write shouldn't silently strip them.
    let preserved = {};
    if (existsSync(out)) {
        try {
            const prev = JSON.parse(readFileSync(out, "utf8"));
            for (const [k, v] of Object.entries(prev))
                if (k.startsWith("_")) preserved[k] = v;
        } catch {
            // ignore — fall through and write a fresh baseline
        }
    }
    writeFileSync(
        out,
        JSON.stringify({ ...baseline, ...preserved }, null, 2) + "\n"
    );
    process.stderr.write(`audit-ui-baseline: wrote ${out}\n`);
}

if (REPORT) {
    // Sort the enumerated lists for stable diffs.
    for (const k of Object.keys(details)) {
        if (Array.isArray(details[k]))
            details[k].sort((a, b) => {
                const aa = typeof a === "string" ? a : a.file;
                const bb = typeof b === "string" ? b : b.file;
                return aa.localeCompare(bb);
            });
    }
    const out = join(ROOT, ".ui-baseline-audit-report.json");
    writeFileSync(out, JSON.stringify(details, null, 2) + "\n");
    process.stderr.write(`audit-ui-baseline: wrote ${out}\n`);
}

if (CHECK) {
    const baselinePath = join(ROOT, ".ui-baseline.json");
    if (!existsSync(baselinePath)) {
        process.stderr.write(
            "No .ui-baseline.json found.\n" +
                "Run: node scripts/ui-baseline-audit.mjs --write\n"
        );
        process.exit(1);
    }
    const committed = JSON.parse(readFileSync(baselinePath, "utf8"));
    const regressions = [];
    const improvements = [];
    const staleKeys = [];

    // Adoption metrics where larger numbers mean more progress. Every other
    // metric in the baseline is a debt counter where smaller is better.
    const LARGER_IS_BETTER = new Set([
        "catalogAdoptionRatio",
        "cnAdopters",
        "storybookStories"
    ]);

    // Informational metrics: reported in the baseline but NOT gated by --check.
    // These regex/whole-file debt counters all penalize the per-function
    // decomposition that the eslint `complexity` rule (the real gate) rewards:
    //   - avgLocPerFile: extracting helpers *raises* LoC while lowering complexity.
    //   - high/mediumComplexityFiles: count a file's *whole* regex branch-sum,
    //     which decomposition redistributes into more functions (same or slightly
    //     higher total) rather than reducing — so a file can tip medium->high
    //     purely from added helper signatures, despite every function now <= 20.
    // Per-function enforcement lives in eslint; these stay as trend dashboards
    // (cf. the removed `totalBranchPoints` global sum).
    const INFORMATIONAL = new Set([
        "complexity.avgLocPerFile",
        "complexity.highComplexityFiles",
        "complexity.mediumComplexityFiles"
    ]);

    // Warn on baseline keys that no longer exist in the current run (likely
    // renamed/removed) or that transitioned from `null` → real number (the
    // committed null silently disabled enforcement; --write is now needed).
    const collectStale = (key, bVal, cVal) => {
        if (cVal === undefined) {
            staleKeys.push({ key, reason: "missing in current run" });
            return;
        }
        if (bVal === null && typeof cVal === "number" && !Number.isNaN(cVal)) {
            staleKeys.push({
                key,
                reason: `committed null, current ${cVal} — re-run --write to enforce`
            });
            return;
        }
        if (
            bVal &&
            typeof bVal === "object" &&
            cVal &&
            typeof cVal === "object"
        ) {
            for (const [k, v] of Object.entries(bVal))
                collectStale(`${key}.${k}`, v, cVal[k]);
        }
    };
    for (const [k, v] of Object.entries(committed)) {
        if (k.startsWith("_")) continue; // documentation fields (e.g. _comments)
        collectStale(k, v, baseline[k]);
    }

    // NOTE: missing keys are silently skipped by cmp() (the `cVal === null`
    // early return covers both intentional `null` placeholders and the
    // `?? null` fallback when a key is absent from the current run). The
    // collectStale() pass above surfaces those cases as a non-failing notice.
    const cmp = (key, bVal, cVal) => {
        if (bVal === null || cVal === null || cVal === undefined) return;
        if (typeof bVal === "object" && typeof cVal === "object") {
            for (const [k, v] of Object.entries(bVal))
                cmp(`${key}.${k}`, v, cVal[k] ?? null);
            return;
        }
        if (INFORMATIONAL.has(key)) return;
        if (typeof bVal !== "number" || typeof cVal !== "number") return;
        if (cVal === bVal) return;
        const grew = cVal > bVal;
        const isImprovement = LARGER_IS_BETTER.has(key) ? grew : !grew;
        const d = Math.abs(cVal - bVal);
        (isImprovement ? improvements : regressions).push({
            key,
            b: bVal,
            c: cVal,
            d
        });
    };

    for (const [k, v] of Object.entries(committed))
        cmp(k, v, baseline[k] ?? null);

    if (staleKeys.length > 0) {
        process.stderr.write(
            "\nℹ️  Stale baseline keys (run --write to refresh):\n"
        );
        for (const s of staleKeys)
            process.stderr.write(`   ${s.key}: ${s.reason}\n`);
    }
    // Same stream for both lists so headers don't interleave in a TTY. The
    // sign reflects the direction of the actual change, not which list the
    // row landed in — for LARGER_IS_BETTER metrics, an improvement is `+`.
    const fmt = (r) => {
        const sign = r.c > r.b ? "+" : "−";
        return `   ${r.key}: ${r.b} → ${r.c} (${sign}${r.d})\n`;
    };
    if (improvements.length > 0) {
        process.stderr.write("\n✅ Improvements detected:\n");
        for (const i of improvements) process.stderr.write(fmt(i));
    }
    if (regressions.length > 0) {
        process.stderr.write("\n❌ Migration baseline regressions:\n");
        for (const r of regressions) process.stderr.write(fmt(r));
        process.stderr.write(
            "\nFix the regression, or migrate the affected code in this PR.\n" +
                "If the increase is genuinely intentional, update .ui-baseline.json in the same commit.\n"
        );
        process.exit(1);
    }
    if (regressions.length === 0 && improvements.length > 0) {
        process.stderr.write(
            "\n⚠️  Update .ui-baseline.json to lock in improvements:\n" +
                "   npm run audit:write\n"
        );
    } else if (regressions.length === 0 && improvements.length === 0) {
        process.stdout.write(
            "✅ Migration baseline check passed. No regressions.\n"
        );
    }
} else if (!WRITE) {
    process.stdout.write(JSON.stringify(baseline, null, 2) + "\n");
}
