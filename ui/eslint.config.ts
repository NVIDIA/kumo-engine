// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fixupPluginRules } from "@eslint/compat";
import eslint from "@eslint/js";
import pluginQuery from "@tanstack/eslint-plugin-query";
import tsparser from "@typescript-eslint/parser";
import pluginBoundaries from "eslint-plugin-boundaries";
import reactHooks from "eslint-plugin-react-hooks";
import simpleImportSort from "eslint-plugin-simple-import-sort";
import globals from "globals";
import tseslint from "typescript-eslint";

const ignores = [
    "**/node_modules/**",
    "**/testing/**",
    "**/build/**",
    "**/dist/**",
    "**/env.js",
    "**/webpack.config.js",
    "**/src/utils/trustedSecurityPolicies.js",
    "**/public/env-config.js",
    "**/src/utils/PQLParsing/grammar/**",
    "**/src/types/training_job.ts",
    "**/public/"
];

export default [
    eslint.configs.recommended,
    ...tseslint.configs.recommended,
    ...pluginQuery.configs["flat/recommended"],
    {
        ignores: ignores
    },

    // Node scripts (build / audit / tooling) — declare node globals so
    // `process`, `__dirname`, etc. are not flagged as `no-undef`.
    {
        files: ["scripts/**/*.{js,mjs,cjs,ts}", "**/*.config.{js,mjs,cjs,ts}"],
        languageOptions: {
            globals: { ...globals.node }
        }
    },

    {
        files: ["**/*.ts", "**/*.tsx", "**/*.js"],
        plugins: {
            "react-hooks": fixupPluginRules(reactHooks),
            "simple-import-sort": simpleImportSort
        },
        languageOptions: {
            parser: tsparser,
            parserOptions: {
                ecmaVersion: "latest",
                ecmaFeatures: {
                    jsx: true
                }
            }
        },
        rules: {
            ...reactHooks.configs.recommended.rules,
            "simple-import-sort/imports": [
                "error",
                {
                    "groups": [
                        // react imports
                        ["^react$", "^react-\\w+"], // imports starting with @ or character
                        ["^@?\\w"], // internal imports with absolute path or relative paths
                        ["^charts/", "^(\\.\\./)*charts/"],
                        ["^components/", "^(\\.\\./)*components/"],
                        ["^context/", "^(\\.\\./)*context/"],
                        ["^converters/", "^(\\.\\./)*converters/"],
                        ["^hooks/", "^(\\.\\./)*hooks/"],
                        ["^icons/", "^(\\.\\./)*icons/"],
                        ["^images/", "^(\\.\\./)*images/"],
                        ["^layout/", "^(\\.\\./)*layout/"],
                        ["^models/", "^(\\.\\./)*models/"],
                        ["^pages/", "^(\\.\\./)*pages/"],
                        ["^routes/", "^(\\.\\./)*routes/"],
                        ["^services/", "^(\\.\\./)*services/"],
                        ["^types/", "^(\\.\\./)*types/"],
                        ["^utils/", "^(\\.\\./)*utils/"], // imports starting with ../
                        ["^\\.\\.(?!/?$)", "^\\.\\./?$"], // imports starting with ./
                        ["^\\./(?=.*/)(?!/?$)", "^\\.(?!/?$)", "^\\./?$"], // css or scss imports
                        ["^.+\\.s?css$"], // side effect imports
                        ["^\\u0000"]
                    ]
                }
            ]
        }
    },

    // -------------------------------------------------------------------------
    // Phase 0 — Guardrails (warn mode; errors enabled per-scope in later phases)
    // -------------------------------------------------------------------------
    // 1. Forbid imports from src/css/  (Phase 3 deleted these files).
    //    Error-level as of Phase 4: src/css/ no longer exists, so any new
    //    `css/*` import is a regression and must be blocked at lint time.
    {
        files: ["**/*.ts", "**/*.tsx", "**/*.js"],
        rules: {
            "no-restricted-imports": [
                "error",
                {
                    patterns: [
                        {
                            group: ["css/*"],
                            message:
                                "App-local CSS imports were removed. " +
                                "Use Tailwind utilities or Kumo catalog tokens instead."
                        }
                    ]
                }
            ]
        }
    },

    // 2. Forbid static inline styles + neutral palette classes + hex colors.
    //    Kept at `warn` globally; the count-based ratchet in
    //    `.github/workflows/ui-baseline.yml` prevents regressions
    //    (counts can only go down). Dynamic / runtime values must use:
    //    // eslint-disable-next-line no-restricted-syntax
    {
        files: ["**/*.tsx", "**/*.ts"],
        rules: {
            "no-restricted-syntax": [
                "warn",
                {
                    selector:
                        "JSXAttribute[name.name='style'] > JSXExpressionContainer > ObjectExpression",
                    message:
                        "Static inline styles are deprecated (Phase 1 migration). " +
                        "Replace with Tailwind utilities. " +
                        "For runtime-calculated values add: // eslint-disable-next-line no-restricted-syntax"
                },
                {
                    // Generic neutral Tailwind palette in any string literal (className strings).
                    selector:
                        "Literal[value=/(^|[\\s\\\"'`])(text|bg|border|ring|outline|fill|stroke|divide)-(slate|gray|zinc|neutral|stone)-[0-9]+/]",
                    message:
                        "Generic neutral Tailwind classes are deprecated when a Kumo token exists. " +
                        "Use text-title-primary, text-title-secondary, bg-surface-grey, " +
                        "border-border-primary, border-border-secondary, text-icon-primary, text-icon-secondary, etc."
                },
                {
                    // Template literal version of the same (className=`text-slate-500 …`).
                    selector:
                        "TemplateElement[value.raw=/(^|[\\s\\\"'`])(text|bg|border|ring|outline|fill|stroke|divide)-(slate|gray|zinc|neutral|stone)-[0-9]+/]",
                    message:
                        "Generic neutral Tailwind classes are deprecated when a Kumo token exists. " +
                        "Use Kumo design tokens instead (text-title-primary, bg-surface-grey, etc.)."
                },
                {
                    // String literal that IS a hex color (covers style={{ color: '#292D36' }}, const C = '#abc', etc.)
                    selector: "Literal[value=/^#[0-9a-fA-F]{3,8}$/]",
                    message:
                        "Hardcoded hex color literals are deprecated. " +
                        "Use Kumo design tokens (e.g., text-icon-primary, bg-surface-grey) or var(--token-name)."
                },
                {
                    // Arbitrary hardcoded length in a Tailwind class, e.g. a bracketed
                    // px/rem/em value. var(--*) and viewport units (vh/vw/…) are
                    // intentionally NOT matched — the migration plan allows them; only
                    // hardcoded px/rem/em must move to the scale.
                    selector: "Literal[value=/\\[[0-9.]+(px|rem|em)\\]/]",
                    message:
                        "Hardcoded arbitrary Tailwind length (bracketed px/rem/em) is deprecated. " +
                        "Use the Tailwind spacing/size scale or a Kumo token."
                },
                {
                    selector:
                        "TemplateElement[value.raw=/\\[[0-9.]+(px|rem|em)\\]/]",
                    message:
                        "Hardcoded arbitrary Tailwind length (bracketed px/rem/em) is deprecated. " +
                        "Use the Tailwind spacing/size scale or a Kumo token."
                },
                {
                    // Arbitrary hex inside a Tailwind class (bracketed #-hex color).
                    selector: "Literal[value=/\\[#[0-9a-fA-F]{3,8}\\]/]",
                    message:
                        "Hardcoded arbitrary hex Tailwind class (bracketed #-hex) is deprecated. " +
                        "Use a Kumo color token."
                },
                {
                    selector:
                        "TemplateElement[value.raw=/\\[#[0-9a-fA-F]{3,8}\\]/]",
                    message:
                        "Hardcoded arbitrary hex Tailwind class (bracketed #-hex) is deprecated. " +
                        "Use a Kumo color token."
                }
            ]
        }
    },

    // 5a. @ui import restrictions (layer violations) — Phase 4: ON as `warn`.
    //     @ui is presentation-only: no app CSS, data-fetching libs, runtime
    //     gen API hooks, hooks, stores, or pages. Type-only gen/* imports are
    //     allowed (erased at compile time). Flip to `error` once clean.
    {
        files: ["src/components/@ui/**/*.ts", "src/components/@ui/**/*.tsx"],
        plugins: { "@typescript-eslint": tseslint.plugin },
        rules: {
            "@typescript-eslint/no-restricted-imports": [
                "warn",
                {
                    patterns: [
                        {
                            group: ["css/**", "**/css/**"],
                            message:
                                "@ui must not import app CSS (presentation layer)."
                        },
                        {
                            group: ["@tanstack/react-query"],
                            message:
                                "@ui must not import data-fetching libraries (presentation only)."
                        },
                        {
                            group: ["gen/v0/**", "gen/v1/**"],
                            allowTypeImports: true,
                            message:
                                "@ui may import TYPES from gen/* but not runtime API hooks."
                        },
                        {
                            group: ["hooks/**", "stores/**", "pages/**"],
                            message:
                                "@ui must not import app hooks, stores, or pages (presentation only)."
                        }
                    ]
                }
            ]
        }
    },

    // 5c. Layer import restrictions for non-@ui layers — Phase 4: ON as `warn`.
    //     Path-based mirror of the audit's layerViolations() matrix. Used
    //     instead of relying on eslint-plugin-boundaries alone because no import
    //     resolver is installed, so boundaries can't resolve baseUrl aliases
    //     (e.g. "hooks/foo"). @hoc may import @ui/@hoc; data layers may not
    //     import any component or page. Uses the @typescript-eslint variant so
    //     it doesn't override the core no-restricted-imports in rule #1 (css/*).
    {
        files: ["src/components/@hoc/**/*.ts", "src/components/@hoc/**/*.tsx"],
        plugins: { "@typescript-eslint": tseslint.plugin },
        rules: {
            "@typescript-eslint/no-restricted-imports": [
                "warn",
                {
                    patterns: [
                        {
                            group: ["pages/**", "hooks/**", "stores/**"],
                            message:
                                "@hoc must not import pages, hooks, or stores."
                        }
                    ]
                }
            ]
        }
    },
    {
        files: ["src/hooks/**/*.ts", "src/hooks/**/*.tsx"],
        plugins: { "@typescript-eslint": tseslint.plugin },
        rules: {
            "@typescript-eslint/no-restricted-imports": [
                "warn",
                {
                    patterns: [
                        {
                            group: ["components/**", "pages/**"],
                            message:
                                "hooks must not import components or pages."
                        }
                    ]
                }
            ]
        }
    },
    {
        files: ["src/stores/**/*.ts", "src/stores/**/*.tsx"],
        plugins: { "@typescript-eslint": tseslint.plugin },
        rules: {
            "@typescript-eslint/no-restricted-imports": [
                "warn",
                {
                    patterns: [
                        {
                            group: ["components/**", "pages/**", "hooks/**"],
                            message:
                                "stores must not import components, pages, or hooks."
                        }
                    ]
                }
            ]
        }
    },
    {
        files: ["src/services/**/*.ts", "src/services/**/*.tsx"],
        plugins: { "@typescript-eslint": tseslint.plugin },
        rules: {
            "@typescript-eslint/no-restricted-imports": [
                "warn",
                {
                    patterns: [
                        {
                            group: [
                                "components/**",
                                "pages/**",
                                "hooks/**",
                                "stores/**"
                            ],
                            message:
                                "services must not import components, pages, hooks, or stores."
                        }
                    ]
                }
            ]
        }
    },
    {
        files: ["src/utils/**/*.ts", "src/utils/**/*.tsx"],
        plugins: { "@typescript-eslint": tseslint.plugin },
        rules: {
            "@typescript-eslint/no-restricted-imports": [
                "warn",
                {
                    patterns: [
                        {
                            group: [
                                "components/**",
                                "pages/**",
                                "hooks/**",
                                "stores/**",
                                "services/**"
                            ],
                            // Type-only imports are erased at compile time and
                            // create no runtime coupling (matches the migration
                            // audit, which ignores `import type`).
                            allowTypeImports: true,
                            message:
                                "utils must not import components, pages, hooks, stores, or services."
                        }
                    ]
                }
            ]
        }
    },

    // 5b. Complexity / coupling caps — Phase 4: ON as `error`.
    //     `complexity` is held at 20 (ESLint's default) and enforced on ALL
    //     code: new/edited functions above 20 fail lint. The 192 functions
    //     that already exceeded 20 are grandfathered via the committed
    //     `eslint-suppressions.json` (ESLint native bulk suppressions) — a
    //     one-directional ratchet that ESLint flags as stale once an offender
    //     is refactored below 20. Regenerate after refactors with:
    //       npx eslint --config eslint.config.ts --suppress-rule complexity src
    //       npx eslint --config eslint.config.ts --prune-suppressions src
    //     Phase 4 end target: complexity 15 (tighten the cap, re-suppress).
    //     max-depth / max-lines-per-function remain plain current-ceiling
    //     ratchets (no suppressions) — lower them as offenders are refactored.
    {
        files: ["**/*.ts", "**/*.tsx"],
        ignores: [
            "**/__tests__/**",
            "**/*.test.ts",
            "**/*.test.tsx",
            "src/tests/**",
            "src/gen/**"
        ],
        rules: {
            complexity: ["error", 20],
            "max-depth": ["error", 6],
            "max-lines-per-function": [
                "error",
                { max: 876, skipBlankLines: true, skipComments: true }
            ]
        }
    },

    // 6. eslint-plugin-boundaries — Phase 4: ON as `warn` (complementary).
    //    Catches element-to-element crossings that resolve via Node (e.g.
    //    relative imports between layers). It can't resolve baseUrl aliases
    //    ("hooks/foo") without an import resolver, so the path-based rules in
    //    #5a/#5c are the primary layer enforcement; this adds the relative-path
    //    cases they miss. To make boundaries comprehensive, add
    //    eslint-import-resolver-typescript and an `import/resolver` setting.
    //    Element order matters: the most specific (@hoc, @ui) must precede the
    //    generic `components`.
    {
        files: ["src/**/*.ts", "src/**/*.tsx"],
        // Tests may import shared fixtures/mocks across feature folders; layer
        // rules apply to shipped source only (mirrors the audit's bidirectional
        // metric, which also excludes tests).
        ignores: ["**/__tests__/**", "**/*.test.ts", "**/*.test.tsx"],
        plugins: { boundaries: pluginBoundaries },
        settings: {
            "boundaries/include": ["src/**/*"],
            "boundaries/elements": [
                { type: "hoc", pattern: "src/components/@hoc/**" },
                { type: "ui", pattern: "src/components/@ui/**" },
                { type: "components", pattern: "src/components/**" },
                { type: "pages", pattern: "src/pages/**" },
                { type: "hooks", pattern: "src/hooks/**" },
                { type: "stores", pattern: "src/stores/**" },
                { type: "services", pattern: "src/services/**" },
                { type: "utils", pattern: "src/utils/**" }
            ]
        },
        rules: {
            "boundaries/dependencies": [
                "warn",
                {
                    default: "allow",
                    rules: [
                        {
                            from: { type: "ui" },
                            disallow: {
                                to: {
                                    type: [
                                        "hooks",
                                        "stores",
                                        "pages",
                                        "components"
                                    ]
                                }
                            },
                            message:
                                "@ui (presentation) must not import ${dependency.type}."
                        },
                        {
                            from: { type: "hoc" },
                            disallow: {
                                to: {
                                    type: [
                                        "hooks",
                                        "stores",
                                        "pages",
                                        "components"
                                    ]
                                }
                            },
                            message: "@hoc must not import ${dependency.type}."
                        },
                        {
                            from: { type: "hooks" },
                            disallow: {
                                to: { type: ["components", "pages"] }
                            },
                            message: "hooks must not import ${dependency.type}."
                        },
                        {
                            from: { type: "stores" },
                            disallow: {
                                to: { type: ["components", "pages", "hooks"] }
                            },
                            message:
                                "stores must not import ${dependency.type}."
                        },
                        {
                            from: { type: "services" },
                            disallow: {
                                to: {
                                    type: [
                                        "components",
                                        "pages",
                                        "hooks",
                                        "stores"
                                    ]
                                }
                            },
                            message:
                                "services must not import ${dependency.type}."
                        },
                        {
                            from: { type: "utils" },
                            disallow: {
                                to: {
                                    type: [
                                        "components",
                                        "pages",
                                        "hooks",
                                        "stores",
                                        "services"
                                    ]
                                }
                            },
                            message: "utils must not import ${dependency.type}."
                        }
                    ]
                }
            ]
        }
    }
];
