// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, mergeConfig } from "vitest/config";
import { storybookTest } from "@storybook/addon-vitest/vitest-plugin";
import { playwright } from "@vitest/browser-playwright";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import viteConfig from "../vite.config";

const __dirname = dirname(fileURLToPath(import.meta.url));

export default mergeConfig(
    viteConfig,
    defineConfig({
        optimizeDeps: {
            include: ["@storybook/react-dom-shim", "react-dom/client"]
        },
        plugins: [
            storybookTest({
                configDir: __dirname,
                storybookScript: "npm run storybook -- --ci"
            })
        ],
        test: {
            name: "storybook",
            retry: 2,
            browser: {
                enabled: true,
                headless: true,
                provider: playwright(),
                instances: [{ browser: "chromium" }],
                viewport: {
                    width: 1280,
                    height: 720
                },
                screenshotDirectory: join(__dirname, "../src/__screenshots__"),
                expect: {
                    toMatchScreenshot: {
                        comparatorOptions: {
                            threshold: 0.2,
                            allowedMismatchedPixelRatio: 0.01
                        }
                    }
                }
            },
            setupFiles: [join(__dirname, "vitest.setup.ts")],
            coverage: {
                provider: "istanbul",
                reportsDirectory: "./coverage",
                reporter: ["text", "lcov", "html"],
                include: ["src/components/**/*.{ts,tsx}"],
                exclude: [
                    "node_modules/",
                    "**/*.stories.tsx",
                    "**/*.test.{ts,tsx}",
                    "src/icons/**",
                    "src/types/**",
                    "src/assets/**",
                    "src/fonts/**",
                    "src/images/**",
                    "src/animations/**"
                ]
            }
        }
    })
);
