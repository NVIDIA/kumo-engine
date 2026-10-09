// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, mergeConfig } from "vitest/config";
import { playwright } from "@vitest/browser-playwright";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import viteConfig from "../vite.config";

const __dirname = dirname(fileURLToPath(import.meta.url));

export default mergeConfig(
    viteConfig,
    defineConfig({
        optimizeDeps: {
            include: ["react-dom/client"]
        },
        test: {
            name: "visual",
            retry: 2,
            include: ["src/**/*.visual.test.{ts,tsx}"],
            browser: {
                enabled: true,
                headless: true,
                provider: playwright({
                    launchOptions: {
                        args: [
                            "--force-device-scale-factor=1",
                            "--font-render-hinting=none",
                            "--disable-font-subpixel-positioning",
                            "--disable-lcd-text"
                        ]
                    }
                }),
                instances: [{ browser: "chromium" }],
                viewport: {
                    width: 1280,
                    height: 720
                },
                expect: {
                    toMatchScreenshot: {
                        comparatorOptions: {
                            // Per-pixel color tolerance (0–1), ignores minor anti-aliasing differences
                            threshold: 0.2,
                            // Fail if more than 1% of total pixels exceed the threshold
                            allowedMismatchedPixelRatio: 0.01
                        }
                    }
                }
            },
            setupFiles: [join(__dirname, "vitest.setup.ts")]
        }
    })
);
