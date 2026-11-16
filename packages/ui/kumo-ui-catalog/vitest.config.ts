// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, mergeConfig } from "vitest/config";
import viteConfig from "./vite.config";

export default mergeConfig(
    viteConfig,
    defineConfig({
        test: {
            name: "unit",
            globals: true,
            environment: "happy-dom",
            setupFiles: ["./test-utils/setup.ts"],
            include: ["src/**/*.test.{ts,tsx}"],
            exclude: ["**/*.stories.tsx", "**/node_modules/**"],
            coverage: {
                provider: "istanbul",
                reportsDirectory: "./coverage",
                reporter: ["text", "lcov", "html"],
                include: ["src/**/*.{ts,tsx}"],
                exclude: [
                    "node_modules/",
                    ".storybook/",
                    "**/*.stories.tsx",
                    "**/main.tsx",
                    "**/vite-env.d.ts",

                    // Remove each of these once the test is added
                    "src/App.tsx",
                    "src/stories/**",
                    "src/icons/**",
                    "src/types/**",
                    "src/assets/**",
                    "src/fonts/**",
                    "src/images/**",
                    "src/animations/**",
                    "src/hooks/**",
                    "src/utils/**",
                    "src/lib/**",
                    "src/layout.tsx",
                    "src/components/theme-provider.tsx",
                    "src/components/blocks/chart/**",
                    "src/components/blocks/data-table/**",
                    "src/components/blocks/layout/**",
                    "src/components/blocks/sidebar/**",
                    "src/components/blocks/visual-graph/**",

                    // UI components without tests (remove individually as tests are added)
                    "src/components/ui/avatar.tsx",
                    "src/components/ui/breadcrumb.tsx",
                    "src/components/ui/card.tsx",
                    "src/components/ui/checkbox.tsx",
                    "src/components/ui/collapsible.tsx",
                    "src/components/ui/combobox.tsx",
                    "src/components/ui/command.tsx",
                    "src/components/ui/confusion-metrics.tsx",
                    "src/components/ui/copy-to-clipboard.tsx",
                    "src/components/ui/datetimeinput.tsx",
                    "src/components/ui/description.tsx",
                    "src/components/ui/dialog.tsx",
                    "src/components/ui/dropdown-menu.tsx",
                    "src/components/ui/empty-table.tsx",
                    "src/components/ui/emptyentity.tsx",
                    "src/components/ui/form.tsx",
                    "src/components/ui/hover-card.tsx",
                    "src/components/ui/key-pair.tsx",
                    "src/components/ui/label.tsx",
                    "src/components/ui/list-view.tsx",
                    "src/components/ui/markdown.tsx",
                    "src/components/ui/metric-card.tsx",
                    "src/components/ui/output-card.tsx",
                    "src/components/ui/pagination.tsx",
                    "src/components/ui/popover.tsx",
                    "src/components/ui/progress.tsx",
                    "src/components/ui/property-card.tsx",
                    "src/components/ui/query-card.tsx",
                    "src/components/ui/radio-group.tsx",
                    "src/components/ui/related-job-card.tsx",
                    "src/components/ui/scroll-area.tsx",
                    "src/components/ui/search.tsx",
                    "src/components/ui/section-header.tsx",
                    "src/components/ui/select.tsx",
                    "src/components/ui/sheet.tsx",
                    "src/components/ui/sidebar.tsx",
                    "src/components/ui/skeleton.tsx",
                    "src/components/ui/slider.tsx",
                    "src/components/ui/sonner.tsx",
                    "src/components/ui/start-job-popover.tsx",
                    "src/components/ui/stats-card.tsx",
                    "src/components/ui/status-card.tsx",
                    "src/components/ui/switch.tsx",
                    "src/components/ui/symbol.tsx",
                    "src/components/ui/table.tsx",
                    "src/components/ui/tabs.tsx",
                    "src/components/ui/text-area.tsx",
                    "src/components/ui/text-input.tsx",
                    "src/components/ui/text-link.tsx",
                    "src/components/ui/text-scroll-area.tsx",
                    "src/components/ui/threshold-metrics.tsx",
                    "src/components/ui/toggle-group.tsx",
                    "src/components/ui/tooltip.tsx",
                    "src/components/ui/calendar.tsx",
                    "src/components/ui/singleselect.tsx",
                    "src/components/ui/popover-button.tsx",
                    "src/components/ui/input.tsx",
                    "src/components/ui/separator.tsx"
                ],
                thresholds: {
                    branches: 80,
                    functions: 80,
                    lines: 80,
                    statements: 80,
                    autoUpdate: false
                }
            }
        }
    })
);
