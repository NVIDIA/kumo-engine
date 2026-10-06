// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StorybookConfig } from "@storybook/react-vite";

const config: StorybookConfig = {
    "stories": ["../src/**/*.mdx", "../src/**/*.stories.@(js|jsx|mjs|ts|tsx)"],
    "addons": [
        "@storybook/addon-themes",
        "@storybook/addon-a11y",
        "@storybook/addon-vitest",
        "@storybook/addon-docs"
    ],
    "framework": {
        "name": "@storybook/react-vite",
        "options": {}
    },
    "core": {
        "disableTelemetry": true
    }
};
export default config;
