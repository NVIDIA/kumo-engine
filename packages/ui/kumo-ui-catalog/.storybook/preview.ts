// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Preview } from "@storybook/react-vite";
import { MINIMAL_VIEWPORTS } from "storybook/viewport";
import { withThemeByClassName } from "@storybook/addon-themes";
import "../src/kumo-ui-catalog.css";

const preview: Preview = {
    parameters: {
        controls: {
            matchers: {
                color: /(background|color)$/i,
                date: /Date$/i
            }
        },

        viewport: {
            options: {
                ...MINIMAL_VIEWPORTS
            }
        },

        a11y: {
            // 'todo' - show a11y violations in the test UI only
            // 'error' - fail CI on a11y violations
            // 'off' - skip a11y checks entirely
            test: "error",
            config: {
                rules: [{ id: "color-contrast", enabled: false }]
            }
        }
    },
    decorators: [
        withThemeByClassName({
            themes: {
                light: "light",
                dark: "dark"
            },
            defaultTheme: "light"
        })
    ]
};

export default preview;
