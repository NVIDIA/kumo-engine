// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from "vitest";
import { render } from "vitest-browser-react";
import { composeStories } from "@storybook/react";
import * as stories from "@/stories/components/ui/Markdown.stories";

const composedStories = composeStories(stories);

// Excluded from visual regression: uses external image URLs unavailable in CI
const SKIP_VISUAL = new Set(["ImagesSupport"]);

describe("Markdown Visual Regression", () => {
    Object.entries(composedStories).forEach(([storyName, Story]) => {
        if (SKIP_VISUAL.has(storyName)) return;
        it(`${storyName} story matches screenshot`, async () => {
            const { container } = await render(<Story />);
            await expect(container).toMatchScreenshot(`markdown-${storyName}`);
        });
    });
});
