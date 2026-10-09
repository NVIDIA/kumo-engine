// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render } from "vitest-browser-react";
import { composeStories } from "@storybook/react";
import * as stories from "@/stories/components/ui/TablesAdvanced.stories";

const composedStories = composeStories(stories);

// LargeDataset uses createSeededRandom for Math.random (story-level).
// new Date() still reads the system clock, so freeze it here.
beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2024-06-15T00:00:00.000Z"));
});

afterEach(() => {
    vi.useRealTimers();
});

describe("TablesAdvanced Visual Regression", () => {
    Object.entries(composedStories).forEach(([storyName, Story]) => {
        it(`${storyName} story matches screenshot`, async () => {
            const { container } = await render(<Story />);
            await expect(container).toMatchScreenshot(
                `tables-advanced-${storyName}`
            );
        });
    });
});
