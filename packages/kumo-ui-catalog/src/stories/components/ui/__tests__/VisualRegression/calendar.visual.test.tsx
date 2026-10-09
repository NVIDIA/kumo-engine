// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render } from "vitest-browser-react";
import { composeStories } from "@storybook/react";
import * as stories from "@/stories/components/ui/Calendar.stories";

const composedStories = composeStories(stories);

// Pin the current date so the calendar always initialises to the same month
// regardless of when the test is run.
const FIXED_DATE = new Date(2026, 1, 25);

describe("Calendar Visual Regression", () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime(FIXED_DATE);
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    Object.entries(composedStories).forEach(([storyName, Story]) => {
        it(`${storyName} story matches screenshot`, async () => {
            const { container } = await render(<Story />);
            await expect(container).toMatchScreenshot(`calendar-${storyName}`);
        });
    });
});
