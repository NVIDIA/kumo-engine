// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, vi } from "vitest";
import { render } from "vitest-browser-react";
import { composeStories } from "@storybook/react";
import * as stories from "@/stories/components/blocks/XaiSubgraph.stories";

const composedStories = composeStories(stories);

describe("XaiSubgraph Visual Regression", () => {
    Object.entries(composedStories).forEach(([storyName, Story]) => {
        it(`${storyName} story matches screenshot`, async () => {
            vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
            try {
                const { container } = await render(<Story />);
                await vi.waitUntil(() => vi.getTimerCount() > 0, {
                    timeout: 2000,
                    interval: 10
                });
                await vi.advanceTimersByTimeAsync(100);
                vi.useRealTimers();
                await expect(container).toMatchScreenshot(
                    `xai-subgraph-${storyName}`
                );
            } finally {
                vi.useRealTimers();
            }
        });
    });
});
