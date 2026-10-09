// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from "vitest";
import { render } from "vitest-browser-react";
import { composeStories } from "@storybook/react";
import * as stories from "@/stories/components/blocks/DistributionChart.stories";

const composedStories = composeStories(stories);

// BarChart has 23 data points with long x-axis labels. ECharts auto-samples
// which labels to show based on container width, causing minor layout
// differences between runs.
const LABEL_SAMPLING_STORIES = ["BarChart"];

describe("DistributionChart Visual Regression", () => {
    Object.entries(composedStories).forEach(([storyName, Story]) => {
        it(`${storyName} story matches screenshot`, async () => {
            const { container } = await render(<Story />);
            if (LABEL_SAMPLING_STORIES.includes(storyName)) {
                await expect(container).toMatchScreenshot(
                    `distribution-chart-${storyName}`,
                    {
                        comparatorOptions: {
                            threshold: 0.2,
                            allowedMismatchedPixelRatio: 0.05
                        }
                    }
                );
            } else {
                await expect(container).toMatchScreenshot(
                    `distribution-chart-${storyName}`
                );
            }
        });
    });
});
