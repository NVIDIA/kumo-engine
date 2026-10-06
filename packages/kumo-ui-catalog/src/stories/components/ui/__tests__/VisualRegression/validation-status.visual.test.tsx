// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as stories from "@/stories/components/ui/ValidationStatus.stories";
import { composeStories } from "@storybook/react";
import { describe, expect, it } from "vitest";
import { render } from "vitest-browser-react";

const composedStories = composeStories(stories);

describe("ValidationStatus Visual Regression", () => {
    Object.entries(composedStories).forEach(([storyName, Story]) => {
        it(`${storyName} story matches screenshot`, async () => {
            const { container } = await render(<Story />);
            await expect(container).toMatchScreenshot(
                `validation-status-${storyName}`
            );
        });
    });
});
