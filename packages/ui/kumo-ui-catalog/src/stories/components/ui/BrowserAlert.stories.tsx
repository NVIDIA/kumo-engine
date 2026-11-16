// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BrowserAlert } from "@/components/ui/browser-alert";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, within } from "storybook/test";

// BrowserAlert renders only when `useIsBrowserSupported()` returns false,
// which checks for "Chrome" in the userAgent. Storybook runs in Chrome, so
// every story decorates the render to spoof a non-Chrome userAgent and force
// the alert to appear.
const spoofUnsupportedBrowser = () => {
    Object.defineProperty(window.navigator, "userAgent", {
        value: "Mozilla/5.0 (UnsupportedBrowser)",
        configurable: true
    });
};

const meta: Meta<typeof BrowserAlert> = {
    title: "UI/BrowserAlert",
    component: BrowserAlert,
    tags: ["autodocs"],
    parameters: { layout: "padded" },
    argTypes: {
        variant: {
            control: { type: "select" },
            options: ["default", "warn", "destructive", "info"]
        }
    },
    decorators: [
        (Story) => {
            spoofUnsupportedBrowser();
            return <Story />;
        }
    ]
};

export default meta;

type Story = StoryObj<typeof BrowserAlert>;

export const Default: Story = {
    args: {
        text: "Kumo works best with Google Chrome. Please consider using Google Chrome for most optimal experience."
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const alert = canvas.getByRole("alert");
        await expect(alert).toBeVisible();
        await expect(alert).toHaveTextContent(/Google Chrome/);
    }
};

export const WarnVariant: Story = {
    args: {
        text: "This browser may not display all UI features correctly.",
        variant: "warn"
    }
};
