// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as a11yAddonAnnotations from "@storybook/addon-a11y/preview";
import { beforeAll } from "vitest";
import { setProjectAnnotations } from "@storybook/react-vite";
import * as previewAnnotations from "./preview";

const annotations = setProjectAnnotations([
    a11yAddonAnnotations,
    previewAnnotations
]);

beforeAll(annotations.beforeAll);

// Disable CSS animations/transitions for stable visual tests
const style = document.createElement("style");
style.textContent = `
  *, *::before, *::after {
    animation-duration: 0s !important;
    animation-delay: 0s !important;
    transition-duration: 0s !important;
    transition-delay: 0s !important;
  }
`;
document.head.appendChild(style);

// Disable ECharts JS-based animations for deterministic visual regression tests.
// The CSS override above only affects CSS animations/transitions, not ECharts'
// internal requestAnimationFrame-based animation system.
// Dynamically import echarts (static import breaks the Vitest runner),
// then patch the ECharts prototype so all instances have animation disabled.
beforeAll(async () => {
    const echarts = await import("echarts");
    const tempContainer = document.createElement("div");
    tempContainer.style.width = "1px";
    tempContainer.style.height = "1px";
    document.body.appendChild(tempContainer);
    const tempInstance = echarts.init(tempContainer);
    const EChartsProto = Object.getPrototypeOf(tempInstance);
    const originalSetOption = EChartsProto.setOption;
    EChartsProto.setOption = function (
        option: Record<string, unknown>,
        ...rest: unknown[]
    ) {
        return originalSetOption.call(
            this,
            { ...option, animation: false },
            ...rest
        );
    };
    echarts.dispose(tempInstance);
    tempContainer.remove();
});
