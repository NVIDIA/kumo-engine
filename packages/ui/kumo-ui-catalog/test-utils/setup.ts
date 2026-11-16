// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { composeStories } from "@storybook/react";

// Cleanup after each test
afterEach(() => {
    cleanup();
});

// Make common test utilities globally available
vi.stubGlobal("composeStories", composeStories);

// Inject CSS to disable animations for stable visual tests
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
