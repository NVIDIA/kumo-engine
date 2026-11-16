// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// eslint-disable-next-line simple-import-sort/imports
import { cleanup } from "@testing-library/react";
import { debug_info } from "stores/__tests__/MockResponse/DebugInfo";
import { mockAxiosInstance } from "./mockAxios";
import { useDebugStore } from "stores/debugStore";
import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";

import { storageKeys } from "utils/Constants";

import "@testing-library/jest-dom/vitest";

vi.mock("@tanstack/react-virtual", () => {
    return {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        useVirtualizer: (options: any) => {
            // Return a mock virtualizer that renders ALL items
            // Cache items array but regenerate when count changes
            let cachedCount = -1;
            let cachedItems: {
                index: number;
                start: number;
                size: number;
                end: number;
                key: number;
                lane: number;
            }[] = [];

            const generateItems = (count: number) => {
                if (count !== cachedCount || cachedItems.length !== count) {
                    cachedCount = count;
                    cachedItems = [];
                    for (let i = 0; i < count; i++) {
                        cachedItems.push({
                            index: i,
                            start: i * 32,
                            size: 32,
                            end: (i + 1) * 32,
                            key: i,
                            lane: 0
                        });
                    }
                }
                return cachedItems;
            };

            const virtualizer = {
                getVirtualItems: () => {
                    const count = options.count || 0;
                    return generateItems(count);
                },
                getTotalSize: () => (options.count || 0) * 32,
                scrollToIndex: vi.fn(),
                scrollToOffset: vi.fn(),
                measure: vi.fn(),
                measureElement: vi.fn(() => () => {}),
                options
            };

            return virtualizer;
        },
        // Export other utilities that might be used
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        defaultRangeExtractor: (range: any) => {
            const items = [];
            for (let i = range.startIndex; i <= range.endIndex; i++) {
                items.push(i);
            }
            return items;
        },
        observeElementRect: () => () => {},
        observeElementOffset: () => () => {},
        observeWindowRect: () => () => {},
        observeWindowOffset: () => () => {}
    };
});

// Suppress CSS-related errors and warnings that may occur in test environments
// This handles both jsdom and happy-dom CSS parsing issues
if (typeof window !== "undefined") {
    const originalError = console.error;
    const originalWarn = console.warn;

    console.error = (...args: unknown[]) => {
        const errorString = args[0]?.toString() || "";
        // Suppress CSS parsing errors
        if (
            errorString.includes("Could not parse CSS") ||
            errorString.includes("CSS stylesheet")
        ) {
            return;
        }
        originalError(...args);
    };

    console.warn = (...args: unknown[]) => {
        const warnString = args[0]?.toString() || "";
        // Suppress CSS-related warnings
        if (
            warnString.includes("Could not parse CSS") ||
            warnString.includes("CSS stylesheet")
        ) {
            return;
        }
        originalWarn(...args);
    };
}

// Mock ResizeObserver for @tanstack/react-virtual.
// Constructor accepts the standard ResizeObserverCallback signature so the mock
// shape matches lib.dom.d.ts; the callback is never invoked because observe()
// is a no-op in tests.
global.ResizeObserver = class ResizeObserver {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    constructor(callback: ResizeObserverCallback) {}
    observe() {}
    unobserve() {}
    disconnect() {}
};

// Mock getBoundingClientRect for virtualization
// We need to return large dimensions for the scrollable container
// so that all virtualized items are considered "visible" and rendered in tests
Element.prototype.getBoundingClientRect = vi.fn(function (this: Element) {
    const className = this.className || "";
    const style = this.getAttribute?.("style") || "";
    const id = this.id || "";

    // Check if this is a scrollable container used by virtualizer
    // Look for overflow-auto class or max-height style
    const isScrollContainer =
        className.includes("overflow-auto") ||
        style.includes("max-height") ||
        style.includes("maxHeight") ||
        (this.hasAttribute?.("data-testid") &&
            this.getAttribute("data-testid")?.includes("listbox"));

    // Check if this is a chart container (ECharts or similar)
    const idLower = id.toLowerCase();
    const isChartContainer =
        idLower.includes("chart") ||
        idLower.includes("heatmap") ||
        className.includes("echarts") ||
        className.includes("chart");

    if (isScrollContainer) {
        // Return a very large height so all virtualized items are rendered
        return {
            width: 500,
            height: 10000, // Large height to make all items visible
            top: 0,
            left: 0,
            bottom: 10000,
            right: 500,
            x: 0,
            y: 0,
            toJSON: () => {}
        };
    }

    if (isChartContainer) {
        // Return proper dimensions for chart containers
        return {
            width: 800,
            height: 400,
            top: 0,
            left: 0,
            bottom: 400,
            right: 800,
            x: 0,
            y: 0,
            toJSON: () => {}
        };
    }

    // For individual items and other elements
    return {
        width: 500,
        height: 32,
        top: 0,
        left: 0,
        bottom: 32,
        right: 500,
        x: 0,
        y: 0,
        toJSON: () => {}
    };
});

// Add clientHeight and scrollHeight mocks for virtualization
Object.defineProperty(HTMLElement.prototype, "clientHeight", {
    configurable: true,
    get: function () {
        const className = this.className || "";
        const style = this.getAttribute?.("style") || "";
        const id = this.id || "";

        // Check for chart containers
        const idLower = id.toLowerCase();
        const isChartContainer =
            idLower.includes("chart") ||
            idLower.includes("heatmap") ||
            className.includes("echarts") ||
            className.includes("chart");

        if (
            className.includes("overflow-auto") ||
            style.includes("max-height")
        ) {
            return 10000; // Large height for scroll containers
        }
        if (isChartContainer) {
            return 400; // Return proper height for chart containers
        }
        return 32;
    }
});

Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
    configurable: true,
    get: function () {
        return 10000; // Return large scroll height
    }
});

Object.defineProperty(HTMLElement.prototype, "offsetHeight", {
    configurable: true,
    get: function () {
        const className = this.className || "";
        const style = this.getAttribute?.("style") || "";
        const id = this.id || "";

        // Check for chart containers
        const idLower = id.toLowerCase();
        const isChartContainer =
            idLower.includes("chart") ||
            idLower.includes("heatmap") ||
            className.includes("echarts") ||
            className.includes("chart");

        if (
            className.includes("overflow-auto") ||
            style.includes("max-height")
        ) {
            return 10000;
        }
        if (isChartContainer) {
            return 400; // Return proper height for chart containers
        }
        return 32;
    }
});

Object.defineProperty(HTMLElement.prototype, "offsetTop", {
    configurable: true,
    get: function () {
        return 0;
    }
});

Object.defineProperty(HTMLElement.prototype, "clientWidth", {
    configurable: true,
    get: function () {
        const className = this.className || "";
        const id = this.id || "";

        // Check for chart containers
        const idLower = id.toLowerCase();
        const isChartContainer =
            idLower.includes("chart") ||
            idLower.includes("heatmap") ||
            className.includes("echarts") ||
            className.includes("chart");

        if (isChartContainer) {
            return 800; // Return proper width for chart containers
        }
        return 500;
    }
});

Object.defineProperty(HTMLElement.prototype, "offsetWidth", {
    configurable: true,
    get: function () {
        const className = this.className || "";
        const id = this.id || "";

        // Check for chart containers
        const idLower = id.toLowerCase();
        const isChartContainer =
            idLower.includes("chart") ||
            idLower.includes("heatmap") ||
            className.includes("echarts") ||
            className.includes("chart");

        if (isChartContainer) {
            return 800; // Return proper width for chart containers
        }
        return 500;
    }
});

// Mock scrollTo and scrollIntoView for virtualization
HTMLElement.prototype.scrollTo = vi.fn();
HTMLElement.prototype.scrollIntoView = vi.fn();

// Mock scrollTop property
Object.defineProperty(HTMLElement.prototype, "scrollTop", {
    configurable: true,
    get: function () {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return (this as any)._scrollTop || 0;
    },
    set: function (value) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (this as any)._scrollTop = value;
    }
});

mockAxiosInstance.get.mockClear(); // Hack to get axios instance loaded and mocked

// Store the mutation observer globally for cleanup
let globalMutationObserver: MutationObserver | null = null;

// Add mutation observer to set proper ARIA roles on CommandItem elements
// This ensures compatibility with tests that use getByRole("option")
if (typeof window !== "undefined" && typeof MutationObserver !== "undefined") {
    const addRoleToOptions = () => {
        // Find all Select option elements and add role="option" if missing
        const optionElements = document.querySelectorAll(
            '[data-testid*="-option-"]'
        );
        optionElements.forEach((el) => {
            if (!el.getAttribute("role")) {
                el.setAttribute("role", "option");
            }
        });
    };

    globalMutationObserver = new MutationObserver((mutations) => {
        let hasRelevantChanges = false;
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === 1) {
                    const element = node as Element;
                    // Only process if it's related to our Select component options
                    if (
                        element.getAttribute &&
                        element
                            .getAttribute("data-testid")
                            ?.includes("-option-")
                    ) {
                        hasRelevantChanges = true;
                    }
                    // Or if it's a container that might contain options
                    if (
                        element.querySelector &&
                        element.querySelector('[data-testid*="-option-"]')
                    ) {
                        hasRelevantChanges = true;
                    }
                }
            });
        });

        // Only update if we found relevant changes
        // Add roles synchronously to avoid timing issues in tests
        if (hasRelevantChanges) {
            addRoleToOptions();
        }
    });

    // Start observing after document is ready
    if (document.body) {
        globalMutationObserver.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    // Also run once on initialization
    if (document.readyState === "complete") {
        addRoleToOptions();
    } else {
        window.addEventListener("load", addRoleToOptions);
    }
}

beforeEach(() => {
    // Every application resource is project-owned. Tests use a stable project
    // unless a scenario explicitly clears or replaces the selection.
    storeToLocalStorage(storageKeys.selectedProject, "project-tests");
    useDebugStore.setState((state) => ({
        ...state,
        configFetched: true,
        data: debug_info
    }));
});

afterEach(() => {
    // Disconnect and reconnect mutation observer to clear any pending mutations
    if (globalMutationObserver && document.body) {
        globalMutationObserver.disconnect();
        globalMutationObserver.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    cleanup();
    vi.clearAllMocks();
});
