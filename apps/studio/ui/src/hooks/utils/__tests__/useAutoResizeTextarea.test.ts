// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useAutoResizeTextarea } from "../useAutoResizeTextarea";

describe("useAutoResizeTextarea", () => {
    let mockTextarea: HTMLTextAreaElement;

    const setScrollHeight = (height: number) => {
        Object.defineProperty(mockTextarea, "scrollHeight", {
            writable: true,
            configurable: true,
            value: height
        });
    };

    const renderAutoResize = (value: string) => {
        return renderHook(({ value }) => useAutoResizeTextarea(value), {
            initialProps: { value }
        });
    };

    const attachRef = (
        result: ReturnType<typeof renderAutoResize>["result"]
    ) => {
        (
            result.current
                .textareaRef as React.MutableRefObject<HTMLTextAreaElement | null>
        ).current = mockTextarea;
    };

    beforeEach(() => {
        mockTextarea = document.createElement("textarea");
        setScrollHeight(50);
        mockTextarea.value = "test";
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    it("Should return a ref object", () => {
        const { result } = renderAutoResize("test");

        expect(result.current.textareaRef).toBeDefined();
        expect(result.current.textareaRef.current).toBeNull();
    });

    it.each([
        {
            description: "scrollHeight less than maxHeight",
            scrollHeight: 50,
            expectedHeight: "50px",
            expectedOverflow: "hidden"
        },
        {
            description: "scrollHeight exceeds maxHeight (76px)",
            scrollHeight: 150,
            expectedHeight: "76px",
            expectedOverflow: "auto"
        }
    ])(
        "Should set height and overflow when $description",
        ({ scrollHeight, expectedHeight, expectedOverflow }) => {
            const { result, rerender } = renderAutoResize("test");
            attachRef(result);

            setScrollHeight(scrollHeight);
            rerender({ value: "updated" });

            expect(mockTextarea.style.height).toBe(expectedHeight);
            expect(mockTextarea.style.overflowY).toBe(expectedOverflow);
        }
    );

    it.each([
        {
            description: "short content (minimum 360px)",
            textValue: "short",
            expectedWidth: "360px"
        },
        {
            description: "content based on longest line",
            textValue: "a".repeat(100),
            expectedWidth: `${100 * 7 + 15}px`
        },
        {
            description: "long content (maximum 1128px)",
            textValue: "a".repeat(500),
            expectedWidth: "1128px"
        },
        {
            description: "multiline text based on longest line",
            textValue: "short\nthis is a much longer line\nshort again",
            expectedWidth: `${Math.max("this is a much longer line".length * 7 + 15, 360)}px`
        }
    ])("Should set width for $description", ({ textValue, expectedWidth }) => {
        const { result, rerender } = renderAutoResize("test");
        attachRef(result);

        mockTextarea.value = textValue;
        rerender({ value: textValue });

        expect(mockTextarea.style.maxWidth).toBe(expectedWidth);
    });

    it("Should not throw error when ref is not attached", () => {
        const { rerender } = renderAutoResize("test");

        expect(() => {
            rerender({ value: "updated" });
        }).not.toThrow();
    });
});
