// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useBodyScrollLock } from "../useBodyScrollLock";

describe("useBodyScrollLock", () => {
    let originalOverflow: string;

    const renderScrollLock = (locked: boolean) => {
        return renderHook(({ locked }) => useBodyScrollLock(locked), {
            initialProps: { locked }
        });
    };

    beforeEach(() => {
        originalOverflow = document.body.style.overflow;
        document.body.style.overflow = "";
    });

    afterEach(() => {
        document.body.style.overflow = originalOverflow;
        vi.clearAllMocks();
    });

    it.each([
        { description: "false", isLocked: false, expectedOverflow: "" },
        { description: "true", isLocked: true, expectedOverflow: "hidden" }
    ])(
        "Should set overflow to '$expectedOverflow' when isLocked is $description",
        ({ isLocked, expectedOverflow }) => {
            renderScrollLock(isLocked);
            expect(document.body.style.overflow).toBe(expectedOverflow);
        }
    );

    it("Should unlock scroll when isLocked changes from true to false", () => {
        const { rerender } = renderScrollLock(true);

        expect(document.body.style.overflow).toBe("hidden");

        rerender({ locked: false });
        expect(document.body.style.overflow).toBe("");
    });

    it("Should restore original overflow value on unmount", () => {
        document.body.style.overflow = "auto";

        const { unmount } = renderScrollLock(true);

        expect(document.body.style.overflow).toBe("hidden");

        unmount();
        expect(document.body.style.overflow).toBe("auto");
    });
});
