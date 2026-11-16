// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import useFluidCardHeight from "../useFluidCardHeight";

describe("useFluidCardHeight", () => {
    const originalGetBoundingClientRect =
        Element.prototype.getBoundingClientRect;

    const stubHeaderBottom = (bottom: number) => {
        Element.prototype.getBoundingClientRect = vi.fn(
            () =>
                ({
                    bottom,
                    top: 0,
                    left: 0,
                    right: 0,
                    width: 0,
                    height: bottom,
                    x: 0,
                    y: 0,
                    toJSON: () => ({})
                }) as DOMRect
        );
    };

    const setWindowHeight = (height: number) => {
        Object.defineProperty(window, "innerHeight", {
            writable: true,
            configurable: true,
            value: height
        });
    };

    beforeEach(() => {
        setWindowHeight(800);
    });

    afterEach(() => {
        Element.prototype.getBoundingClientRect = originalGetBoundingClientRect;
        vi.clearAllMocks();
    });

    const renderWithAttachedRef = (
        options?: Parameters<typeof useFluidCardHeight>[0]
    ) => {
        const hook = renderHook(() =>
            useFluidCardHeight<HTMLDivElement>(options)
        );
        const el = document.createElement("div");
        document.body.appendChild(el);
        (
            hook.result.current
                .headerRef as React.MutableRefObject<HTMLDivElement | null>
        ).current = el;
        return { hook, el };
    };

    it("Should initialize cardHeight as undefined before ref is attached", () => {
        const { result } = renderHook(() => useFluidCardHeight());
        expect(result.current.cardHeight).toBeUndefined();
    });

    it("Should compute height as innerHeight - rect.bottom - bottomPadding on resize", () => {
        stubHeaderBottom(100);
        const { hook } = renderWithAttachedRef();

        act(() => {
            window.dispatchEvent(new Event("resize"));
        });

        // 800 - 100 - 24 = 676
        expect(hook.result.current.cardHeight).toBe(676);
    });

    it("Should clamp to minHeight floor", () => {
        stubHeaderBottom(750);
        const { hook } = renderWithAttachedRef({ minHeight: 300 });

        act(() => {
            window.dispatchEvent(new Event("resize"));
        });

        // 800 - 750 - 24 = 26, clamped to 300
        expect(hook.result.current.cardHeight).toBe(300);
    });

    it("Should respect custom bottomPadding", () => {
        stubHeaderBottom(100);
        const { hook } = renderWithAttachedRef({ bottomPadding: 50 });

        act(() => {
            window.dispatchEvent(new Event("resize"));
        });

        // 800 - 100 - 50 = 650
        expect(hook.result.current.cardHeight).toBe(650);
    });

    it("Should re-measure when window resizes", () => {
        stubHeaderBottom(100);
        const { hook } = renderWithAttachedRef();

        act(() => {
            window.dispatchEvent(new Event("resize"));
        });
        expect(hook.result.current.cardHeight).toBe(676);

        act(() => {
            setWindowHeight(1000);
            window.dispatchEvent(new Event("resize"));
        });
        expect(hook.result.current.cardHeight).toBe(876);
    });

    it("Should remove resize listener on unmount", () => {
        const removeSpy = vi.spyOn(window, "removeEventListener");
        const { hook } = renderWithAttachedRef();

        hook.unmount();

        expect(removeSpy).toHaveBeenCalledWith("resize", expect.any(Function));
    });

    it("Should re-measure when a value in deps changes", () => {
        stubHeaderBottom(100);

        const hook = renderHook(
            ({ showBanner }: { showBanner: boolean }) =>
                useFluidCardHeight<HTMLDivElement>({ deps: [showBanner] }),
            { initialProps: { showBanner: false } }
        );
        const el = document.createElement("div");
        document.body.appendChild(el);
        (
            hook.result.current
                .headerRef as React.MutableRefObject<HTMLDivElement | null>
        ).current = el;

        act(() => {
            window.dispatchEvent(new Event("resize"));
        });
        expect(hook.result.current.cardHeight).toBe(676);

        // Banner toggles on — header position shifts (simulated by changing
        // rect.bottom), and deps change forces a re-measure even though the
        // header itself didn't resize.
        stubHeaderBottom(160);
        act(() => {
            hook.rerender({ showBanner: true });
        });
        expect(hook.result.current.cardHeight).toBe(616);
    });
});
