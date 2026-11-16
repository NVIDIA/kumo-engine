// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useSidebarLinksPrefetch } from "../route";

import "tests/mockAxios";

vi.mock("react-router", () => ({
    useLocation: vi.fn().mockReturnValue({ pathname: "/", search: "" }),
    useParams: vi.fn().mockReturnValue({})
}));

let queryClient: QueryClient;

beforeEach(() => {
    vi.clearAllMocks();
    queryClient = createQueryClient();
});

afterEach(() => {
    vi.unstubAllGlobals();
    queryClient.clear();
});

const renderSidebarPrefetch = () =>
    renderHook(() => useSidebarLinksPrefetch(), {
        wrapper: createWrapper(queryClient)
    }).result.current;

const triggerHover = (hoverLink: string) => {
    vi.spyOn(queryClient, "prefetchQuery").mockResolvedValue(undefined);
    renderSidebarPrefetch()(hoverLink);
};

describe("useSidebarLinksPrefetch", () => {
    it.each([
        "Connectors",
        "Tables",
        "Graphs",
        "Models",
        "Predictions",
        "NewPrediction"
    ])("Should trigger prefetch on %s hover", (hoverLink) => {
        triggerHover(hoverLink);
        expect(queryClient.prefetchQuery).toHaveBeenCalled();
    });

    it("Should not trigger prefetch for unknown hover link", () => {
        triggerHover("Unknown");
        expect(queryClient.prefetchQuery).not.toHaveBeenCalled();
    });
});
