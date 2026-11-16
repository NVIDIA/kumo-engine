// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
    Admin,
    ConnectorsList,
    ConnectTables,
    GraphsList,
    JobsList,
    NewConnector,
    NewGraph,
    NewPredictionJob,
    NewTrainingJob,
    TablesList
} from "routes/Routes";

import { usePreloadRoutes } from "../preloadRoutes";

vi.mock("routes/Routes", () => {
    const mockRoute = () => ({ preload: vi.fn().mockResolvedValue(undefined) });
    return {
        Admin: mockRoute(),
        ConnectorsList: mockRoute(),
        ConnectTables: mockRoute(),
        GraphsList: mockRoute(),
        JobsList: mockRoute(),
        NewConnector: mockRoute(),
        NewGraph: mockRoute(),
        NewPredictionJob: mockRoute(),
        NewTrainingJob: mockRoute(),
        TablesList: mockRoute()
    };
});

beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
});

afterEach(() => {
    vi.useRealTimers();
});

describe("usePreloadRoutes", () => {
    it.each([
        {
            advanceMs: 1,
            modules: [ConnectorsList, TablesList, GraphsList, JobsList],
            description: "tier 1 immediately after idle callback"
        },
        {
            advanceMs: 501,
            modules: [
                NewConnector,
                ConnectTables,
                NewGraph,
                NewTrainingJob,
                NewPredictionJob
            ],
            description: "tier 2 after 500ms"
        },
        {
            advanceMs: 1501,
            modules: [Admin],
            description: "tier 3 after additional 1000ms"
        }
    ])("Should preload $description routes", ({ advanceMs, modules }) => {
        renderHook(() => usePreloadRoutes());
        vi.advanceTimersByTime(advanceMs);
        modules.forEach((mod) => expect(mod.preload).toHaveBeenCalled());
    });

    it("Should not preload again when re-rendered", () => {
        const { rerender } = renderHook(() => usePreloadRoutes());

        vi.advanceTimersByTime(1);
        vi.clearAllMocks();

        rerender();
        vi.advanceTimersByTime(1);

        expect(ConnectorsList.preload).not.toHaveBeenCalled();
    });
});
