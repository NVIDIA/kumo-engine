// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    isRouteErrorResponse,
    useLocation,
    useNavigate,
    useRouteError
} from "react-router";

import { vi } from "vitest";

type LocationState = Record<string, unknown> | null;

// Mock the react-router module
vi.mock("react-router", async () => {
    const routerData = await vi.importActual("react-router");
    return {
        ...routerData,
        useLocation: vi.fn(() => ({
            state: null // Default state, can be dynamically changed
        })), // Mock useLocation
        useNavigate: vi.fn(), // Mock useNavigate with the mock
        useRouteError: vi.fn(),
        isRouteErrorResponse: vi.fn()
    };
});

// Define the mock functions
export const mockUseLocation = vi.fn(() => ({
    state: null as LocationState // Default state, can be dynamically changed
}));

export const mockUseNavigate = vi.fn();

export const mockNavigateCall = () => {
    vi.mocked(useNavigate).mockReturnValue(mockUseNavigate);
};

export const mockRouteError = (error: unknown) => {
    vi.mocked(useRouteError).mockReturnValue(error);
};

export const mockIsRouteErrorResponse = (value: boolean) => {
    vi.mocked(isRouteErrorResponse).mockReturnValue(value);
};

export const mockLocationStateValue = (
    data: LocationState,
    pathName?: string
) => {
    vi.mocked(useLocation).mockReturnValue({
        state: data,
        key: "",
        pathname: pathName ?? "",
        search: "",
        hash: ""
    });
};
