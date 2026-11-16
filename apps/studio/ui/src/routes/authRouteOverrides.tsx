// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Navigate, type RouteObject } from "react-router";

import { navRoutes } from "utils/Constants";

interface AuthRouteOverrideOptions {
    authDisabled: boolean;
}

const loginPaths = new Set(["/", "/login"]);

export const resolveAuthRoutes = (
    appRoutes: RouteObject[],
    { authDisabled }: AuthRouteOverrideOptions
) => {
    if (authDisabled) {
        return appRoutes.map((route) =>
            loginPaths.has(route.path || "") || route.path === "/logout"
                ? {
                      ...route,
                      lazy: undefined,
                      element: <Navigate to={navRoutes.Home} replace />
                  }
                : route
        );
    }

    return appRoutes;
};
