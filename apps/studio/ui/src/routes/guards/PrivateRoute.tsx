// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Navigate } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { useIsLoggedIn } from "stores/userStore";

import MainLayout from "components/Layout/MainLayout";

import { usePreloadRoutes } from "hooks/routes/preloadRoutes";

import { publicPaths, storageKeys } from "utils/Constants";
/**
 * Wrapper component to handle Private Routes
 *
 * This component will restricts the user from
 * accessing private pages
 *
 * Only logged in customer can view Private pages
 *
 * @returns <PrivateRoute />
 */
const PrivateRoute = () => {
    const isLoggedIn = useIsLoggedIn();
    const currentPath = window.location.pathname;

    // Preload routes in tiers after user is authenticated
    usePreloadRoutes();

    if (!isLoggedIn && !publicPaths.includes(currentPath)) {
        // Storing current location path is to navigate the same page after
        // login success

        storeToLocalStorage(
            storageKeys.navPath,
            window.location.pathname + window.location.search
        );
    }
    return isLoggedIn ? <MainLayout /> : <Navigate to="/login" />;
};

export default PrivateRoute;
