// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Navigate, Outlet } from "react-router";

import { useAdminInfo } from "hooks/auth/admin";

import { navRoutes } from "utils/Constants";

/**
 * Guard for the Admin route.
 *
 * Administration uses the same group/project model in every deployment.
 *
 * Only customer admins may render the admin section. Anyone else (including
 * an admin who just demoted themselves, once the AdminInfo query refetches)
 * is sent Home instead of being left on an admin page whose requests all 403.
 */
const AdminRouteGuard = () => {
    const { isAdmin, isLoading } = useAdminInfo();

    if (isLoading) return null;
    return isAdmin ? <Outlet /> : <Navigate to={navRoutes.Home} replace />;
};

export default AdminRouteGuard;
