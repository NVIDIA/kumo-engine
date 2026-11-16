// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getProjectPath } from "utils/Constants";

export type ScopedRoutes = ReturnType<typeof getProjectPath>;

/**
 * Resolve resource route targets from explicit URL project context.
 */
export const getScopedRoutes = (projectId?: string | null): ScopedRoutes => {
    if (!projectId) {
        throw new Error("Project resource navigation requires projectId");
    }
    return getProjectPath(projectId);
};
