// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const projectQueryParams = (projectId?: string | null) => {
    return projectId ? { project_id: projectId } : undefined;
};
