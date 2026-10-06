// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ProjectDBEntry } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

export const mockProjectData: ProjectDBEntry = {
    id: "project_abc123def456",
    slug: "test-project",
    group_id: "group-001",
    display_name: "Test Project",
    description: "A test project description",
    is_private: false,
    created_by: "user-001",
    status: "active",
    timestamps: {
        created_at: "2026-01-01T10:00:00.000000+00:00",
        updated_at: "2026-01-15T14:30:00.000000+00:00"
    }
};

let projectResponseType = "";

export const mockGetProjectResponse = (
    projectId: string = mockProjectData.id
) => {
    switch (projectResponseType) {
        case testResponseType.ERROR:
            return mockErrorResponse();
        case testResponseType.INVALID:
            return mockInvalidResponse();
        default:
            return Promise.resolve(
                createMockResponse({ ...mockProjectData, id: projectId })
            );
    }
};

export const setProjectResponseType = (type: string = "") => {
    projectResponseType = type;
};

let updateProjectResponseType = "";

export const mockUpdateProjectResponse = () => {
    switch (updateProjectResponseType) {
        case testResponseType.ERROR:
            return mockErrorResponse();
        case testResponseType.INVALID:
            return mockInvalidResponse();
        default:
            return Promise.resolve(createMockResponse(mockProjectData));
    }
};

export const setUpdateProjectResponseType = (type: string = "") => {
    updateProjectResponseType = type;
};

const mockErrorResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

const mockInvalidResponse = () => {
    return Promise.reject({
        response: {
            status: 422,
            statusText: "Unprocessable Entity",
            data: "Failed to update Project"
        }
    });
};
