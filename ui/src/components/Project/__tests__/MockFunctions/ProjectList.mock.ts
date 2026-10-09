// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ProjectDBEntry } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { AxiosResponse } from "types/common";

export const mockProjectsData: ProjectDBEntry[] = [
    {
        "timestamps": {
            "created_at": "2026-04-01T11:07:29.541702+00:00",
            "updated_at": "2026-04-01T11:15:11.817787+00:00"
        },
        "id": "project_f6716b2d5f474852adbd316d29cbfeff",
        "slug": "marketing",
        "group_id": "group_89b9d0e181f64758987e249f06285aec",
        "display_name": "Marketing",
        "created_by":
            "oidc|oidc-kumo-entraid|qw4gMmkz2KF6h2jXw4IBeaIv1MgbVLijEol5Kr4MeVE",
        "description": "Marketing project",
        "is_private": true,
        "status": "active"
    },
    {
        "timestamps": {
            "created_at": "2026-04-01T11:07:29.541702+00:00",
            "updated_at": "2026-04-01T11:15:11.817787+00:00"
        },
        "id": "project_adbd316d29cbfefff6716b2d5f474852",
        "slug": "sales",
        "group_id": "group_987e249f06285aec89b9d0e181f64758",
        "display_name": "Sales",
        "created_by":
            "oidc|oidc-kumo-entraid|MgbVLijEol5Kr4MeVEqw4gMmkz2KF6h2jXw4IBeaIv1",
        "description": "Sales project",
        "is_private": false,
        "status": "active"
    }
];

let projectsResponseType = "";

export const mockProjectsByGroupResponse = (): Promise<
    AxiosResponse<ProjectDBEntry[]>
> => {
    if (projectsResponseType === testResponseType.ERROR) {
        return Promise.reject(
            createMockResponse(new Error("Server Error"), 500)
        );
    }
    if (projectsResponseType === testResponseType.EMPTY) {
        return Promise.resolve(
            createMockResponse([]) as unknown as AxiosResponse<ProjectDBEntry[]>
        );
    }
    return Promise.resolve(
        createMockResponse(mockProjectsData) as unknown as AxiosResponse<
            ProjectDBEntry[]
        >
    );
};

export const setProjectsResponseType = (type: string = "") => {
    projectsResponseType = type;
};
