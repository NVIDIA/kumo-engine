# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def _response_json(response):
    assert response.status_code in (200, 201), response.text
    return response.json()


def _create_group(client: TestClient, suffix: str) -> tuple[str, str]:
    group_display_name = f"phase1b_project_group_{suffix}"
    group = _response_json(
        client.post(
            "/v0/groups",
            json={
                "display_name": group_display_name,
                "description": "Phase 1b black-box project group",
            },
        ))
    assert group["display_name"] == group_display_name
    return group["id"], group_display_name


def test_project_lifecycle_is_api_visible(client: TestClient):
    suffix = uuid.uuid4().hex[:8]
    group_id, _ = _create_group(client, suffix)

    project_name = f"phase1b_project_{suffix}"
    validated_name = _response_json(
        client.post(
            "/v0/projects/utils/validate_name",
            params={"name": project_name},
        ))
    assert validated_name == {"is_valid": True, "error_msg": None}

    created_project = _response_json(
        client.post(
            f"/v0/groups/{group_id}/projects",
            json={
                "name": project_name,
                "description": "Phase 1b black-box project",
                "is_private": False,
            },
        ))
    project_id = created_project["id"]
    assert created_project["display_name"] == project_name
    assert created_project["group_id"] == group_id
    assert created_project["description"] == "Phase 1b black-box project"
    assert created_project["is_private"] is False
    assert created_project["status"] == "active"

    duplicate_name = _response_json(
        client.post(
            "/v0/projects/utils/validate_name",
            params={"name": project_name},
        ))
    assert duplicate_name["is_valid"] is False
    assert duplicate_name["error_msg"] == "Project name is not available."

    visible_projects = _response_json(client.get("/v0/projects/"))
    assert project_id in {project["id"] for project in visible_projects}

    projects_in_group = _response_json(
        client.get(f"/v0/groups/{group_id}/projects"))
    assert project_id in {project["id"] for project in projects_in_group}

    fetched_project = _response_json(client.get(f"/v0/projects/{project_id}"))
    assert fetched_project["id"] == project_id
    assert fetched_project["display_name"] == project_name

    updated_project_name = f"{project_name}_updated"
    updated_project = _response_json(
        client.patch(
            f"/v0/projects/{project_id}",
            json={
                "name": updated_project_name,
                "description": "Updated through the black-box API",
                "is_private": True,
            },
        ))
    assert updated_project["id"] == project_id
    assert updated_project["display_name"] == updated_project_name
    assert updated_project["description"] == "Updated through the black-box API"
    assert updated_project["is_private"] is True

    archived_project = _response_json(
        client.post(f"/v0/projects/{project_id}/archive"))
    assert archived_project["id"] == project_id
    assert archived_project["status"] == "archived"

    fetched_archived_project = _response_json(
        client.get(f"/v0/projects/{project_id}"))
    assert fetched_archived_project["id"] == project_id
    assert fetched_archived_project["status"] == "archived"
