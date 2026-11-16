# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def _response_json(response):
    assert response.status_code == 200, response.text
    return response.json()


def test_group_lifecycle_is_api_visible(client: TestClient, ):
    suffix = uuid.uuid4().hex[:8]

    group_display_name = f"phase1b_group_{suffix}"
    create_groups = client.post(
        "/v0/groups",
        json={
            "display_name": group_display_name,
            "description": "Phase 1b black-box group",
        },
    )
    assert create_groups.status_code == 201, create_groups.text
    group = create_groups.json()
    group_id = group["id"]
    assert group["display_name"] == group_display_name

    listed_groups = _response_json(client.get("/v0/groups"))
    assert group_id in {group["id"] for group in listed_groups}

    fetched_group = _response_json(client.get(f"/v0/groups/{group_id}"))
    assert fetched_group["id"] == group_id
    assert fetched_group["display_name"] == group_display_name
    assert fetched_group["members"] == []
    assert fetched_group["external_group_mappings"] == []

    member_response = client.post(
        f"/v0/groups/{group_id}/members",
        json={
            "principal_id": f"stable-user-{suffix}",
            "principal_type": "user",
        },
    )
    assert member_response.status_code == 201, member_response.text
    member = member_response.json()
    members = _response_json(client.get(f"/v0/groups/{group_id}/members"))
    assert [item["id"] for item in members] == [member["id"]]

    member_replacement = _response_json(
        client.put(
            f"/v1/groups/{group_id}/members",
            json={
                "members": [
                    {
                        "principal_id": f" stable-user-{suffix} ",
                        "principal_type": "user",
                    },
                    {
                        "principal_id": f"service-{suffix}",
                        "principal_type": "service_account",
                    },
                    {
                        "principal_id": f"stable-user-{suffix}",
                        "principal_type": "user",
                    },
                ]
            },
        ))
    assert [(item["principal_type"], item["principal_id"])
            for item in member_replacement["members"]] == [
                ("service_account", f"service-{suffix}"),
                ("user", f"stable-user-{suffix}"),
            ]
    assert next(item["id"] for item in member_replacement["members"]
                if item["principal_type"] == "user") == member["id"]

    external_group_id = f"phase1b_idp_{suffix}"
    mapping_response = client.post(
        f"/v0/groups/{group_id}/external-mappings",
        json={
            "identity_provider_id": "direct_connect",
            "external_group_id": external_group_id,
        },
    )
    assert mapping_response.status_code == 201, mapping_response.text
    mapping = mapping_response.json()
    assert mapping["group_id"] == group_id
    assert mapping["external_group_id"] == external_group_id

    mappings = _response_json(
        client.get(f"/v0/groups/{group_id}/external-mappings"))
    assert {item["id"] for item in mappings} == {mapping["id"]}

    group_detail = _response_json(client.get(f"/v0/groups/{group_id}"))
    assert [(item["principal_type"], item["principal_id"])
            for item in group_detail["members"]] == [
                ("service_account", f"service-{suffix}"),
                ("user", f"stable-user-{suffix}"),
            ]
    assert group_detail["external_group_mappings"] == [{
        "id": mapping["id"],
        "identity_provider_id": "direct_connect",
        "external_group_id": external_group_id,
    }]

    second_external_group_id = f"phase1b_idp_second_{suffix}"
    replacement = _response_json(
        client.put(
            f"/v1/groups/{group_id}/external-mappings",
            json={
                "mappings": [
                    {
                        "identity_provider_id": " DIRECT_CONNECT ",
                        "external_group_id": second_external_group_id,
                    },
                    {
                        "identity_provider_id": "direct_connect",
                        "external_group_id": external_group_id,
                    },
                ]
            },
        ))
    assert [item["external_group_id"]
            for item in replacement["mappings"]] == [
                external_group_id,
                second_external_group_id,
            ]
    assert next(item["id"] for item in replacement["mappings"]
                if item["external_group_id"] == external_group_id
                ) == mapping["id"]

    updated_group_name = f"{group_display_name}_updated"
    updated_group = _response_json(
        client.patch(
            f"/v0/groups/{group_id}",
            json={
                "display_name": updated_group_name,
                "description": "Updated through the black-box API",
            },
        ))
    assert updated_group["id"] == group_id
    assert updated_group["display_name"] == updated_group_name

    fetched_updated_group = _response_json(client.get(f"/v0/groups/{group_id}"))
    assert fetched_updated_group["id"] == group_id
    assert fetched_updated_group["display_name"] == updated_group_name

    delete_mapping = client.delete(
        f"/v0/groups/{group_id}/external-mappings/{mapping['id']}")
    assert delete_mapping.status_code == 204, delete_mapping.text

    delete_member = client.delete(
        f"/v0/groups/{group_id}/members/stable-user-{suffix}")
    assert delete_member.status_code == 204, delete_member.text

    delete_group = client.delete(f"/v0/groups/{group_id}")
    assert delete_group.status_code == 204, delete_group.text

    listed_after_delete = _response_json(client.get("/v0/groups"))
    assert group_id not in {group["id"] for group in listed_after_delete}
