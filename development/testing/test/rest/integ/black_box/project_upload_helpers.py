# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

from fastapi.testclient import TestClient


def project_upload_connector_id(
    client: TestClient,
    project_id: str,
    *,
    api_prefix: str = '/v0',
) -> str:
    response = client.get(
        f"{api_prefix}/projects/{project_id}/connectors/uploads/connector")
    assert response.status_code == HTTPStatus.OK, response.text
    return response.json()["connector_id"]


def upload_project_csv(
    client: TestClient,
    project_id: str,
    source_path: Path,
    *,
    api_prefix: str = '/v0',
) -> str:
    source_table_name = source_path.stem
    start = client.post(
        f"{api_prefix}/projects/{project_id}/connectors/uploads/start",
        json={
            "source_table_name": source_table_name,
            "file_type": "CSV",
            "num_parts": 1,
        },
    )
    assert start.status_code == HTTPStatus.OK, start.text
    upload = start.json()

    part = client.put(
        upload["presigned_part_urls"]["1"],
        content=source_path.read_bytes(),
    )
    assert part.status_code == HTTPStatus.OK, part.text

    complete = client.post(
        f"{api_prefix}/projects/{project_id}/connectors/uploads/complete",
        json={
            "source_table_name": source_table_name,
            "file_type": "CSV",
            "temp_upload_path": upload["temp_upload_path"],
            "upload_id": upload["upload_id"],
            "parts_metadata": [{
                "part_number": 1,
                "etag": part.json()["etag"],
            }],
            "validate_data": False,
        },
    )
    assert complete.status_code == HTTPStatus.OK, complete.text
    return project_upload_connector_id(
        client,
        project_id,
        api_prefix=api_prefix,
    )
