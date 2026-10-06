# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from test.rest.integ.black_box.project_upload_helpers import upload_project_csv

from fastapi.testclient import TestClient

# Converted from the empty-table validation branch in
# test/rest/integ/test_schema.py::test_table_refresh_validation. The
# successful table refresh path remains lower-level because it waits on
# Temporal/data-backend workflow results.
_SMALL_SYNTHETIC_DIR = (Path(__file__).resolve().parents[3] / "csv_data" /
                        "small_synthetic")
_DEFAULT_PROJECT_ID = "default"


def test_empty_table_validation_error_is_api_visible(client: TestClient):
    empty_table_path = _SMALL_SYNTHETIC_DIR / "empty.csv"
    connector_id = upload_project_csv(client, _DEFAULT_PROJECT_ID,
                                      empty_table_path)
    response = client.post(
        "/v1/tables/",
        params={
            "project_id": _DEFAULT_PROJECT_ID,
            "name_alias": "empty",
        },
        json={
            "cols": [
                {
                    "name": "A",
                    "stype": "ID",
                    "dtype": "int",
                },
                {
                    "name": "feature",
                    "stype": "categorical",
                    "dtype": "int",
                },
            ],
            "source_table": {
                "data_source_type": "S3",
                "connector_id": connector_id,
                "source_table_name": "empty",
                "file_type": "CSV",
            },
            "pkey":
            "A",
        },
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "Please check whether table was accidentally dropped" in response.text
