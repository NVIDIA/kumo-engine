# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import json
from http import HTTPStatus
from pathlib import Path
from test.rest.integ.black_box.project_upload_helpers import upload_project_csv

import kumoapi.model_plan as plan_api
import pytest
from fastapi.testclient import TestClient

from io_utils.json import from_json_dict, to_json_dict

_PQUERY_RESOURCE_PATH = (Path(__file__).resolve().parents[1] / "test_data" /
                         "pquery_resource.json")
_FINANCIAL_DATA_PATH = (Path(__file__).resolve().parents[2] / 'csv_data' /
                        'FINANCIAL')


def _response_json(response):
    assert response.status_code == HTTPStatus.OK, response.text
    return response.json()


def _post_json(
    client: TestClient,
    path: str,
    payload: dict,
    project_id: str,
):
    return _response_json(
        client.post(path, params={'project_id': project_id}, json=payload))


def _get_json(client: TestClient, path: str, project_id: str,
              params: dict | None = None):
    request_params = dict(params or {})
    request_params['project_id'] = project_id
    return _response_json(client.get(path, params=request_params))


def _patch_tags(client: TestClient, path: str, project_id: str, params: dict):
    request_params = dict(params)
    request_params['project_id'] = project_id
    response = client.patch(path, params=request_params)
    body = _response_json(response)
    assert set(body) == {"resource_updated"}
    return body["resource_updated"]


def _patch_tags_error(
    client: TestClient,
    path: str,
    project_id: str,
    params: dict,
    expected_detail: str,
):
    request_params = dict(params)
    request_params['project_id'] = project_id
    response = client.patch(path, params=request_params)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert expected_detail in response.text


@pytest.fixture
def job_tags_project_id(state_services) -> str:
    async def ensure_project() -> str:
        _, project = await state_services.groups.ensure_default_resources()
        return str(project.id)

    return asyncio.run(ensure_project())


def _load_pquery_resource(
    client: TestClient,
    named_pquery: bool,
    project_id: str,
) -> tuple[dict, str]:
    pquery = json.loads(_PQUERY_RESOURCE_PATH.read_text(encoding="utf-8"))
    if not named_pquery:
        pquery["name"] = None
        pquery["desc"] = ""
    connector_id = None
    for table in pquery['graph']['tables'].values():
        source_table = table['source_table']
        source = source_table['source']
        source_path = _FINANCIAL_DATA_PATH / source['path']
        connector_id = upload_project_csv(
            client,
            project_id,
            source_path,
            api_prefix='/v1',
        )
        source['path'] = source_path.stem
    assert connector_id is not None
    for table in pquery['graph']['tables'].values():
        table['source_table']['connector_id'] = connector_id
    return pquery, project_id


def _create_graph_and_pquery(
    client: TestClient,
    pquery: dict,
    project_id: str,
) -> tuple[str, str]:
    graph = _post_json(client, "/v1/graphs", pquery["graph"], project_id)
    pquery_response = _post_json(
        client,
        "/v1/predictive_queries/?dialect=PQLANG_V2",
        pquery,
        project_id,
    )
    return graph["id"], pquery_response["id"]


def _create_generate_train_table_job(
    client: TestClient,
    pquery: dict,
    project_id: str,
) -> tuple[str, str]:
    graph_id, pquery_id = _create_graph_and_pquery(client, pquery, project_id)

    plan = _post_json(
        client,
        "/v1/predictive_queries/train_table_generation_plan?dialect=PQLANG_V2",
        {
            "query_string": pquery["query_string"],
            "graph_id": graph_id,
            "run_mode": "fast",
        },
        project_id,
    )
    job = _post_json(
        client,
        "/v1/gentraintable_jobs",
        {
            "job_tags": {
                "custom_tag_key": "custom value"
            },
            "pquery_id": pquery_id,
            "plan": plan,
            "graph_snapshot_id": None,
        },
        project_id,
    )
    return pquery_id, job["id"]


def _create_training_job(
    client: TestClient,
    pquery: dict,
    project_id: str,
) -> tuple[str, str]:
    graph_id, pquery_id = _create_graph_and_pquery(client, pquery, project_id)

    model_plan_info = from_json_dict(
        _post_json(
            client,
            "/v1/predictive_queries/model_plan?dialect=PQLANG_V2",
            {
                "query_string": pquery["query_string"],
                "graph_id": graph_id,
                "run_mode": "fast",
            },
            project_id,
        ),
        plan_api.ModelPlanInfo,
    )
    job = _post_json(
        client,
        "/v1/training_jobs",
        {
            "job_tags": {
                "custom_tag_key": "custom value"
            },
            "pquery_id": pquery_id,
            "model_plan": to_json_dict(model_plan_info.model_plan),
            "graph_snapshot_id": None,
        },
        project_id,
    )
    return pquery_id, job["id"]


def _expected_tags(named_pquery: bool, pquery_id: str) -> dict:
    expected_tags = {
        "custom_tag_key": "custom value",
        "pquery_id": pquery_id,
    }
    if named_pquery:
        expected_tags["pquery_name"] = "bpqv2"
    return expected_tags


def _assert_job_matches(
    job: dict,
    *,
    job_id: str,
    pquery_id: str,
    tags: dict,
) -> None:
    assert job["job_id"] == job_id
    assert job["config"]["pquery_id"] == pquery_id
    assert job["tags"] == tags
    assert "job_status_report" in job
    assert "status" in job["job_status_report"]


def _assert_job_tag_lifecycle(
    client: TestClient,
    *,
    job_path: str,
    job_id: str,
    pquery_id: str,
    expected_tags: dict,
    project_id: str,
) -> None:
    job = _get_json(client, f"{job_path}/{job_id}", project_id)
    _assert_job_matches(job, job_id=job_id, pquery_id=pquery_id,
                        tags=expected_tags)

    custom_tag_matches = _get_json(
        client,
        job_path,
        project_id,
        {"custom_tag_key": "custom value"},
    )
    assert len(custom_tag_matches) == 1
    _assert_job_matches(custom_tag_matches[0], job_id=job_id,
                        pquery_id=pquery_id, tags=expected_tags)

    all_tag_matches = _get_json(client, job_path, project_id, expected_tags)
    assert len(all_tag_matches) == 1
    _assert_job_matches(all_tag_matches[0], job_id=job_id, pquery_id=pquery_id,
                        tags=expected_tags)

    assert _get_json(
        client,
        job_path,
        project_id,
        {"custom_tag_key": "some other value"},
    ) == []
    assert _get_json(client, job_path, project_id,
                     {"pquery_id": "some_other_id"}) == []

    assert _patch_tags(
        client,
        f"{job_path}/{job_id}/tags",
        project_id,
        {"custom_tag_key": "new tag value"},
    ) is True
    updated_tags = copy.deepcopy(expected_tags)
    updated_tags["custom_tag_key"] = "new tag value"
    job = _get_json(client, f"{job_path}/{job_id}", project_id)
    _assert_job_matches(job, job_id=job_id, pquery_id=pquery_id,
                        tags=updated_tags)
    updated_matches = _get_json(client, job_path, project_id,
                                {"custom_tag_key": "new tag value"})
    assert len(updated_matches) == 1
    _assert_job_matches(updated_matches[0], job_id=job_id, pquery_id=pquery_id,
                        tags=updated_tags)
    assert _get_json(client, job_path, project_id,
                     {"custom_tag_key": "custom value"}) == []

    assert _patch_tags(
        client,
        f"{job_path}/{job_id}/tags",
        project_id,
        {"experiment_ver": "v1.2"},
    ) is True
    updated_tags["experiment_ver"] = "v1.2"
    job = _get_json(client, f"{job_path}/{job_id}", project_id)
    _assert_job_matches(job, job_id=job_id, pquery_id=pquery_id,
                        tags=updated_tags)
    experiment_matches = _get_json(client, job_path, project_id,
                                   {"experiment_ver": "v1.2"})
    assert len(experiment_matches) == 1
    _assert_job_matches(experiment_matches[0], job_id=job_id,
                        pquery_id=pquery_id, tags=updated_tags)

    assert _patch_tags(
        client,
        f"{job_path}/{job_id}/tags",
        project_id,
        {
            "experiment_ver": "null",
            "custom_tag_key": "none",
        },
    ) is True
    assert _patch_tags(
        client,
        f"{job_path}/{job_id}/tags",
        project_id,
        {"experiment_ver": "na"},
    ) is False
    assert _patch_tags(
        client,
        f"{job_path}/{job_id}/tags",
        project_id,
        {"custom_tag_key": "none"},
    ) is False
    deleted_tags = {
        key: value
        for key, value in updated_tags.items()
        if key not in {"custom_tag_key", "experiment_ver"}
    }
    job = _get_json(client, f"{job_path}/{job_id}", project_id)
    _assert_job_matches(job, job_id=job_id, pquery_id=pquery_id,
                        tags=deleted_tags)
    null_matches = _get_json(client, job_path, project_id,
                             {"experiment_ver": "null"})
    assert len(null_matches) == 1
    _assert_job_matches(null_matches[0], job_id=job_id, pquery_id=pquery_id,
                        tags=deleted_tags)
    assert _get_json(client, job_path, project_id,
                     {"experiment_ver": "v1.2"}) == []

    _patch_tags_error(
        client,
        f"{job_path}/{job_id}/tags",
        project_id,
        {"status": "my status"},
        "must not be a reserved keyword",
    )
    _patch_tags_error(
        client,
        f"{job_path}/{job_id}/tags",
        project_id,
        {"pquery_name": "new pquery name"},
        "must not be kumo predefined tag key",
    )


@pytest.mark.parametrize("named_pquery", [True, False])
def test_job_tags_gen_traintable(
    public_api_client: TestClient,
    named_pquery: bool,
    job_tags_project_id: str,
):
    pquery, project_id = _load_pquery_resource(
        public_api_client, named_pquery, job_tags_project_id)
    pquery_id, job_id = _create_generate_train_table_job(
        public_api_client, pquery, project_id)

    assert _post_json(
        public_api_client,
        f"/v1/gentraintable_jobs/{job_id}/cancel",
        {},
        project_id,
    ) == {}

    _assert_job_tag_lifecycle(
        public_api_client,
        job_path="/v1/gentraintable_jobs",
        job_id=job_id,
        pquery_id=pquery_id,
        expected_tags=_expected_tags(named_pquery, pquery_id),
        project_id=project_id,
    )


@pytest.mark.parametrize("named_pquery", [True, False])
def test_job_tags_training(
    public_api_client: TestClient,
    named_pquery: bool,
    job_tags_project_id: str,
):
    pquery, project_id = _load_pquery_resource(
        public_api_client, named_pquery, job_tags_project_id)
    pquery_id, job_id = _create_training_job(public_api_client, pquery,
                                             project_id)

    cancel_response = _post_json(
        public_api_client,
        f"/v1/training_jobs/{job_id}/cancel",
        {},
        project_id,
    )
    assert set(cancel_response) == {"is_cancelled"}
    assert isinstance(cancel_response["is_cancelled"], bool)

    _assert_job_tag_lifecycle(
        public_api_client,
        job_path="/v1/training_jobs",
        job_id=job_id,
        pquery_id=pquery_id,
        expected_tags=_expected_tags(named_pquery, pquery_id),
        project_id=project_id,
    )
