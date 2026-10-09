# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import json
import time
import uuid
from http import HTTPStatus
from pathlib import Path
from test.rest.integ.fixture_connector_args import duplicate_fixture_sources
from test.rest.integ.public_api_smoke_helpers import (
    minimize_model_plan,
    provision_smoke_resource,
)
from typing import Any

import pytest
from fastapi.testclient import TestClient

# Phase 1b converted-owner coverage extracted from
# test/rest/integ/test_e2e_public_api.py::test_invalid_job_request and
# ::test_cancelled_generate_train_table_job_request_rejected, the
# generated-train-table frozen-pquery assertion from
# ::test_freeze_pquery_at_job_launch, and the generated-train-table
# list/status/readback branch plus aggregate /v0/jobs read/filter behavior from
# e2e job coverage. Keep this file raw HTTP/JSON only; the original owners
# retain broader workflow paths and storage-coupled assertions that are not part
# of this gate.
pytestmark = [
    pytest.mark.usefixtures("_pool_subprocesses"),
    pytest.mark.parametrize("storage_backend", ["local"], indirect=True),
]

_PQUERY_RESOURCE_PATH = (Path(__file__).resolve().parents[1] / "test_data" /
                         "pquery_resource.json")
_POLL_INTERVAL_SECONDS = 1
_MAX_POLL_SECONDS = 60


def _response_json(response: Any) -> Any:
    assert response.status_code == HTTPStatus.OK, response.text
    return response.json()


def _post_json(
    client: TestClient,
    path: str,
    payload: dict,
    *,
    params: dict | None = None,
) -> Any:
    return _response_json(
        client.post(
            path,
            json=payload,
            params=params or {},
            follow_redirects=True,
        ))


def _get_json(
    client: TestClient,
    path: str,
    params: dict | None = None,
) -> Any:
    return _response_json(client.get(path, params=params or {}))


def _post_ok(client: TestClient, path: str, payload: dict | None) -> None:
    response = client.post(path, json=payload, follow_redirects=True)
    assert response.status_code == HTTPStatus.OK, response.text


def _post_no_body_json(
    client: TestClient,
    path: str,
    *,
    params: dict | None = None,
) -> Any:
    return _response_json(
        client.post(path, params=params or {}, follow_redirects=True))


def _post_error(
    client: TestClient,
    path: str,
    payload: dict,
    expected_detail: str,
) -> None:
    response = client.post(path, json=payload, follow_redirects=True)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert expected_detail in response.text


def _load_pquery_resource() -> dict:
    pquery = json.loads(_PQUERY_RESOURCE_PATH.read_text(encoding="utf-8"))
    pquery["name"] = f"{pquery['name']}_{uuid.uuid4().hex[:10]}"
    return pquery


def _status(resource: dict) -> str:
    status = resource["job_status_report"]["status"]
    return str(status).split(".")[-1]


def _enum_name(value: Any) -> str:
    return str(value).split(".")[-1]


def _id_from_response(body: Any) -> str:
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        value = body.get("id")
        assert value, body
        return value
    raise AssertionError(body)


def _extract_model_plan(model_plan_response: dict) -> dict:
    if "model_plan" in model_plan_response:
        return model_plan_response["model_plan"]
    return model_plan_response["suggested_model_plan"]["model_plan"]


def _active_job_status_name(value: Any) -> str:
    if isinstance(value, str):
        return _enum_name(value)
    return _enum_name(value["metadata"]["status"])


def _wait_for_job_status(
    client: TestClient,
    path: str,
    expected_status: str,
) -> dict:
    deadline = time.monotonic() + _MAX_POLL_SECONDS
    last_resource: dict | None = None
    while time.monotonic() < deadline:
        last_resource = _get_json(client, path)
        if _status(last_resource) == expected_status:
            return last_resource
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"{path} did not reach {expected_status} in "
                       f"{_MAX_POLL_SECONDS}s; last_resource={last_resource}")


def _create_pquery_and_plans(client: TestClient) -> tuple[str, dict, dict]:
    pquery = _load_pquery_resource()
    graph = _post_json(client, "/v1/graphs", pquery["graph"])
    pquery_response = _post_json(
        client,
        "/v1/predictive_queries/?dialect=PQLANG_V2",
        pquery,
    )
    pquery_id = pquery_response["id"]

    plan_request = {
        "query_string": pquery["query_string"],
        "graph_id": graph["id"],
        "run_mode": "fast",
    }
    train_table_plan = _post_json(
        client,
        "/v1/predictive_queries/train_table_generation_plan?dialect=PQLANG_V2",
        plan_request,
    )
    model_plan = _post_json(
        client,
        "/v1/predictive_queries/model_plan?dialect=PQLANG_V2",
        plan_request,
    )["model_plan"]
    assert "neighbor_sampling" in model_plan
    assert "num_neighbors" in model_plan["neighbor_sampling"]
    return pquery_id, train_table_plan, model_plan


def _graph_with_duplicate_sources(
    graph: dict,
    duplicate_paths: dict[str, str],
) -> dict:
    modified_graph = copy.deepcopy(graph)
    for table in modified_graph["tables"].values():
        source = table["source_table"]["source"]
        source["path"] = duplicate_paths[source["path"]]
    return modified_graph


def _assert_frozen_pquery_matches_graph(
    client: TestClient,
    frozen_pquery_id: str,
    *,
    original_pquery_name: str,
    expected_query_string: str,
    modified_graph: dict,
    original_graph: dict,
) -> dict:
    assert frozen_pquery_id != original_pquery_name
    assert frozen_pquery_id.startswith("pquery-")
    frozen_pquery = _get_json(client,
                              f"/v1/predictive_queries/{frozen_pquery_id}")
    assert frozen_pquery["query_string"] == expected_query_string
    assert frozen_pquery["graph"] == modified_graph
    assert frozen_pquery["graph"] != original_graph
    return frozen_pquery


def _create_smoke_generate_train_table_job(
    client: TestClient,
    *,
    job_tags: dict | None = None,
) -> tuple[str, str]:
    _, _, graph, pquery = provision_smoke_resource(client)

    pquery_name = pquery["name"]
    assert pquery_name
    _post_json(
        client,
        "/v1/predictive_queries/",
        pquery,
        params={"dialect": "PQLANG_V2"},
    )
    graph_id = _post_json(client, "/v1/graphs", graph)["id"]
    train_table_plan = _post_json(
        client,
        "/v1/predictive_queries/train_table_generation_plan",
        {
            "query_string": pquery["query_string"],
            "graph_id": graph_id,
            "run_mode": "fast",
        },
        params={"dialect": "PQLANG_V2"},
    )
    train_table_job_id = _post_json(
        client,
        "/v1/gentraintable_jobs",
        {
            "job_tags": job_tags or {},
            "pquery_id": pquery_name,
            "plan": train_table_plan,
            "graph_snapshot_id": None,
        },
    )["id"]
    return pquery_name, train_table_job_id


def _assert_generate_train_table_job_readback(
    job: dict,
    *,
    job_id: str,
    pquery_name: str,
    expected_status: str,
    expected_custom_tags: dict | None = None,
) -> None:
    assert job["job_id"] == job_id
    assert _status(job) == expected_status
    assert job["config"]["pquery_id"].startswith("pquery-")
    assert job["tags"]["pquery_name"] == pquery_name
    assert job["tags"]["pquery_id"].startswith("pquery-")
    for key, value in (expected_custom_tags or {}).items():
        assert job["tags"][key] == value


def _aggregate_jobs(client: TestClient, payload: dict) -> dict:
    response = _post_json(client, "/v0/jobs/", payload)
    assert set(response) == {"jobs", "total_num_items"}
    assert isinstance(response["jobs"], list)
    assert isinstance(response["total_num_items"], int)
    return response


def _aggregate_job_ids(response: dict) -> list[str]:
    return [job["job_id"] for job in response["jobs"]]


def _assert_aggregate_generate_train_table_job(
    job: dict,
    *,
    job_id: str,
    pquery_name: str,
    expected_status: str,
    expected_custom_tags: dict,
) -> None:
    assert job["job_id"] == job_id
    assert _enum_name(job["job_type"]) == "GENERATE_TRAIN_TABLE_JOB"
    metadata = job["metadata"]
    assert _enum_name(metadata["job_type"]) == "GENERATE_TRAIN_TABLE_JOB"
    assert _enum_name(metadata["status"]) == expected_status
    assert metadata["tags"]["pquery_name"] == pquery_name
    assert metadata["tags"]["pquery_id"].startswith("pquery-")
    for key, value in expected_custom_tags.items():
        assert metadata["tags"][key] == value


def test_frozen_pquery_at_generate_train_table_launch_is_api_visible(
    public_api_client: TestClient, ):
    fixture, connector, graph, pquery = provision_smoke_resource(
        public_api_client)
    duplicate_paths = asyncio.run(duplicate_fixture_sources(
        connector, fixture))

    pquery_name = pquery["name"]
    assert pquery_name
    _post_json(
        public_api_client,
        "/v1/predictive_queries/",
        pquery,
        params={"dialect": "PQLANG_V2"},
    )
    original_pquery = _get_json(public_api_client,
                                f"/v1/predictive_queries/{pquery_name}")

    original_graph_id = _post_json(public_api_client, "/v1/graphs", graph)["id"]
    modified_graph = _graph_with_duplicate_sources(graph, duplicate_paths)
    modified_graph_id = _post_json(public_api_client, "/v1/graphs",
                                   modified_graph)["id"]
    modified_graph_resource = _get_json(public_api_client,
                                        f"/v1/graphs/{modified_graph_id}")
    snapshot_id = _post_json(
        public_api_client,
        "/v1/graphsnapshots",
        {},
        params={"graph_id": modified_graph_id},
    )["id"]

    plan_request = {
        "query_string": pquery["query_string"],
        "graph_id": original_graph_id,
        "run_mode": "fast",
    }
    train_table_plan = _post_json(
        public_api_client,
        "/v1/predictive_queries/train_table_generation_plan",
        plan_request,
        params={"dialect": "PQLANG_V2"},
    )
    train_table_job_id = _post_json(
        public_api_client,
        "/v1/gentraintable_jobs",
        {
            "job_tags": {},
            "pquery_id": pquery_name,
            "plan": train_table_plan,
            "graph_snapshot_id": snapshot_id,
        },
    )["id"]
    train_table_job = _wait_for_job_status(
        public_api_client,
        f"/v1/gentraintable_jobs/{train_table_job_id}",
        "DONE",
    )
    _assert_frozen_pquery_matches_graph(
        public_api_client,
        train_table_job["config"]["pquery_id"],
        original_pquery_name=pquery_name,
        expected_query_string=pquery["query_string"],
        modified_graph=modified_graph_resource["graph"],
        original_graph=original_pquery["graph"],
    )


def test_generate_train_table_job_list_status_readback_is_api_visible(
    public_api_client: TestClient, ):
    custom_tags = {"phase1b_slice": "job-list-readback"}
    pquery_name, train_table_job_id = _create_smoke_generate_train_table_job(
        public_api_client,
        job_tags=custom_tags,
    )
    train_table_job = _wait_for_job_status(
        public_api_client,
        f"/v1/gentraintable_jobs/{train_table_job_id}",
        "DONE",
    )
    _assert_generate_train_table_job_readback(
        train_table_job,
        job_id=train_table_job_id,
        pquery_name=pquery_name,
        expected_status="DONE",
        expected_custom_tags=custom_tags,
    )

    listed_jobs = _get_json(
        public_api_client,
        "/v1/gentraintable_jobs",
        {"pquery_name": pquery_name},
    )
    assert [job["job_id"] for job in listed_jobs] == [train_table_job_id]
    _assert_generate_train_table_job_readback(
        listed_jobs[0],
        job_id=train_table_job_id,
        pquery_name=pquery_name,
        expected_status="DONE",
        expected_custom_tags=custom_tags,
    )

    done_jobs = _get_json(
        public_api_client,
        "/v1/gentraintable_jobs",
        {
            "pquery_name": pquery_name,
            "job_status": "DONE",
            "limit": "1",
        },
    )
    assert [job["job_id"] for job in done_jobs] == [train_table_job_id]
    assert _get_json(
        public_api_client,
        "/v1/gentraintable_jobs",
        {
            "pquery_name": pquery_name,
            "job_status": "RUNNING",
        },
    ) == []


def test_aggregate_generate_train_table_jobs_sort_page_archive_is_api_visible(
    client: TestClient,
    public_api_client: TestClient,
):
    slice_tag = f"aggregate-sort-page-archive-{uuid.uuid4().hex[:10]}"
    first_tags = {"phase1b_slice": slice_tag, "phase1b_order": "first"}
    second_tags = {"phase1b_slice": slice_tag, "phase1b_order": "second"}

    first_pquery_name, first_job_id = _create_smoke_generate_train_table_job(
        public_api_client,
        job_tags=first_tags,
    )
    _wait_for_job_status(
        public_api_client,
        f"/v1/gentraintable_jobs/{first_job_id}",
        "DONE",
    )

    second_pquery_name, second_job_id = _create_smoke_generate_train_table_job(
        public_api_client,
        job_tags=second_tags,
    )
    _wait_for_job_status(
        public_api_client,
        f"/v1/gentraintable_jobs/{second_job_id}",
        "DONE",
    )

    base_payload = {
        "job_type": ["GENERATE_TRAIN_TABLE_JOB"],
        "status": ["DONE"],
        "tags_filter": [slice_tag],
        "sort_by": "metadata.start_time",
    }
    assert _aggregate_jobs(client, {
        **base_payload,
        "status": ["RUNNING"],
    }) == {"jobs": [], "total_num_items": 0}
    ascending_jobs = _aggregate_jobs(
        client,
        {
            **base_payload,
            "sort_desc": False,
            "limit": 2,
            "page": 0,
        },
    )
    descending_jobs = _aggregate_jobs(
        client,
        {
            **base_payload,
            "sort_desc": True,
            "limit": 2,
            "page": 0,
        },
    )
    expected_ids = {first_job_id, second_job_id}
    assert ascending_jobs["total_num_items"] == 2
    assert descending_jobs["total_num_items"] == 2
    assert set(_aggregate_job_ids(ascending_jobs)) == expected_ids
    assert _aggregate_job_ids(descending_jobs) == list(
        reversed(_aggregate_job_ids(ascending_jobs)))

    desc_page_0 = _aggregate_jobs(
        client,
        {
            **base_payload,
            "sort_desc": True,
            "limit": 1,
            "page": 0,
        },
    )
    desc_page_1 = _aggregate_jobs(
        client,
        {
            **base_payload,
            "sort_desc": True,
            "limit": 1,
            "page": 1,
        },
    )
    assert desc_page_0["total_num_items"] == 2
    assert desc_page_1["total_num_items"] == 2
    assert len(desc_page_0["jobs"]) == 1
    assert len(desc_page_1["jobs"]) == 1
    assert _aggregate_job_ids(desc_page_0) == _aggregate_job_ids(
        descending_jobs)[:1]
    assert _aggregate_job_ids(desc_page_1) == _aggregate_job_ids(
        descending_jobs)[1:]
    assert set(_aggregate_job_ids(desc_page_0)).isdisjoint(
        _aggregate_job_ids(desc_page_1))

    expected_pqueries = {
        first_job_id: first_pquery_name,
        second_job_id: second_pquery_name,
    }
    expected_tags = {
        first_job_id: first_tags,
        second_job_id: second_tags,
    }
    for job in descending_jobs["jobs"]:
        _assert_aggregate_generate_train_table_job(
            job,
            job_id=job["job_id"],
            pquery_name=expected_pqueries[job["job_id"]],
            expected_status="DONE",
            expected_custom_tags=expected_tags[job["job_id"]],
        )

    archived_job_id = _aggregate_job_ids(desc_page_0)[0]
    unarchived_job_id = (expected_ids - {archived_job_id}).pop()
    assert _post_no_body_json(
        client,
        f"/v0/jobs/{archived_job_id}/archive",
        params={"archived": "true"},
    ) is True

    default_after_archive = _aggregate_jobs(
        client,
        {
            **base_payload,
            "sort_desc": True,
            "limit": 2,
            "page": 0,
        },
    )
    assert default_after_archive["total_num_items"] == 1
    assert _aggregate_job_ids(default_after_archive) == [unarchived_job_id]

    archived_view = _aggregate_jobs(
        client,
        {
            **base_payload,
            "tags_filter": [slice_tag, "archived"],
            "search_keyword": archived_job_id,
            "sort_desc": True,
            "limit": 2,
            "page": 0,
        },
    )
    assert archived_view["total_num_items"] == 1
    assert _aggregate_job_ids(archived_view) == [archived_job_id]

    assert _post_no_body_json(
        client,
        f"/v0/jobs/{archived_job_id}/archive",
        params={"archived": "false"},
    ) is True
    default_after_unarchive = _aggregate_jobs(
        client,
        {
            **base_payload,
            "sort_desc": True,
            "limit": 2,
            "page": 0,
        },
    )
    assert default_after_unarchive["total_num_items"] == 2
    assert set(_aggregate_job_ids(default_after_unarchive)) == expected_ids

    archived_view_after_unarchive = _aggregate_jobs(
        client,
        {
            **base_payload,
            "tags_filter": [slice_tag, "archived"],
            "search_keyword": archived_job_id,
            "limit": 2,
            "page": 0,
        },
    )
    assert archived_view_after_unarchive == {"jobs": [], "total_num_items": 0}


def test_active_jobs_returns_status_only_for_known_training_job(
    client: TestClient, ):
    assert _post_no_body_json(client, "/v0/jobs/get_active_jobs") == {}

    _, _, graph, pquery = provision_smoke_resource(client)
    graph_id = _post_json(client, "/v1/graphs", graph)["id"]

    model_plan_response = _post_json(
        client,
        "/v1/predictive_queries/model_plan",
        {
            "query_string": pquery["query_string"],
            "graph_id": graph_id,
            "graph_definition": graph,
            "run_mode": "debug",
        },
        params={"dialect": "PQLANG_V2"},
    )
    model_plan = minimize_model_plan(_extract_model_plan(model_plan_response))
    training_job_id = _id_from_response(
        _post_json(
            client,
            "/v0/training_jobs/train",
            {
                "job_tags": {
                    "phase1b_slice": "active-jobs"
                },
                "predictive_query": pquery["query_string"],
                "graph_definition": graph,
                "training_table_gen_plan":
                model_plan["training_table_generation"],
                "model_plan": model_plan,
            },
        ))
    assert training_job_id.startswith("trainingjob")

    active_jobs = _post_no_body_json(client, "/v0/jobs/get_active_jobs")
    generate_train_table_job_id = next(
        job_id for job_id in active_jobs
        if job_id.startswith("gen-traintable-job"))
    assert set(active_jobs) == {generate_train_table_job_id, training_job_id}
    assert active_jobs[training_job_id]["job_id"] == training_job_id
    assert (_enum_name(
        active_jobs[training_job_id]["job_type"]) == "TRAINING_JOB")
    assert (active_jobs[generate_train_table_job_id]["job_id"] ==
            generate_train_table_job_id)
    assert (_enum_name(active_jobs[generate_train_table_job_id]["job_type"]) ==
            "GENERATE_TRAIN_TABLE_JOB")
    assert {_active_job_status_name(job)
            for job in active_jobs.values()}.issubset({"QUEUED", "RUNNING"})

    active_jobs_with_known_training = _response_json(
        client.post("/v0/jobs/get_active_jobs", json=[training_job_id]))
    assert set(active_jobs_with_known_training) == {
        generate_train_table_job_id,
        training_job_id,
    }
    assert _enum_name(active_jobs_with_known_training[training_job_id]) in {
        "QUEUED",
        "RUNNING",
    }
    assert isinstance(
        active_jobs_with_known_training[generate_train_table_job_id],
        dict,
    )
    assert (active_jobs_with_known_training[generate_train_table_job_id]
            ["job_id"] == generate_train_table_job_id)
    assert (_active_job_status_name(
        active_jobs_with_known_training[generate_train_table_job_id])
            in {"QUEUED", "RUNNING"})


def test_public_api_job_request_validation_is_api_visible(
    public_api_client: TestClient, ):
    pquery_id, train_table_plan, model_plan = _create_pquery_and_plans(
        public_api_client)

    bad_train_table_plan = copy.deepcopy(train_table_plan)
    bad_train_table_plan["split"] = "invalid_split!!!"
    _post_error(
        public_api_client,
        "/v1/gentraintable_jobs",
        {
            "job_tags": {},
            "pquery_id": pquery_id,
            "plan": bad_train_table_plan,
            "graph_snapshot_id": None,
        },
        "is not a valid split",
    )
    _post_error(
        public_api_client,
        "/v1/gentraintable_jobs",
        {
            "job_tags": {},
            "pquery_id": pquery_id,
            "plan": train_table_plan,
            "graph_snapshot_id": "bad_graph_snapshot_id",
        },
        "GRAPH snapshot ID is invalid",
    )

    bad_model_plan = copy.deepcopy(model_plan)
    bad_model_plan["training_job"]["tune_metric"] = "XYZ"
    _post_error(
        public_api_client,
        "/v1/training_jobs",
        {
            "job_tags": {},
            "pquery_id": pquery_id,
            "model_plan": bad_model_plan,
        },
        "The tune metric 'XYZ' is not present in the list of metrics",
    )
    _post_error(
        public_api_client,
        "/v1/training_jobs",
        {
            "job_tags": {},
            "pquery_id": pquery_id,
            "model_plan": model_plan,
            "graph_snapshot_id": None,
            "train_table_job_id": "bad_train_table_job_id",
        },
        "Generate train table job is not found",
    )
    _post_error(
        public_api_client,
        "/v1/training_jobs",
        {
            "job_tags": {},
            "pquery_id": pquery_id,
            "model_plan": model_plan,
            "graph_snapshot_id": "bad_graph_snapshot_id",
        },
        "GRAPH snapshot ID is invalid",
    )


def test_cancelled_generate_train_table_job_rejects_training_request(
    public_api_client: TestClient, ):
    pquery_id, train_table_plan, model_plan = _create_pquery_and_plans(
        public_api_client)
    train_table_job_id = _post_json(
        public_api_client,
        "/v1/gentraintable_jobs",
        {
            "job_tags": {},
            "pquery_id": pquery_id,
            "plan": train_table_plan,
            "graph_snapshot_id": None,
        },
    )["id"]

    _post_ok(
        public_api_client,
        f"/v1/gentraintable_jobs/{train_table_job_id}/cancel",
        {},
    )
    _wait_for_job_status(
        public_api_client,
        f"/v1/gentraintable_jobs/{train_table_job_id}",
        "CANCELLED",
    )

    _post_error(
        public_api_client,
        "/v1/training_jobs",
        {
            "job_tags": {},
            "pquery_id": pquery_id,
            "model_plan": model_plan,
            "graph_snapshot_id": None,
            "train_table_job_id": train_table_job_id,
        },
        "CANCELLED generate-train-table job",
    )
