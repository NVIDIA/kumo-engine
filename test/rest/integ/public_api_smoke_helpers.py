# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import io
import os
import time
from http import HTTPStatus
from pathlib import Path
from test.rest.integ.fixture_connector_args import provision_fixture_connectors
from test.rest.integ.public_response_reader import (
    flatten_public_parquet_refs,
    read_public_response_bytes,
)
from typing import Any

from io_utils.json import load_json, to_json_dict
from kumo.testing.rest import randomize_fixture_ids_v2
from rest.config.api_command import FixtureDefinition

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (REPO_ROOT / "test/rest/integ/test_cases/test_e2e_fixtures.json")
PQUERY_CASE = "saas_temporal_regression_base"
EXPECTED_GRAPH_TABLES = {"ACCOUNT", "LOAN", "TRANS"}
EXPECTED_EDGE_TOTAL_COUNTS = {
    "LOAN.ACCOUNT_ID.ACCOUNT": 682,
    "TRANS.ACCOUNT_ID.ACCOUNT": 5000,
}

POLL_INTERVAL_SECONDS = float(
    os.environ.get("PHASE1B_POLL_INTERVAL_SECONDS", "5"))
MAX_POLL_SECONDS = 1200
PRED_TABLE_ANCHOR_TIME_NS = 1609804800000000000
TERMINAL_STAGE_STATUSES = {"DONE", "FAILED", "CANCELLED"}
SMOKE_JOB_TAGS = {
    "train_table": {
        "phase1b_smoke": "train-table"
    },
    "training": {
        "phase1b_smoke": "training"
    },
    "prediction_table": {
        "phase1b_smoke": "prediction-table"
    },
    "batch_prediction": {
        "phase1b_smoke": "batch-prediction"
    },
}


def _response_json(response: Any) -> Any:
    assert response.status_code == HTTPStatus.OK, response.text
    return response.json()


def _post_json(
    client: Any,
    path: str,
    payload: dict,
    *,
    params: dict | None = None,
) -> Any:
    return _response_json(client.post(path, params=params or {}, json=payload))


def _get_json(client: Any, path: str, *, params: dict | None = None) -> Any:
    return _response_json(client.get(path, params=params or {}))


def _is_transient_gateway_status(status_code: int) -> bool:
    return status_code in {
        HTTPStatus.BAD_GATEWAY,
        HTTPStatus.SERVICE_UNAVAILABLE,
        HTTPStatus.GATEWAY_TIMEOUT,
    }


def _get_json_or_transient_gateway(
    client: Any,
    path: str,
    *,
    params: dict | None = None,
) -> tuple[Any | None, Any]:
    response = client.get(path, params=params or {})
    if _is_transient_gateway_status(response.status_code):
        return None, response
    return _response_json(response), response


def _status_name(status: Any) -> str:
    return str(status).split(".")[-1]


def _job_status(resource: dict) -> str:
    return _status_name(resource["job_status_report"]["status"])


def _snapshot_stage_statuses(resource: dict) -> list[str]:
    return [
        _status_name(stage["status"])
        for stage in resource.get("stages", {}).values()
    ]


def _stage_statuses(resource: dict) -> dict[str, str]:
    return {
        str(stage_name).split(".")[-1]: _status_name(stage["status"])
        for stage_name, stage in resource.get("stages", {}).items()
    }


def provision_smoke_resource(
    client: Any,
    fixture_root: Path | None = None,
) -> tuple[FixtureDefinition, Any, dict, dict]:
    """Provision the public smoke graph through the project upload boundary."""
    loaded = load_json(str(FIXTURE_PATH), FixtureDefinition)
    query = next(query for query in loaded.queries
                 if query.pquery_name == PQUERY_CASE)
    graph = next(graph for graph in loaded.graphs
                 if graph.fpid == query.graph_fpid)
    fixture = randomize_fixture_ids_v2(
        FixtureDefinition(tables=[], graphs=[graph], queries=[query]))
    _, connectors = asyncio.run(
        provision_fixture_connectors(
            client,
            fixture,
            api_prefix='/v1',
            fixture_root=fixture_root,
        ))
    graph_payload = to_json_dict(fixture.graphs[0])
    query = fixture.queries[0]
    assert set(graph_payload["tables"]) == EXPECTED_GRAPH_TABLES
    pquery = {
        "name": query.pquery_name,
        "desc": query.description,
        "query_string": query.query_str,
        "graph": graph_payload,
    }
    return fixture, connectors[0], graph_payload, pquery


def _wait_for_job_done(
    client: Any,
    path: str,
    *,
    label: str,
) -> dict:
    deadline = time.monotonic() + MAX_POLL_SECONDS
    last_resource: dict | None = None
    transient_error_count = 0
    while time.monotonic() < deadline:
        response = client.get(path, params={})
        if _is_transient_gateway_status(response.status_code):
            transient_error_count += 1
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        last_resource = _response_json(response)
        status = _job_status(last_resource)
        if status in {"DONE", "FAILED", "CANCELLED"}:
            assert status == "DONE", f"{label} ended in {status}: {last_resource}"
            return last_resource
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"{label} did not reach DONE in {MAX_POLL_SECONDS}s; "
                       f"last_resource={last_resource}; "
                       f"transient_errors={transient_error_count}")


def _wait_for_graph_snapshot_done(client: Any, snapshot_id: str) -> dict:
    deadline = time.monotonic() + MAX_POLL_SECONDS
    last_resource: dict | None = None
    while time.monotonic() < deadline:
        body, response = _get_json_or_transient_gateway(
            client,
            f"/v1/graphsnapshots/{snapshot_id}",
        )
        if body is None:
            last_resource = {
                "transient_body": response.text[:200],
                "transient_status": response.status_code,
            }
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        last_resource = body
        statuses = _snapshot_stage_statuses(last_resource)
        if statuses and all(status in TERMINAL_STAGE_STATUSES
                            for status in statuses):
            assert all(status == "DONE" for status in statuses), last_resource
            return last_resource
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"graph snapshot {snapshot_id} did not reach DONE in "
                       f"{MAX_POLL_SECONDS}s; last_resource={last_resource}")


def _wait_for_table_snapshot_stats_done(
    client: Any,
    table_snapshot_id: str,
    *,
    table_name: str,
) -> dict:
    deadline = time.monotonic() + MAX_POLL_SECONDS
    last_resource: dict | None = None
    while time.monotonic() < deadline:
        body, response = _get_json_or_transient_gateway(
            client,
            f"/v1/tablesnapshots/{table_snapshot_id}",
        )
        if body is None:
            last_resource = {
                "transient_body": response.text[:200],
                "transient_status": response.status_code,
            }
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        last_resource = body
        statuses = _stage_statuses(last_resource)
        required = {
            "INGEST": statuses.get("INGEST"),
            "FULL_COL_STATS": statuses.get("FULL_COL_STATS"),
        }
        if all(status in TERMINAL_STAGE_STATUSES
               for status in required.values()):
            assert required == {
                "INGEST": "DONE",
                "FULL_COL_STATS": "DONE",
            }, (table_name, table_snapshot_id, last_resource)
            return last_resource
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"table snapshot {table_name}/{table_snapshot_id} did not reach "
        f"INGEST and FULL_COL_STATS DONE in {MAX_POLL_SECONDS}s; "
        f"last_resource={last_resource}")


def _normalize_edge_name(edge_name: str) -> str:
    return edge_name.replace("__", ".")


def _verify_edge_health_readback(client: Any,
                                 snapshot_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + MAX_POLL_SECONDS
    edge_health: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        body, response = _get_json_or_transient_gateway(
            client,
            f"/v1/graphsnapshots/edge_health/{snapshot_id}",
        )
        if body is None:
            edge_health = {
                "transient_body": response.text[:200],
                "transient_status": response.status_code,
            }
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        edge_health = body
        if edge_health["is_ready"] is True:
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    assert edge_health is not None, snapshot_id
    assert edge_health["is_ready"] is True, edge_health
    statistics = edge_health.get("statistics")
    assert isinstance(statistics, dict), edge_health
    normalized_statistics = {
        _normalize_edge_name(edge_name): edge_stats
        for edge_name, edge_stats in statistics.items()
    }
    assert set(normalized_statistics) == set(EXPECTED_EDGE_TOTAL_COUNTS), (
        sorted(normalized_statistics),
        sorted(EXPECTED_EDGE_TOTAL_COUNTS),
    )
    for edge_name, expected_total in EXPECTED_EDGE_TOTAL_COUNTS.items():
        actual_total = normalized_statistics[edge_name]["total_num_edges"]
        assert actual_total == expected_total, (
            edge_name,
            actual_total,
            expected_total,
        )
    return {
        "edge_count": len(statistics),
        "edge_names": sorted(normalized_statistics),
        "is_ready": edge_health["is_ready"],
        "snapshot_id": snapshot_id,
    }


def _verify_snapshot_stats_readback(
    client: Any,
    snapshot_id: str,
    snapshot: dict,
) -> dict[str, Any]:
    table_ids = snapshot["table_ids"]
    assert set(table_ids) == EXPECTED_GRAPH_TABLES, snapshot

    table_snapshots = {}
    for table_name in sorted(table_ids):
        table_snapshot_id = table_ids[table_name]
        table_snapshot = _wait_for_table_snapshot_stats_done(
            client,
            table_snapshot_id,
            table_name=table_name,
        )
        column_stats = table_snapshot.get("column_stats")
        assert isinstance(column_stats, list) and column_stats, table_snapshot
        table_snapshots[table_name] = {
            "column_count": len(column_stats),
            "stage_statuses": _stage_statuses(table_snapshot),
            "table_snapshot_id": table_snapshot_id,
        }

    return {
        "edge_health": _verify_edge_health_readback(client, snapshot_id),
        "snapshot_id": snapshot_id,
        "snapshot_stage_statuses": _snapshot_stage_statuses(snapshot),
        "table_ids": sorted(table_ids),
        "table_snapshots": table_snapshots,
    }


def _create_graph_snapshot(client: Any,
                           graph_id: str) -> tuple[str, dict, dict]:
    snapshot_response = _post_json(
        client,
        "/v1/graphsnapshots",
        {},
        params={
            "graph_id": graph_id,
            "refresh_source": "true"
        },
    )
    snapshot_id = snapshot_response["id"]
    snapshot = _wait_for_graph_snapshot_done(client, snapshot_id)
    assert set(snapshot["table_ids"]) == EXPECTED_GRAPH_TABLES
    snapshot_stats = _verify_snapshot_stats_readback(
        client,
        snapshot_id,
        snapshot,
    )
    return snapshot_id, snapshot, snapshot_stats


def _verify_train_table_output_readback(
    client: Any,
    train_table_job_id: str,
) -> dict[str, Any]:
    table_data_refs = _get_json(
        client,
        f"/v1/gentraintable_jobs/{train_table_job_id}/table_data",
    )
    assert isinstance(table_data_refs,
                      list) and table_data_refs, table_data_refs
    assert all(isinstance(item, str) and item
               for item in table_data_refs), table_data_refs
    parquet_refs = flatten_public_parquet_refs(table_data_refs)
    assert parquet_refs, table_data_refs

    import pyarrow.parquet as pq

    schemas = []
    total_rows = 0
    for parquet_ref in parquet_refs:
        data = read_public_response_bytes(parquet_ref)
        assert data, f'Public data reference was empty: {parquet_ref}'
        parquet_file = pq.ParquetFile(io.BytesIO(data))
        schemas.append(parquet_file.schema_arrow)
        total_rows += parquet_file.metadata.num_rows

    assert all(schema == schemas[0] for schema in schemas), schemas
    assert total_rows > 0, 'Public train-table response contained no rows'

    split_masks = _get_json(
        client,
        f"/v1/gentraintable_jobs/{train_table_job_id}/split_masks",
    )
    assert isinstance(split_masks, dict), split_masks
    assert set(split_masks) == {"train", "val", "test"}, split_masks
    assert all(
        isinstance(value, str) and value
        for value in split_masks.values()), split_masks

    return {
        "train_table_data_ref_count": len(parquet_refs),
        "train_table_row_count": total_rows,
        "train_table_split_mask_stage_count": len(split_masks),
        "train_table_split_mask_stages": sorted(split_masks),
    }


def minimize_model_plan(model_plan: dict) -> dict:
    model_plan = copy.deepcopy(model_plan)
    training_job = model_plan["training_job"]
    training_job["num_experiments"] = 1
    training_job["manual_seed"] = 42
    training_job["disable_explain"] = True
    if "refit_trainval" in training_job:
        training_job["refit_trainval"] = False
    if "refit_full" in training_job:
        training_job["refit_full"] = False

    optimization = model_plan.get("optimization", {})
    if "max_epochs" in optimization:
        optimization["max_epochs"] = 1
    if "max_steps_per_epoch" in optimization:
        optimization["max_steps_per_epoch"] = 10
    if "max_val_steps" in optimization:
        optimization["max_val_steps"] = 10
    if "max_test_steps" in optimization:
        optimization["max_test_steps"] = 20
    return model_plan


def run_train_prediction_smoke(
    client: Any,
    *,
    fixture_root: Path | None = None,
    stop_after_snapshot_stats: bool = False,
) -> dict[str, Any]:
    _, _, graph, pquery = provision_smoke_resource(client, fixture_root)

    pquery_response = _post_json(
        client,
        "/v1/predictive_queries/",
        pquery,
        params={"dialect": "PQLANG_V2"},
    )
    pquery_id = pquery_response["id"]
    assert pquery_id.startswith("pquery-")

    graph_response = _post_json(
        client,
        "/v1/graphs",
        graph,
        params={"name_alias": f"{pquery['name']}_graph"},
    )
    graph_id = graph_response["id"]
    assert graph_id.startswith("graph-")

    snapshot_id, _snapshot, snapshot_stats = _create_graph_snapshot(
        client,
        graph_id,
    )
    if stop_after_snapshot_stats:
        return {
            "graph_alias": f"{pquery['name']}_graph",
            "graph_id": graph_id,
            "pquery_id": pquery_id,
            "pquery_name": pquery["name"],
            "snapshot_id": snapshot_id,
            "snapshot_stats_readback": snapshot_stats,
            "stopped_after_snapshot_stats": True,
        }

    plan_request = {
        "query_string": pquery["query_string"],
        "graph_id": graph_id,
        "run_mode": "debug",
    }
    train_table_plan = _post_json(
        client,
        "/v1/predictive_queries/train_table_generation_plan",
        plan_request,
        params={"dialect": "PQLANG_V2"},
    )
    assert train_table_plan["split"]

    train_table_response = _post_json(
        client,
        "/v1/gentraintable_jobs",
        {
            "job_tags": SMOKE_JOB_TAGS["train_table"],
            "pquery_id": pquery_id,
            "plan": train_table_plan,
            "graph_snapshot_id": snapshot_id,
        },
    )
    train_table_job_id = train_table_response["id"]
    train_table_job = _wait_for_job_done(
        client,
        f"/v1/gentraintable_jobs/{train_table_job_id}",
        label="generate-train-table job",
    )
    assert train_table_job["config"]["pquery_id"] == pquery_id
    assert train_table_job["config"]["graph_snapshot_id"] == snapshot_id
    train_table_output = _verify_train_table_output_readback(
        client,
        train_table_job_id,
    )

    model_plan_response = _post_json(
        client,
        "/v1/predictive_queries/model_plan",
        plan_request,
        params={"dialect": "PQLANG_V2"},
    )
    model_plan = minimize_model_plan(model_plan_response["model_plan"])

    training_response = _post_json(
        client,
        "/v1/training_jobs",
        {
            "job_tags": SMOKE_JOB_TAGS["training"],
            "pquery_id": pquery_id,
            "model_plan": model_plan,
            "graph_snapshot_id": snapshot_id,
            "train_table_job_id": train_table_job_id,
        },
    )
    training_job_id = training_response["id"]
    training_job = _wait_for_job_done(
        client,
        f"/v1/training_jobs/{training_job_id}",
        label="training job",
    )
    assert training_job["config"]["pquery_id"] == pquery_id
    assert training_job["config"]["train_table_job_id"] == train_table_job_id
    assert training_job.get("result"), training_job

    pred_table_response = _post_json(
        client,
        "/v1/genpredtable_jobs",
        {
            "job_tags": SMOKE_JOB_TAGS["prediction_table"],
            "pquery_id": pquery_id,
            "plan": {
                "anchor_time": PRED_TABLE_ANCHOR_TIME_NS
            },
            "graph_snapshot_id": snapshot_id,
        },
    )
    pred_table_job_id = pred_table_response["id"]
    pred_table_job = _wait_for_job_done(
        client,
        f"/v1/genpredtable_jobs/{pred_table_job_id}",
        label="generate-prediction-table job",
    )
    assert pred_table_job["config"]["pquery_id"] == pquery_id
    assert pred_table_job["config"]["graph_snapshot_id"] == snapshot_id

    batch_prediction_response = _post_json(
        client,
        "/v1/prediction_jobs",
        {
            "job_tags": SMOKE_JOB_TAGS["batch_prediction"],
            "model_training_job_id": training_job_id,
            "predict_options": {
                "num_workers": 1
            },
            "outputs": [],
            "graph_snapshot_id": snapshot_id,
            "pred_table_job_id": pred_table_job_id,
        },
    )
    assert isinstance(batch_prediction_response,
                      list), batch_prediction_response
    batch_prediction_job_id = batch_prediction_response[0]
    assert batch_prediction_job_id
    batch_prediction_job = _wait_for_job_done(
        client,
        f"/v1/prediction_jobs/{batch_prediction_job_id}",
        label="batch prediction job",
    )
    assert batch_prediction_job["config"]["model_training_job_id"] == (
        training_job_id)
    assert batch_prediction_job["config"]["pred_table_job_id"] == (
        pred_table_job_id)
    assert batch_prediction_job["result"], batch_prediction_job

    return {
        "batch_prediction_job_id": batch_prediction_job_id,
        "graph_alias": f"{pquery['name']}_graph",
        "graph_id": graph_id,
        "pred_table_job_id": pred_table_job_id,
        "pquery_id": pquery_id,
        "pquery_name": pquery["name"],
        "snapshot_id": snapshot_id,
        "snapshot_stats_readback": snapshot_stats,
        "train_table_job_id": train_table_job_id,
        "train_table_output_readback": train_table_output,
        "training_job_id": training_job_id,
    }
