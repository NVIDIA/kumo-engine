#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed the local Playwright compose stack through real REST APIs."""

from __future__ import annotations

import asyncio
import copy
import faulthandler
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

BASE_URL = os.environ.get("PLAYWRIGHT_SEED_BASE_URL", "http://localhost:8000")
FIXTURE_PATH = Path(
    os.environ.get("PLAYWRIGHT_SEED_FIXTURE",
                   "/tmp/playwright-e2e-fixtures.json"))
CONNECTOR_ARGS_PATH = Path(
    os.environ.get(
        "PLAYWRIGHT_SEED_CONNECTORS",
        FIXTURE_PATH.with_name("test_e2e_connector_args.json"),
    ))
LOCALIZED_FIXTURE_PATH = Path("/tmp/playwright-e2e-fixtures-local.json")
ORIGINAL_FIXTURE_FPIDS_PATH = Path(
    "/tmp/playwright-e2e-fixtures-original-fpids.json")
LOCALIZED_FIXTURE_FPIDS_PATH = Path(
    "/tmp/playwright-e2e-fixtures-local-fpids.json")
CSV_ROOT = os.environ.get("PLAYWRIGHT_SEED_CSV_ROOT",
                          "/tmp/playwright-csv-data")
PQUERY_NAME = os.environ.get("PLAYWRIGHT_SEED_PQUERY_NAME",
                             "saas_temporal_regression_base")
TRAINING_JOB_TAG = os.environ.get("PLAYWRIGHT_SEED_TRAINING_JOB_TAG",
                                  "playwright-local-training")
GRAPH_ALIAS = os.environ.get("PLAYWRIGHT_SEED_GRAPH_ALIAS", "graph_d1fa556b")
PROJECT_ID = os.environ.get(
    "PLAYWRIGHT_SEED_PROJECT_ID",
    "default",
)
REQUEST_TIMEOUT = float(os.environ.get("PLAYWRIGHT_SEED_REQUEST_TIMEOUT", "30"))
WATCHDOG_SECONDS = int(os.environ.get("PLAYWRIGHT_SEED_WATCHDOG_SECONDS", "90"))


def request_json(
        method: str,
        path: str,
        data: Any | None = None,
        *,
        ok_statuses: tuple[int, ...] = (200, ),
) -> Any:
    body = None
    headers = {"Content-Type": "application/json"}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL.rstrip('/')}{path}", data=body,
                                 headers=headers, method=method)
    started = time.monotonic()
    print(f"seed: HTTP {method} {path} ...", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            if resp.status not in ok_statuses:
                raise RuntimeError(
                    f"{method} {path} returned HTTP {resp.status}: {raw}")
            if raw == "":
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw.strip('"')
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", errors="replace")
        if err.code in ok_statuses:
            return json.loads(raw) if raw else None
        raise RuntimeError(
            f"{method} {path} failed: HTTP {err.code}: {raw}") from err
    finally:
        print(
            f"seed: HTTP {method} {path} done in {time.monotonic() - started:.1f}s",
            flush=True,
        )


def request_bytes(method: str, path: str, data: bytes) -> Any:
    """Send one upload part to either a product-relative or presigned URL."""
    url = (path if path.startswith(
        ('http://', 'https://')) else f"{BASE_URL.rstrip('/')}{path}")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/octet-stream"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        raw = resp.read().decode('utf-8')
        if resp.status != 200:
            raise RuntimeError(
                f"{method} {path} returned HTTP {resp.status}: {raw}")
        return json.loads(raw) if raw else None


def localize_fixture(raw: Any) -> Any:
    marker = "test/csv_data/"
    if isinstance(raw, str) and marker in raw:
        return f"{CSV_ROOT.rstrip('/')}/{raw.split(marker, 1)[1]}"
    if isinstance(raw, list):
        return [localize_fixture(item) for item in raw]
    if isinstance(raw, dict):
        return {key: localize_fixture(value) for key, value in raw.items()}
    return raw


def fixture_graph_fpids(fixture_data: dict[str, Any],
                        fixture_path: Path) -> list[str]:
    from io_utils.json import load_json
    from rest.config.api_command import FixtureDefinition

    fixture_path.write_text(json.dumps(fixture_data), encoding="utf-8")
    fixture = load_json(str(fixture_path), FixtureDefinition)
    return [str(graph.fpid) for graph in fixture.graphs]


def remap_localized_graph_fpids(original: dict[str, Any],
                                localized: dict[str, Any]) -> None:
    original_fpids = fixture_graph_fpids(original, ORIGINAL_FIXTURE_FPIDS_PATH)
    localized_fpids = fixture_graph_fpids(localized,
                                          LOCALIZED_FIXTURE_FPIDS_PATH)
    if len(original_fpids) != len(localized_fpids):
        raise RuntimeError(
            "Localized fixture changed graph count: "
            f"{len(original_fpids)} original, {len(localized_fpids)} localized")

    fpid_map = dict(zip(original_fpids, localized_fpids))
    for query in localized.get("queries", []):
        graph_fpid = query.get("graph_fpid")
        if graph_fpid in fpid_map:
            query["graph_fpid"] = fpid_map[graph_fpid]


def filtered_fixture(raw: dict[str, Any]) -> dict[str, Any]:
    original = copy.deepcopy(raw)
    fixture = localize_fixture(copy.deepcopy(raw))
    remap_localized_graph_fpids(original, fixture)
    queries = [
        query for query in fixture.get("queries", [])
        if query.get("pquery_name") == PQUERY_NAME
    ]
    if len(queries) != 1:
        raise RuntimeError(
            f"Expected exactly one {PQUERY_NAME!r} query, got {len(queries)}")
    original_fpids = fixture_graph_fpids(original, ORIGINAL_FIXTURE_FPIDS_PATH)
    selected_original_fpid = next(query['graph_fpid']
                                  for query in original.get('queries', [])
                                  if query.get('pquery_name') == PQUERY_NAME)
    try:
        graph_index = original_fpids.index(selected_original_fpid)
    except ValueError as exc:
        raise RuntimeError(
            f"Selected query graph {selected_original_fpid} is absent") from exc

    fixture["queries"] = queries
    fixture["graphs"] = [fixture["graphs"][graph_index]]
    # The v2 graph is authoritative; retain only its project-scoped tables so
    # the loader cannot import unrelated legacy connector aliases.
    fixture["tables"] = list(fixture["graphs"][0]["tables"].values())
    fixture["predictions"] = []
    return fixture


def fixture_target(fixture_path: Path):
    from io_utils.json import load_json
    from rest.config.api_command import FixtureDefinition
    from rest.routers.fixture_utils import fixture_to_pquery_resources_v2

    fixture = load_json(str(fixture_path), FixtureDefinition)
    resources = fixture_to_pquery_resources_v2(fixture)
    matches = [
        resource for resource in resources if resource.name == PQUERY_NAME
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one resource named {PQUERY_NAME!r}, got {len(matches)}")
    return matches[0]


def as_json_dict(value: Any) -> dict[str, Any]:
    from io_utils.json import to_json_dict

    return to_json_dict(value)


def seed_fixture() -> None:
    request_json(
        "POST",
        "/v0/fixtures/load_json",
        {
            "file_path": str(LOCALIZED_FIXTURE_PATH),
            "project_id": PROJECT_ID,
        },
    )


def list_names(path: str) -> set[str]:
    separator = '&' if '?' in path else '?'
    response = request_json("GET", f'{path}{separator}project_id={PROJECT_ID}')
    if not isinstance(response, list):
        return set()
    return {
        str(item.get("name"))
        for item in response if isinstance(item, dict) and item.get("name")
    }


def require_names(path: str, expected_names: set[str]) -> None:
    actual_names = list_names(path)
    missing_names = expected_names - actual_names
    if missing_names:
        raise RuntimeError(
            f"Expected {path} to include {sorted(missing_names)}, "
            f"got {sorted(actual_names)}")


def source_table_name(source: dict[str, Any]) -> str:
    """Return a source name from the final relative-source API contract."""
    source_ref = source['source']
    if (isinstance(source_ref, dict) and source_ref.get('mode') == 'relative'
            and source_ref.get('path')):
        return str(source_ref['path'])
    raise ValueError('Playwright seed requires a relative source table path')


def seed_project_uploads(
    fixture: dict[str, Any],
    connector_args: list[dict[str, Any]],
) -> None:
    """Upload the selected fixture's files into its hidden project connector."""
    roots = {
        str(connector['config']['name']): Path(connector['config']['root_dir'])
        for connector in connector_args
    }
    upload_connector = request_json(
        'GET',
        f'/v0/projects/{PROJECT_ID}/connectors/uploads/connector',
    )['connector_id']

    uploaded: set[tuple[str, str]] = set()
    for graph in fixture['graphs']:
        for table in graph['tables'].values():
            source = table['source_table']
            alias = str(source['connector_id'])
            source_name = source_table_name(source)
            file_type = str(source['file_type']).upper()
            key = (source_name, file_type)
            if key not in uploaded:
                suffix = '.csv' if file_type == 'CSV' else '.parquet'
                source_path = roots[alias] / f'{source_name}{suffix}'
                start = request_json(
                    'POST',
                    f'/v0/projects/{PROJECT_ID}/connectors/uploads/start',
                    {
                        'source_table_name': source_name,
                        'file_type': file_type,
                        'num_parts': 1,
                    },
                )
                part = request_bytes(
                    'PUT',
                    start['presigned_part_urls']['1'],
                    source_path.read_bytes(),
                )
                request_json(
                    'POST',
                    f'/v0/projects/{PROJECT_ID}/connectors/uploads/complete',
                    {
                        'source_table_name':
                        source_name,
                        'file_type':
                        file_type,
                        'temp_upload_path':
                        start['temp_upload_path'],
                        'upload_id':
                        start['upload_id'],
                        'parts_metadata': [{
                            'part_number': 1,
                            'etag': part['etag'],
                        }],
                        'validate_data':
                        False,
                    },
                )
                uploaded.add(key)
                print(f'Uploaded project table {source_name}', flush=True)
            source['connector_id'] = upload_connector

    for table in fixture['tables']:
        table['source_table']['connector_id'] = upload_connector


def seed_ui_resources() -> None:
    """Create UI-listable table and graph aliases from the fixture graph.

    The JSON fixture loader creates predictive-query resources, but the
    Playwright route tests exercise the public /v1 table and graph list pages.
    Those list pages only show aliased table/graph entries, so create the exact
    aliases the tests navigate to through the real REST routers.
    """
    target = fixture_target(LOCALIZED_FIXTURE_PATH)
    graph = as_json_dict(target.graph)

    table_names = set(graph.get("tables", {}))
    existing_tables = list_names("/v1/tables/")
    for table_name, table_def in graph.get("tables", {}).items():
        if table_name in existing_tables:
            continue
        query = urlencode({
            "name_alias": table_name,
            "project_id": PROJECT_ID,
        })
        request_json("POST", f"/v1/tables/?{query}", table_def)
        print(f"Created UI table alias {table_name}")

    existing_graphs = list_names("/v1/graphs/")
    graph_alias = GRAPH_ALIAS
    if graph_alias not in existing_graphs:
        query = urlencode({
            "name_alias": graph_alias,
            "project_id": PROJECT_ID,
        })
        response = request_json("POST", f"/v1/graphs/?{query}", graph)
        print(f"Created UI graph alias {graph_alias}: {response}")

    require_names("/v1/tables/", table_names)
    require_names("/v1/graphs/", {graph_alias})


def list_training_jobs_by_tag(tag_value: str) -> list[dict[str, Any]]:
    try:
        response = request_json(
            "POST",
            "/v0/jobs/",
            {
                "project_id": PROJECT_ID,
                "page": 0,
                "limit": 50,
                "job_type": ["TRAINING_JOB"],
                "search_keyword": tag_value,
            },
            ok_statuses=(200, ),
        )
    except Exception:
        return []
    jobs = response.get("jobs", response) if isinstance(response,
                                                        dict) else response
    if not isinstance(jobs, list):
        return []
    return [
        job for job in jobs
        if isinstance(job, dict) and job.get("job_type") == "TRAINING_JOB" and
        job.get("metadata", {}).get("tags", {}).get("playwright") == tag_value
    ]


def training_job_exists(tag_value: str) -> bool:
    return any(
        job.get("execution", {}).get("status") == "DONE"
        for job in list_training_jobs_by_tag(tag_value))


def _resolve_pquery_def_fpid(pqueries, graph_fpid: str, query_str: str):
    """Find the def-fpid of the pquery the fixture loader created in StateDB.

    The fixture load + UI graph alias already persisted exactly one pquery for
    this graph + query string, so match on those two immutable fields rather
    than recomputing the fingerprint.
    """
    from kumo_core.types.schema import FPID

    for definition in pqueries:
        if str(definition.graph_fpid) == str(graph_fpid) and (
                definition.query_str == query_str):
            return FPID(str(definition.fpid))
    raise RuntimeError(
        f"No pquery found for graph {graph_fpid} matching the seed query")


async def _inject_done_training_job(graph_fpid: str, query_str: str) -> str:
    """Write a DONE TRAINING_JOB row straight into the REST StateDB.

    Mirrors the train handler's StateDB writes (resolve pquery + graph,
    create_training_job, apply job_tags) but skips the Temporal workflow and
    flips the row to DONE in-process. The Playwright specs only read row
    metadata (status, the playwright + pquery_id tags, graph_data_version).
    """
    from rest.config.training_job import TrainingJobOutput, TrainingJobResults
    from rest.settings import get_db_storage_root
    from service_framework.deploy_env_config import deploy_env
    from statedb.config.common import (
        GraphIDv2,
        JobEventLogEntry,
        JobStatus,
        JobType,
        PredictiveQueryID,
        ProjectID,
    )
    from statedb.config.job_tag import (
        TAG_KEY_PQUERY_ID,
        TAG_KEY_PQUERY_NAME,
        JobTagKey,
        JobTagValue,
    )
    from statedb.config.timestamp import now_utc
    from statedb.db import MLJobService, open_state_db_from_deploy_config
    from statedb.db.services import PQueryService, StateServices

    deploy_config = deploy_env()
    state_db = await open_state_db_from_deploy_config(
        deploy_config,
        state_root=get_db_storage_root(),
    )
    try:
        pqueries = await PQueryService(state_db).list_definitions()
        pquery_def_fpid = _resolve_pquery_def_fpid(pqueries, graph_fpid,
                                                   query_str)
        project_id = ProjectID(PROJECT_ID)
        graph_entry = await StateServices(state_db).graphs.get_graph_entry(
            project_id, graph_fpid)
        if graph_entry is None:
            raise RuntimeError(f"Graph {graph_fpid} not found in StateDB")

        svc = MLJobService(state_db)
        now = now_utc()
        graph_data_version = now.strftime("%Y%m%dT%H%M%S")
        job_id, _ = await svc.create_training_job(
            pquery_id=PredictiveQueryID.public(pquery_def_fpid),
            pquery_def_fpid=pquery_def_fpid,
            graph_id=GraphIDv2.compose(fpid=graph_entry.id),
            now=now,
            advanced_options=_default_model_plan(),
            graph_data_version=graph_data_version,
            project_id=project_id,
        )

        # job_tags live in a separate table merged by search_jobs. The UI
        # resolves taskType from the pquery_id tag; the playwright tag is the
        # spec's search key.
        await svc.apply_job_tags(
            JobType.TRAINING_JOB,
            job_id,
            {
                JobTagKey("playwright"): JobTagValue(TRAINING_JOB_TAG),
                TAG_KEY_PQUERY_ID: JobTagValue.of_pquery_id(pquery_def_fpid),
                TAG_KEY_PQUERY_NAME: JobTagValue.of_pquery_name(PQUERY_NAME),
            },
        )

        done_at = now_utc()
        await svc.save_training_job_output(
            job_id,
            TrainingJobOutput(
                status=JobStatus.DONE,
                results=TrainingJobResults(),
                log=JobEventLogEntry.done(done_at),
                last_updated_at=done_at,
            ),
        )
        return str(job_id)
    finally:
        await state_db.close()


def _default_model_plan():
    """A defaults-only ModelPlan; create_training_job derives the implicit
    generate-train-table job's plan from its training_table_generation."""
    from kumo_core.model_plan import ModelPlan

    return ModelPlan()


def seed_training_job() -> None:
    if training_job_exists(TRAINING_JOB_TAG):
        print(f"Training job tagged {TRAINING_JOB_TAG} already exists")
        return

    target = fixture_target(LOCALIZED_FIXTURE_PATH)
    # The fixture loader created the pquery + graph in StateDB; we only need
    # their identifiers. graph.fpid is a computed property, not a serialized
    # field, so read it off the resource rather than the JSON dict.
    job_id = asyncio.run(
        _inject_done_training_job(str(target.graph.fpid), target.query_string))
    print(f"Injected DONE training job {job_id}", flush=True)
    if not training_job_exists(TRAINING_JOB_TAG):
        raise RuntimeError(
            f"Injected training job {job_id} not visible as DONE via REST")


def main() -> int:
    print("seed: start (loading + localizing fixture)", flush=True)
    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture = filtered_fixture(raw)
    connectors = localize_fixture(
        json.loads(CONNECTOR_ARGS_PATH.read_text(encoding="utf-8")))
    if not isinstance(connectors, list):
        raise RuntimeError("Connector arguments must be a JSON list")
    client_config = request_json("GET", "/v0/authentication/client-config")
    if not client_config.get("auth_disabled"):
        raise RuntimeError(
            f"Expected auth_disabled=true from local stack, got {client_config}"
        )

    print("seed: phase project uploads", flush=True)
    seed_project_uploads(fixture, connectors)
    # Replacing import aliases with the project upload connector changes the
    # graph fingerprint one final time; keep the selected query attached to
    # that canonical graph.
    fixture['queries'][0]['graph_fpid'] = fixture_graph_fpids(
        fixture,
        LOCALIZED_FIXTURE_FPIDS_PATH,
    )[0]
    LOCALIZED_FIXTURE_PATH.write_text(json.dumps(fixture), encoding="utf-8")
    print("seed: fixture localized and project-scoped", flush=True)
    print("seed: phase load_json", flush=True)
    seed_fixture()
    print("seed: phase ui resources", flush=True)
    seed_ui_resources()
    print("seed: phase inject training job", flush=True)
    seed_training_job()
    print("Playwright local stack seed complete", flush=True)
    return 0


if __name__ == "__main__":
    # If any phase wedges (it has, opaquely, on the CI runner), dump every
    # thread's stack to stderr on a repeating timer so the step log shows
    # exactly where it is stuck instead of a silent timeout with no output.
    faulthandler.dump_traceback_later(WATCHDOG_SECONDS, repeat=True)
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
