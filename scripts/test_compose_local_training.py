#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Smoke test: submit a diskgraph training job to the running compose-local stack
and wait for completion.

Run inside kumo-rest container:
  docker exec -it kumo-rest python /opt/app/scripts/test_compose_local_training.py
"""
import json
import sys
import time
from functools import partial
from pathlib import Path

import requests

BASE_URL = "http://localhost:8888"
FIXTURE_PATH = Path(
    "/opt/app/test/rest/integ/test_cases/test_e2e_fixtures.json")
CONNECTOR_ARGS_PATH = Path(
    "/opt/app/test/rest/integ/connector_args/test_e2e_connector_args.json")
PQL_NAME = "saas_temporal_regression_base"
POLL_INTERVAL = 5
TIMEOUT = 1200


def request_headers():
    return {"Content-Type": "application/json"}


def post(path, payload_str):
    r = requests.post(
        f"{BASE_URL}{path}",
        data=payload_str,
        headers=request_headers(),
        params={"skip_cache": "true"},
        timeout=60,
    )
    if not r.ok:
        print(f"  ERROR {r.status_code}: {r.text[:500]}")
        sys.exit(1)
    return r.json()


def get(path):
    r = requests.get(
        f"{BASE_URL}{path}",
        headers=request_headers(),
        params={"skip_cache": "true"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def dump(obj):
    from io_utils.json import trusted_custom_pydantic_encoder
    return json.dumps(
        obj, default=partial(trusted_custom_pydantic_encoder, insecure=True))


def main():
    from kumoapi.model_plan import RunMode as ApiRunMode
    from kumoapi.model_plan import SuggestModelPlanRequest

    from io_utils.json import load_json
    from kumo.config.model_plan_yaml import model_plan_from_yaml
    from rest.config.api_command import FixtureDefinition
    from rest.config.training_job_v2 import TrainingJobRequest
    from rest.routers.fixture_utils import fixture_to_pquery_resources_v2

    _LOCAL_TO_S3 = {
        "./test/csv_data/FINANCIAL":
        "s3://kumo-public-datasets/financial/csv_data",
        "./test/csv_data/FINANCIAL_E2E":
        "s3://kumo-public-datasets/test-fixtures/FINANCIAL_E2E",
        "./test/csv_data/COUPON_PURCHASE_PREDICTION":
        "s3://kumo-public-datasets/test-fixtures/COUPON_PURCHASE_PREDICTION",
    }

    print(f"Loading fixture: {FIXTURE_PATH}")
    fixture = load_json(str(FIXTURE_PATH), FixtureDefinition)
    connectors = json.loads(CONNECTOR_ARGS_PATH.read_text(encoding="utf-8"))
    for conn in connectors:
        config = conn["config"]
        if config["root_dir"] in _LOCAL_TO_S3:
            config["root_dir"] = _LOCAL_TO_S3[config["root_dir"]]

    print("Creating connectors...")
    for conn in connectors:
        r = requests.post(
            f"{BASE_URL}/v0/projects/default/connectors/",
            data=json.dumps(conn),
            headers=request_headers(),
            params={"skip_cache": "true"},
            timeout=60,
        )
        if r.ok:
            print(f"  created: {r.json()}")
        elif r.status_code == 422 and "already exists" in r.text:
            print(f"  exists: {conn['config']['name']}")
        else:
            print(f"  ERROR {r.status_code}: {r.text[:500]}")
            sys.exit(1)

    resources = fixture_to_pquery_resources_v2(fixture)
    target = next((r for r in resources if r.name == PQL_NAME), None)
    if target is None:
        print(f"ERROR: PQL '{PQL_NAME}' not found in fixture")
        sys.exit(1)

    query_args = next(q for q in fixture.queries if q.pquery_name == PQL_NAME)

    print(f"Registering PQuery: {PQL_NAME}...")
    post("/v1/predictive_queries/?dialect=PQLANG_V2", dump(target))

    if not query_args.model_planner_yaml:
        print("Getting default model plan...")
        from rest.config.graph_v2 import GraphIDv2
        graph_id_v2 = str(GraphIDv2.compose(fpid=target.graph.fpid))
        model_plan_response = post(
            "/v1/predictive_queries/model_plan",
            dump(
                SuggestModelPlanRequest(
                    query_string=target.query_string,
                    graph_id=graph_id_v2,
                    run_mode=ApiRunMode.DEBUG,
                    graph_definition=target.graph,
                )),
        )
        query_args.model_planner_yaml = (
            model_plan_response["suggested_model_plan_yaml"])

    model_plan = model_plan_from_yaml(query_args.model_planner_yaml)

    print("Submitting training job...")
    training_job_id = post(
        "/v0/training_jobs/train",
        dump(
            TrainingJobRequest(
                predictive_query=target.query_string,
                graph_definition=target.graph,
                training_table_gen_plan=model_plan.training_table_generation,
                model_plan=model_plan,
            )),
    )
    print(f"Training job ID: {training_job_id}")

    print("Polling for completion (timeout: 20 min)...")
    start = time.time()
    while time.time() - start < TIMEOUT:
        job = get(f"/v0/training_jobs/{training_job_id}")
        status = job["execution"]["status"]
        elapsed = int(time.time() - start)
        print(f"  [{elapsed}s] status={status}")
        if status == "DONE":
            print("\nTRAINING SUCCEEDED")
            return 0
        if status in ("FAILED", "CANCELLED"):
            print(f"\nTRAINING {status}")
            print(json.dumps(job["execution"].get("results"), indent=2))
            return 1
        time.sleep(POLL_INTERVAL)

    print("\nTIMEOUT: training did not complete within 20 min")
    return 1


if __name__ == "__main__":
    sys.exit(main())
