#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
This tool helps list ingested data path.
Note that use this tool with the version similar to training job.

Example usage:
    ./tools/list_ingested_data.py \
        --customer demo \
        --job trainingjob-d1576fa360df47b7be873b6033b14575
"""
import argparse
import asyncio
import os.path as osp

from io_utils import fsutil, json, logging
from kumo.config.graph import GraphConfig, TableConfig
from kumo.runtime import init
from service_framework.deploy_env_config import deploy_env
from statedb.config.common import JobType, PredictiveQueryID, TrainingJobID
from statedb.config.job_tag import (
    TAG_KEY_PQUERY_ID,
    TAG_KEY_PQUERY_NAME,
    JobTagValue,
)
from statedb.db.config import open_state_db_from_deploy_config
from statedb.db.services import StateServices

log = logging.getLogger(__name__)

init(
    initialize_logging=True,
    initialize_tracing=False,
    logging_mode=logging.LoggingMode.PRODUCTION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--customer', type=str, default="demo",
                        help="The customer name, default to demo")
    parser.add_argument('--job', type=str,
                        default="trainingjob-d1576fa360df47b7be873b6033b14575",
                        help="The job id, default to an old job on demo")
    parser.add_argument(
        '--pquery', type=str, default='',
        help=('The pquery name to list ingested data (from most recent'
              ' training)'))
    parser.add_argument('--json_file', type=str, default='',
                        help='The file to output in json format')
    parser.add_argument('--dataplane', type=str,
                        default='s3://kumo-prod-dataplane-bucket',
                        help='The dataplane to use, default to prod')
    parser.add_argument('--version', type=str, default='',
                        help='The version used by the job')
    return parser.parse_args()


def load_graph_config(customer: str, job: str, dataplane: str) -> GraphConfig:
    job_path = f'{dataplane}/{customer}/prod/jobs/{job}'
    try:
        graph_dir = osp.join(job_path, 'graphs')
        graph_dir = fsutil.listdir(graph_dir)[0]
        graph_config_path = osp.join(graph_dir, 'config.json')
    except Exception as e:
        raise ValueError('Unable to locate the graph config for: '
                         f'{customer}/{job} under {job_path}') from e

    return json.load_json(graph_config_path, GraphConfig)


def ingested_table_path(customer: str, table: TableConfig, dataplane: str,
                        version: str) -> str:
    # NOTE: this should be kept in sync with any changes in
    # kumo.data.utils.get_materialize_path but cannot use that util directly
    # due to its inference of the root dir and stage
    table_fpid = table.fingerprint()
    # Check if version is at least v2.10.0
    import re

    def parse_version(ver):
        m = re.match(r'v(\d+)\.(\d+)\.(\d+)', ver)
        if not m:
            return (0, 0, 0)
        return tuple(map(int, m.groups()))

    if parse_version(version) >= (2, 10, 0):
        ingest_fpid_location = table_fpid.ingest_path_fpid
    else:
        ingest_fpid_location = table_fpid.fpid()
    table_path = (
        f'{dataplane}/{customer}/prod/'
        f'data-persist-dir/ingest/{ingest_fpid_location}/{table.data_version}/')

    if not fsutil.exists(osp.join(table_path, '_SUCCESS')):
        fingerprint_path = (f'{dataplane}/{customer}/prod/'
                            f'data-persist-dir/ingest/{ingest_fpid_location}')
        print(f"fingerprint_path: {fingerprint_path}")
        versions = []
        if fsutil.exists(fingerprint_path):
            versions = fsutil.listdir(osp.join(fingerprint_path))
            if versions:
                print(
                    f"Fingerprint {table_fpid} is valid for {table.table_name}"
                    f" and no ingested data found at version {table.data_version}"
                    " but some ingested data does exist.")
                latest_version = sorted(versions)[-1].split('/')[-1]
                table_path_with_latest = osp.join(fingerprint_path,
                                                  latest_version)
                return table_path_with_latest

        raise ValueError(
            f"Unable to find ingested table {table} from {table_path}; "
            "invalid fingerprint path or no ingested data available: "
            f"{fingerprint_path}")
    return table_path


async def get_job_from_pquery(customer: str, pquery_name: str,
                              dataplane: str) -> str:
    state_root = f'{dataplane}/{customer}/prod/db'
    try:
        state_db = await open_state_db_from_deploy_config(
            deploy_env(),
            state_root=state_root,
        )
        services = StateServices(state_db)
        pquery_def = None
        try:
            pquery_id = PredictiveQueryID(pquery_name)
            pquery_def = await services.pqueries.get_definition_by_id(pquery_id)
        except ValueError:
            pquery_def = None
        if pquery_def is None:
            pquery_fpid = await services.pqueries.get_alias(pquery_name)
            if pquery_fpid is not None:
                pquery_def = await services.pqueries.get_definition(pquery_fpid)
        if pquery_def is not None:
            filters = {
                str(TAG_KEY_PQUERY_ID):
                str(JobTagValue.of_pquery_id(pquery_def.fpid))
            }
        else:
            filters = {
                str(TAG_KEY_PQUERY_NAME):
                str(JobTagValue.of_pquery_name(pquery_name))
            }
        tagged_jobs = await services.ml_jobs.list_job_tags(
            JobType.TRAINING_JOB,
            filters,
        )
        if not tagged_jobs:
            raise ValueError(f"No training job found for {pquery_name}.")
        training_jobs = await services.ml_jobs.get_training_jobs(
            [TrainingJobID(str(job_id)) for job_id in tagged_jobs])
    except Exception as exc:
        raise ValueError(
            "Unable to load StateDB metadata to get training job id "
            f"from pquery {pquery_name}") from exc

    if not training_jobs:
        raise ValueError(f"No training job found for {pquery_name}.")
    return str(max(training_jobs, key=job_sort_value).id)


def job_sort_value(job) -> str:
    execution = getattr(job, "execution", None)
    for value in (
            getattr(execution, "last_updated_at", None),
            getattr(execution, "start_time", None),
            getattr(getattr(job, "timestamps", None), "updated_at", None),
            getattr(getattr(job, "timestamps", None), "created_at", None),
            getattr(job, "id", ""),
    ):
        if value is not None:
            return str(value)
    return ""


def main(args: argparse.Namespace):
    # Load the graph config from the job.
    customer = args.customer
    # special case where doordash prod is under /kumo/prod
    if 'kumo-doordash-cache' in args.dataplane == 's3://kumo-doordash-cache':
        customer = 'kumo'
    if args.pquery:
        job = asyncio.run(
            get_job_from_pquery(customer, args.pquery, args.dataplane))
        log.info(f"Using job {job} from pquery {args.pquery}")
    else:
        job = args.job

    graph_config = load_graph_config(customer, job, args.dataplane)

    # For each table, find the source data path.
    ingested_data_source = {}
    table_metadata = {}
    json_output = {}
    for table in graph_config.tables:
        if table.end_time_col:
            table.cols = table.cols + [table.end_time_col]  # type: ignore
        ingested_data_source[table.name] = ingested_table_path(
            customer, table, args.dataplane, args.version)
        table_metadata[table.name] = table
        assert isinstance(table, TableConfig)

        json_output[table.name] = {
            'path': ingested_data_source[table.name],
            'fingerprint': table.fingerprint().fpid()
        }
        print(f'Table name: {table.name}, '
              f'path: {ingested_data_source[table.name]}')

    if args.json_file:
        json.save_json(json_output, args.json_file)


if __name__ == '__main__':
    args: argparse.Namespace = parse_args()
    main(args)
