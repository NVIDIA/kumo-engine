# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script is used by "make reset-devserver" in order to reset REST
# devserver to a clean state.  This will:
# - wipe out DB storage
# - terminate open Temporal workflows
# - wipe out previously persisted data (on local or dev dataplane)

import argparse
import asyncio

from io_utils import fsutil
from kumo.data import get_persist_dir
from rest.backends.ml_backend.data_util import get_job_base_dir
from rest.settings import get_db_storage_root
from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_stage import DeployStage
from temporalio_commons import terminate_open_workflows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--local',
        dest='local_devserver',
        required=False,
        action='store_true',
        default=False,
        help=('If true, reset REST devserver for running in local mode, '
              'i.e., make devserver-local'),
    )
    parser.add_argument('-n', '--dry_run', dest='dry_run', action='store_true',
                        required=False, default=False, help='Dry run.')
    args = parser.parse_args()

    deploy_env = DeployEnvConfig.parse_from_env(deploy_mode=(
        DeployStage.DEV_LOCAL if args.local_devserver else DeployStage.DEV))

    db_storage_root = get_db_storage_root()
    print(f'Delete db storage at {db_storage_root}')
    if not args.dry_run:
        fsutil.rm(db_storage_root, ignore_errors=True)

    data_persist_dir = get_persist_dir()
    print(f'Delete data persist dir at {data_persist_dir}')
    if not args.dry_run:
        fsutil.rm(data_persist_dir, ignore_errors=True)

    jobs_dir = get_job_base_dir()
    print(f'Delete jobs dir at {jobs_dir}')
    if not args.dry_run:
        fsutil.rm(jobs_dir, ignore_errors=True)

    print(
        f'Terminate open workflows in {deploy_env.temporal_server_address} Temporal'
    )
    asyncio.run(terminate_open_workflows.run(deploy_env, ignore_error=True))
