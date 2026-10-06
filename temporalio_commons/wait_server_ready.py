# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from io_utils import logging
from service_framework.deploy_env_config import TemporalCloudConfig, deploy_env
from temporalio_commons.temporal_client import TemporalClient

log = logging.getLogger(__name__)


async def main(max_timeout: timedelta):
    cfg = deploy_env()
    if isinstance(cfg.orchestration, TemporalCloudConfig):
        log.info('Skip setup of temporal cluster: using Temporal Cloud')
        return
    temporal_client = TemporalClient(cfg)
    try:
        start_time = datetime.now(timezone.utc)
        while datetime.now(timezone.utc) - start_time < max_timeout:
            try:
                await temporal_client.init()
                await temporal_client.create_namespace()
                log.info('Successfully connected to namespace %s at %s',
                         cfg.temporal_namespace, cfg.temporal_server_address)
                return
            except Exception as ex:  # pylint: disable=broad-except
                log.info('Temporal server is not ready, ex: %s', ex)
                await asyncio.sleep(5.0)
        raise RuntimeError(
            f'Temporal is not ready at {cfg.temporal_server_address} '
            f'after {datetime.now(timezone.utc) - start_time}')
    finally:
        # This module runs as a short-lived readiness subprocess.  Explicitly
        # release the Rust-backed gRPC bridge before interpreter finalization;
        # leaving it to CPython teardown can abort with PyGILState_Release.
        await temporal_client.close()


if __name__ == "__main__":
    logging.initialize_logging(logging.LoggingMode.DEBUG, log_to_file=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timeout_seconds', dest='timeout_seconds',
                        default=120, type=int, required=False)
    args = parser.parse_args()

    asyncio.run(main(timedelta(seconds=int(args.timeout_seconds))))
