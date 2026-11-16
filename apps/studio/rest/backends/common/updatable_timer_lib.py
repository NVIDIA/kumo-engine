# SPDX-FileCopyrightText: Copyright (c) 2022 Temporal Technologies Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT

import asyncio
from datetime import datetime, timedelta

from temporalio import workflow

from io_utils import logging

logger = logging.getLogger(__name__)


class UpdatableTimer:
    """An updatable timer class for use within Temporal workflows.

    Adapted from Temporal's updatable timer sample at commit
    7577bd6f52652cb82ad9818074931a794d568a39.
    """
    def __init__(self, wake_up_time: datetime) -> None:
        self.wake_up_time = wake_up_time
        self.wake_up_time_updated = False

    async def sleep(self) -> None:
        logger.info(f"sleep_until: {self.wake_up_time}")
        while True:
            now = workflow.now()

            sleep_interval = self.wake_up_time - now
            if sleep_interval <= timedelta(0):
                break
            logger.info(f"Going to sleep for {sleep_interval}")

            try:
                self.wake_up_time_updated = False
                await workflow.wait_condition(
                    lambda: self.wake_up_time_updated,
                    timeout=sleep_interval,
                )
            except asyncio.TimeoutError:
                # checks condition at the beginning of the loop
                continue
        logger.info("sleep_until completed")

    def update_wake_up_time(self, wake_up_time: datetime) -> None:
        logger.info(f"update_wake_up_time: {wake_up_time}")
        self.wake_up_time = wake_up_time
        self.wake_up_time_updated = True

    def get_wake_up_time(self) -> datetime:
        return self.wake_up_time
