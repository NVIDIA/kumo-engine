# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest


@pytest.fixture(autouse=True)
def disable_logging():
    # In ML tests, we usually don't care about heaps of logs from other modules
    logging.getLogger('kumo.data').setLevel(logging.ERROR)
    logging.getLogger('kumo.pquery').setLevel(logging.ERROR)
    logging.getLogger('kumo.stats').setLevel(logging.ERROR)
    logging.getLogger('kumo.graph').setLevel(logging.ERROR)
    logging.getLogger('kumo.connector').setLevel(logging.ERROR)
    logging.getLogger('io_utils').setLevel(logging.ERROR)
