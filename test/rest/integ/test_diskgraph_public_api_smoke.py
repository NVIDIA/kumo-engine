# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from test.rest.integ.public_api_smoke_helpers import run_train_prediction_smoke

import pytest
from fastapi.testclient import TestClient

# Phase 1b converted-owner smoke extracted from
# test/rest/integ/test_diskgraph_public_api_e2e.py::
# test_diskgraph_decoupled_training_and_prediction. Keep this file raw
# HTTP/JSON only; the original owner retains diskgraph filesystem/config
# assertions that are not part of the black-box gate.
pytestmark = [
    pytest.mark.usefixtures("_pool_subprocesses"),
    pytest.mark.parametrize("storage_backend", ["local"], indirect=True),
]


def test_diskgraph_minimal_train_prediction_smoke(public_api_client: TestClient):
    run_train_prediction_smoke(public_api_client)


def test_diskgraph_snapshot_stats_smoke(public_api_client: TestClient):
    run_train_prediction_smoke(
        public_api_client,
        stop_after_snapshot_stats=True,
    )
