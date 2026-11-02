# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check stream admission budgets without starting a native listener."""

from __future__ import annotations

import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumo_core.diskgraph.sampling_service import (
    SamplingService,
    SamplingServiceConfig,
)


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    """Keep configuration checks independent of application services."""
    return False


@pytest.mark.parametrize(
    ("max_streams", "expected"), [(None, 3), (1, 1), (4, 4), (6, 6)]
)
def test_sampling_service_passes_stream_budget_to_native_server(
    monkeypatch: pytest.MonkeyPatch,
    max_streams: int | None,
    expected: int,
) -> None:
    """Default to full refit overlap and preserve explicit worker budgets."""
    root = Path("/prepared/graph")
    monkeypatch.setattr(Path, "resolve", Mock(return_value=root))
    monkeypatch.setattr(Path, "is_file", Mock(return_value=True))
    native_server = Mock()
    native_server.return_value.port = 8815
    monkeypatch.setitem(
        sys.modules, "diskgraph", SimpleNamespace(SamplingServer=native_server)
    )
    config = (
        SamplingServiceConfig(root)
        if max_streams is None
        else SamplingServiceConfig(root, max_streams=max_streams)
    )

    with closing(SamplingService(config)) as service:
        assert service.port == 8815
        native_server.assert_called_once_with(
            root, host="127.0.0.1", port=0, max_streams=expected, max_threads=2
        )

    native_server.return_value.close.assert_called_once_with()
