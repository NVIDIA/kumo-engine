# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the public embedded server handle and its native resource ownership."""

from __future__ import annotations

import gc
import os
import socket
import time
from pathlib import Path

import diskgraph
import pytest
from pyarrow import flight


@pytest.fixture
def graph(tmp_path: Path) -> Path:
    """Prepare metadata sufficient for listener tests without opening an index."""
    (tmp_path / "index_metadata.json").write_text("{}")
    return tmp_path


def test_embedded_listener_uses_the_calling_process(graph: Path) -> None:
    """The PyO3 handle exposes the same native health through Python and Flight."""
    with diskgraph.SamplingServer(graph) as server:
        with flight.connect(server.endpoint()) as client:
            results = list(client.do_action(flight.Action("health", b"")))
        body = results[0].body.to_pybytes()
        assert f'"pid":{os.getpid()}'.encode() in body
        assert b'"ready":true' in body
        assert server.health()["pid"] == os.getpid()
        assert server.health()["ready"] is True
        port = server.port
    assert server.health()["ready"] is False
    server.close()
    with diskgraph.SamplingServer(graph, port=port) as replacement:
        assert replacement.port == port


def test_occupied_port_failure_does_not_affect_an_existing_server(graph: Path) -> None:
    """Failed startup leaves the existing listener responsive and owns no thread."""
    with diskgraph.SamplingServer(graph) as server:
        with pytest.raises(diskgraph.DiskgraphSystemError, match="Bind sampling"):
            diskgraph.SamplingServer(graph, port=server.port)
        with flight.connect(server.endpoint()) as client:
            assert len(list(client.do_action(flight.Action("health", b"")))) == 1


def test_context_exit_after_exception_closes_the_listener(graph: Path) -> None:
    """Exceptional actor setup cannot leave an embedded listener running."""
    with (
        pytest.raises(RuntimeError, match="setup failed"),
        diskgraph.SamplingServer(graph) as server,
    ):
        port = server.port
        raise RuntimeError("setup failed")
    assert server.health()["ready"] is False
    with diskgraph.SamplingServer(graph, port=port):
        pass


@pytest.mark.parametrize("options", [{"max_threads": 0}, {"max_streams": 0}])
def test_invalid_budgets_fail_before_serving(
    graph: Path, options: dict[str, int]
) -> None:
    """Native resource validation applies even without the core Python wrapper."""
    with pytest.raises(diskgraph.DiskgraphConfigError, match="positive sampling"):
        diskgraph.SamplingServer(graph, **options)


def test_invalid_host_is_rejected(graph: Path) -> None:
    """Invalid bind addresses do not create native server threads."""
    with pytest.raises(ValueError):
        diskgraph.SamplingServer(graph, host="invalid host")


def test_missing_graph_metadata_is_rejected(tmp_path: Path) -> None:
    """A directory without a completed graph cannot serve sampling requests."""
    with pytest.raises(diskgraph.DiskgraphConfigError, match="completed graph"):
        diskgraph.SamplingServer(tmp_path)


def test_advertised_endpoint_preserves_the_bound_port(graph: Path) -> None:
    """Wildcard binds advertise loopback locally or the caller's remote IP."""
    with diskgraph.SamplingServer(graph, host="0.0.0.0") as server:
        assert server.endpoint() == f"grpc://127.0.0.1:{server.port}"
        assert server.endpoint("10.1.2.3") == f"grpc://10.1.2.3:{server.port}"
        assert server.endpoint("::1") == f"grpc://[::1]:{server.port}"


def test_dropping_the_handle_releases_its_listener(graph: Path) -> None:
    """Garbage collection signals shutdown without joining while holding the GIL."""
    server = diskgraph.SamplingServer(graph)
    port = server.port
    del server
    gc.collect()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            probe.settimeout(0.1)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(0.01)
    raise AssertionError("Dropped embedded server retained its listening socket")
