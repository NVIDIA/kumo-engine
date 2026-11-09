# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.diskgraph import stores


@pytest.mark.parametrize("backend", [None, "mmap", "buffered", "direct"])
@pytest.mark.parametrize("platform,start_method", [
    ("linux", "fork"),
    ("linux", "spawn"),
    ("linux", None),
    ("darwin", "spawn"),
    ("darwin", None),
    ("darwin", "fork"),
    ("win32", "spawn"),
])
@pytest.mark.parametrize("edges,mode,workers,threads", [
    (1_000, "fused", 16, 1),
    (2_000, "fused", 16, 1),
    (5_000, "standalone", 16, 3),
    (100_000, "standalone", 8, 6),
])
def test_backend_auto_config(backend, platform, start_method, edges, mode,
                             workers, threads, monkeypatch):
    if backend is None:
        monkeypatch.delenv("DG_EDGE_BACKEND", raising=False)
    else:
        monkeypatch.setenv("DG_EDGE_BACKEND", backend)
    monkeypatch.setattr(stores.sys, "platform", platform)
    monkeypatch.setattr(stores.multiprocessing, "get_start_method",
                        lambda allow_none: start_method)
    monkeypatch.setattr(stores, "shm_safe_num_workers", lambda count: count)
    if platform != "linux" and start_method in (None, "spawn"):
        mode, workers, threads = "fused", 0, 1
    if backend in {"buffered", "direct"}:
        mode = "standalone"

    assert stores.diskgraph_auto_config(64, 500 * 1024**3, 100 * 1024**3,
                                        edges) == {
                                            "mode": mode,
                                            "num_workers": workers,
                                            "rayon_threads": threads,
                                        }


@pytest.mark.parametrize("backend", [
    "", "MMAP", "Direct", " buffered", "direct ", "unknown"
])
def test_invalid_backend_rejected(backend, monkeypatch):
    monkeypatch.setenv("DG_EDGE_BACKEND", backend)
    with pytest.raises(ValueError, match="DG_EDGE_BACKEND must be"):
        stores.diskgraph_auto_config(64, 500 * 1024**3, 100 * 1024**3,
                                     1_000)
