# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import kumo_core.diskgraph.runner as diskgraph_runner
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.diskgraph.ingest import (
    DiskgraphIngestSpec,
    execute_diskgraph_ingest,
)
from kumo_core.diskgraph.snapshot import (
    DiskgraphIngestResult,
    DiskgraphLocalConfig,
    DiskgraphSnapshotError,
    DiskgraphSnapshotHandle,
    LocalSnapshotStore,
)

ZERO_COUNTS = ("diskgraph ingest completed: 0 node tables, 0 edge relations, "
               "0 nodes, 0 edges")


@pytest.mark.parametrize("stats,summary", [
    ({}, ZERO_COUNTS),
    ({"stats": None}, ZERO_COUNTS),
    ({"node_tables": 9, "total_nodes": 9}, ZERO_COUNTS),
    ({"stats": {"node_tables": 2, "edge_relations": 1,
                "total_nodes": 7, "total_edges": 8}},
     "diskgraph ingest completed: 2 node tables, 1 edge relations, "
     "7 nodes, 8 edges"),
])
def test_execute_ingest_uses_real_converter_runner_boundary(
        stats: dict, summary: str, caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    caplog.set_level("INFO")
    converted = {"tables": [{"name": "events", "columns": []}]}
    captured: dict[str, object] = {}

    class Converter:
        def __init__(self, graph, persist, versions, *, source_paths):
            captured.update(
                graph=graph,
                persist=persist,
                versions=versions,
                source_paths=source_paths,
            )

        def convert(self):
            return converted

    monkeypatch.setitem(sys.modules, "diskgraph",
                        SimpleNamespace(DiskgraphConfigConverter=Converter))

    def runner(ingest_config, output_dir, execution):
        captured.update(
            ingest_config=ingest_config,
            output_dir=output_dir,
            execution=execution,
        )
        (output_dir / "col_stats.json").write_text("{}")
        (output_dir / "ingest_stats.json").write_text("{}")
        (output_dir / "kumo_col_stats").mkdir()
        return diskgraph_runner.IngestRunResult(stats, 8, 1, {"context": 8})

    result = execute_diskgraph_ingest(
        {
            "name":
            "features",
            "tables":
            [{
                "table_name": "events",
                "source_table_name": "main.features.events",
                "connector_id": "native",
                "cols": [{
                    "name": "event_id",
                    "dtype": "int",
                    "stype": "id",
                }],
                "fkeys": [],
            }]
        },
        DiskgraphIngestSpec(
            data_persist_dir=str(tmp_path / "persist"),
            table_data_versions={"events": "7"},
            source_paths={"events": str(tmp_path / "parquet")},
            output_dir=str(tmp_path / "index"),
        ),
        runner=runner,
    )

    assert captured["ingest_config"] == converted
    assert captured["versions"] == {"events": "7"}
    converter_graph = captured["graph"]
    assert isinstance(converter_graph, dict)
    assert converter_graph["name"] == "features"
    converter_table = converter_graph["tables"][0]
    assert converter_table["data_source"] == "native"
    assert "connector_id" not in converter_table
    assert result == DiskgraphIngestResult.for_index(tmp_path / "index")
    assert summary in [r.getMessage() for r in caplog.records]


def test_configured_runner_maps_portable_policy(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(config, output_dir, **kwargs):
        captured.update(config=config, output_dir=output_dir, kwargs=kwargs)
        return diskgraph_runner.IngestRunResult({}, 4, 1)

    monkeypatch.setattr(diskgraph_runner, "run_ingest", fake_run)
    execution = diskgraph_runner.IngestExecutionConfig(
        initial_threads=4,
        max_attempts=2,
        poll_interval_seconds=0.25,
        extra_args=("--example", ),
    )

    diskgraph_runner.run_ingest_configured({}, tmp_path, execution)

    assert captured["kwargs"]["initial_threads"] == 4
    assert captured["kwargs"]["max_inner_attempts"] == 2
    assert captured["kwargs"]["extra_args"] == ["--example"]


def test_runner_persists_resume_config_and_forwards_extra_arguments(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        (output / "ingest_stats.json").write_text(json.dumps({"total_nodes":
                                                              3}))
        return 0, ""

    monkeypatch.setattr(
        diskgraph_runner,
        "_resolve_ingest_bin",
        lambda: "/bin/ingest",
    )
    monkeypatch.setattr(diskgraph_runner, "_run_with_heartbeat", run)
    output = tmp_path / "graph" / "version"

    result = diskgraph_runner.run_ingest(
        {"dataset_name": "small"},
        output,
        initial_threads=1,
        extra_args=["--source-config-file", "/tmp/source.cfg"],
    )

    config_path = diskgraph_runner.persisted_ingest_config_path(output)
    assert json.loads(config_path.read_text()) == {"dataset_name": "small"}
    assert result.stats == {"total_nodes": 3}
    assert commands[0][-2:] == [
        "--source-config-file",
        "/tmp/source.cfg",
    ]

    diskgraph_runner.run_ingest(None, output, initial_threads=1)
    assert json.loads(config_path.read_text()) == {"dataset_name": "small"}


def _require_ingest_binary() -> None:
    if not os.access(diskgraph_runner._resolve_ingest_bin(), os.X_OK):
        pytest.skip("DiskGraph ingest binary is not installed")


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def test_real_ingest_logs_progress_and_graph_counts(
        caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    _require_ingest_binary()
    for table, columns in {
            "users": {
                "user_id": [1, 2, 3]
            },
            "orders": {
                "order_id": [10, 11, 12, 13],
                "user_id": [1, 2, 2, 3]
            },
    }.items():
        (tmp_path / table).mkdir()
        pq.write_table(pa.table(columns), tmp_path / table / "part.parquet")

    def table(name: str, fkeys: list[dict[str, str]]) -> dict:
        return {
            "table_name": name,
            "source_table_name": name,
            "connector_id": "native",
            "pkey": {
                "name": f"{name[:-1]}_id",
                "dtype": "int",
                "stype": "ID"
            },
            "cols": [],
            "fkeys": fkeys,
        }

    caplog.set_level("INFO")
    execute_diskgraph_ingest(
        {
            "name":
            "shop",
            "tables": [
                table("users", []),
                table("orders", [{
                    "name": "user_id",
                    "dtype": "int",
                    "stype": "ID",
                    "dst_table": "users",
                    "dst_key": "user_id",
                }]),
            ],
        },
        DiskgraphIngestSpec(
            data_persist_dir=str(tmp_path / "persist"),
            table_data_versions={
                "users": "1",
                "orders": "1"
            },
            source_paths={
                "users": str(tmp_path / "users"),
                "orders": str(tmp_path / "orders")
            },
            output_dir=str(tmp_path / "index"),
        ),
    )

    messages = _messages(caplog)
    assert 'ingest: table started phase="encode" table=users' in messages
    assert any(
        message.startswith("ingest: Ingestion stage progress phase=Index "
                           "table=orders") and "progress_pct=100" in message
        for message in messages)
    assert any(
        message.startswith("ingest: index stage complete")
        for message in messages)
    assert not any("\x1b" in message for message in messages)
    assert ("diskgraph ingest completed: 2 node tables, 2 edge relations, "
            "7 nodes, 8 edges") in messages


def test_real_ingest_nulls_uint64_values_a_declared_int64_cannot_hold(
        tmp_path: Path) -> None:
    _require_ingest_binary()
    (tmp_path / "t").mkdir()
    pq.write_table(
        pa.table({
            "id": range(4),
            "big": pa.array([5, 2**63 - 1, 2**63, 2**64 - 1], pa.uint64()),
        }), tmp_path / "t" / "part.parquet")

    execute_diskgraph_ingest(
        {
            "name":
            "g",
            "tables": [{
                "table_name": "t",
                "source_table_name": "t",
                "connector_id": "native",
                "pkey": {
                    "name": "id",
                    "dtype": "int64",
                    "stype": "ID"
                },
                "cols": [{
                    "name": "big",
                    "dtype": "int64",
                    "stype": "categorical"
                }],
                "fkeys": [],
            }],
        },
        DiskgraphIngestSpec(
            data_persist_dir=str(tmp_path / "persist"),
            table_data_versions={"t": "1"},
            source_paths={"t": str(tmp_path / "t")},
            output_dir=str(tmp_path / "index"),
        ),
    )

    stats = json.loads(
        (tmp_path / "index" / "col_stats.json").read_text())["t"]["big"]
    assert dict(stats["category_counts"]) == {5: 1, 2**63 - 1: 1}
    assert stats["null_count"] == 2
    assert stats["parse_fail_count"] == 2


def test_failed_real_ingest_keeps_diagnostic_out_of_progress_log(
        caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    _require_ingest_binary()
    missing = tmp_path / "missing.parquet"
    caplog.set_level("INFO")

    with pytest.raises(diskgraph_runner.NonRetryableDiskgraphError) as error:
        diskgraph_runner.run_ingest(
            {
                "dataset_name":
                "shop",
                "tables": [{
                    "name": "users",
                    "columns": [{
                        "name": "user_id",
                        "ingested_dtype": "int64"
                    }],
                    "primary_key": "user_id",
                    "source": {
                        "type": "parquet",
                        "paths": [str(missing)]
                    },
                }],
            },
            tmp_path / "index",
            initial_threads=1,
        )

    assert error.value.diagnostic is not None
    assert error.value.diagnostic.code == "source.metadata_failed"
    assert str(missing) in error.value.private_output_tail
    assert not any(str(missing) in message for message in _messages(caplog))


def test_progress_is_logged_while_the_child_is_still_running(
        caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    release = tmp_path / "release"
    child = ("import pathlib, sys, time\n"
             "print('\\x1b[2m2026-10-02T00:00:00Z\\x1b[0m \\x1b[32m INFO"
             "\\x1b[0m diskgraph_ingest: table started table=users')\n"
             "print('Error: bad row value', 'row-' + 'secret', flush=True)\n"
             "deadline = time.monotonic() + 30\n"
             "while not pathlib.Path(sys.argv[1]).exists():\n"
             "    if time.monotonic() > deadline: sys.exit(1)\n"
             "    time.sleep(0.01)\n")

    def heartbeat() -> None:
        if "ingest: table started table=users" in _messages(caplog):
            release.touch()

    caplog.set_level("INFO")
    rc, tail = diskgraph_runner._run_with_heartbeat(
        [sys.executable, "-c", child, str(release)],
        mem_limit_bytes=None,
        heartbeat_interval_s=0.05,
        heartbeat=heartbeat,
    )

    assert rc == 0
    assert "row-secret" in tail
    assert not any("row-secret" in message for message in _messages(caplog))


def test_forward_progress_strips_non_sgr_ansi_sequences(
        caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    line = (b"\x1b[2K2026-10-02T00:00:00Z INFO diskgraph_ingest: "
            b"table started table=users\n")
    with tempfile.TemporaryFile(mode="w+b") as output:
        output.write(line)
        output.flush()
        offset, skipping = diskgraph_runner._forward_progress(output, 0)

    assert offset == len(line)
    assert skipping is False
    assert "ingest: table started table=users" in _messages(caplog)


def test_forward_progress_drops_overlong_line_then_forwards_next(
        caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    cap = diskgraph_runner._PROGRESS_READ_CAP
    # Never breaks for a newline within the first `cap` bytes.
    overlong = b"x" * (cap + 10) + b"\n"
    progress = (b"2026-10-02T00:00:00Z INFO diskgraph_ingest: "
                b"table started table=users\n")
    with tempfile.TemporaryFile(mode="w+b") as output:
        output.write(overlong)
        output.write(progress)
        output.flush()

        # First poll only has a bounded read available: no newline in it,
        # so the overlong line is dropped up to the cap and "skipping"
        # latches on instead of re-reading the ever-growing line.
        offset, skipping = diskgraph_runner._forward_progress(output, 0)
        assert offset == cap
        assert skipping is True
        assert _messages(caplog) == []

        # Next poll sees the overlong line's trailing newline, discards
        # its remainder, and still forwards the real progress line after
        # it.
        offset, skipping = diskgraph_runner._forward_progress(
            output, offset, skipping)
        assert offset == len(overlong) + len(progress)
        assert skipping is False
        assert _messages(caplog) == ["ingest: table started table=users"]


def test_final_read_drains_output_past_one_chunk(
        caplog: pytest.LogCaptureFixture) -> None:
    junk = diskgraph_runner._PROGRESS_READ_CAP * 3
    child = (f"print(('x' * 99 + chr(10)) * ({junk} // 100))\n"
             "print('2026-10-02T00:00:00Z  INFO diskgraph_ingest: "
             "index stage complete table=users')\n")
    caplog.set_level("INFO")
    rc, _ = diskgraph_runner._run_with_heartbeat(
        [sys.executable, "-c", child],
        mem_limit_bytes=None,
        heartbeat_interval_s=60,
    )

    assert rc == 0
    assert "ingest: index stage complete table=users" in _messages(caplog)


def test_progress_read_failure_does_not_fail_the_ingest(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_args):
        raise OSError("read failed")

    monkeypatch.setattr(diskgraph_runner.os, "pread", broken)
    rc, tail = diskgraph_runner._run_with_heartbeat(
        [sys.executable, "-c", "print('done')"],
        mem_limit_bytes=None,
        heartbeat_interval_s=0.05,
    )

    assert rc == 0
    assert "done" in tail


def _diagnostic(*, code: str, message: str) -> dict:
    return {
        "version": 1,
        "event": "diskgraph.ingest_failed",
        "diagnostic_id": f"dg-{'a' * 24}",
        "visibility": "ordinary",
        "component": "diskgraph-ingest",
        "operation": "validate_input",
        "category": "validation",
        "code": code,
        "retryable": False,
        "summary": message,
        "public": {
            "code": code,
            "message": message,
            "non_retryable": True,
        },
        "fields": {
            "attempt": 1,
            "phase": "encode",
            "exit_code": None,
            "signal": None,
        },
        "chain": [{
            "kind": "error",
            "summary": message,
        }],
        "frames": [],
        "fingerprint": f"dg-v1-{'b' * 64}",
        "truncated": {
            "chain": False,
            "frames": False,
            "summary": False,
        },
    }


def test_diagnostic_exposes_only_allowlisted_public_projection(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        diskgraph_runner,
        "_resolve_ingest_bin",
        lambda: "/bin/ingest",
    )

    def run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        (output / ".diagnostic.json").write_text(
            json.dumps(
                _diagnostic(
                    code="diskgraph_ingest_validation",
                    message="source schema is missing column 'symbol'",
                )))
        return 78, "provider-secret-canary"

    monkeypatch.setattr(diskgraph_runner, "_run_with_heartbeat", run)

    with pytest.raises(diskgraph_runner.NonRetryableDiskgraphError) as caught:
        diskgraph_runner.run_ingest({}, tmp_path / "graph", initial_threads=1)

    assert str(caught.value) == "source schema is missing column 'symbol'"
    assert "provider-secret-canary" not in str(caught.value)
    assert caught.value.private_output_tail == "provider-secret-canary"
    assert caught.value.diagnostic is not None
    assert len(caught.value.output_fingerprint) == 64


def test_ingest_error_fingerprint_uses_normalized_multibyte_tail() -> None:
    error = diskgraph_runner.NonRetryableDiskgraphError(
        "safe detail",
        private_output_tail="€" * 6_667,
    )

    normalized_bytes = error.private_output_tail.encode("utf-8")
    assert len(normalized_bytes) <= 20_000
    assert error.private_output_tail == "€" * 6_666
    assert error.output_fingerprint == hashlib.sha256(
        normalized_bytes).hexdigest()


def test_diagnostic_rejects_untrusted_public_projection(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(diskgraph_runner, "_resolve_ingest_bin",
                        lambda: "/bin/ingest")

    def run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        (output / ".diagnostic.json").write_text(
            json.dumps(
                _diagnostic(
                    code="provider_error",
                    message="provider-secret-canary",
                )))
        return 78, "provider-secret-canary"

    monkeypatch.setattr(diskgraph_runner, "_run_with_heartbeat", run)

    with pytest.raises(diskgraph_runner.NonRetryableDiskgraphError) as caught:
        diskgraph_runner.run_ingest({}, tmp_path / "graph", initial_threads=1)

    assert "provider-secret-canary" not in str(caught.value)
    assert "no trusted public error detail" in str(caught.value)


def test_portable_error_keeps_retry_classification() -> None:
    assert diskgraph_runner.DiskgraphNoSpaceError("full").non_retryable is False


def test_typed_no_space_diagnostic_names_output_path(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(diskgraph_runner, "_resolve_ingest_bin",
                        lambda: "/bin/ingest")

    def run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        diagnostic = _diagnostic(code="diskgraph_ingest_validation",
                                 message="unused")
        diagnostic.update(operation="write_snapshot", category="resource",
                          code="ingest.no_space", retryable=True,
                          summary="No space left on device", public=None)
        (output / ".diagnostic.json").write_text(json.dumps(diagnostic))
        return 1, "private path canary"

    monkeypatch.setattr(diskgraph_runner, "_run_with_heartbeat", run)
    output = tmp_path / "graph"

    with pytest.raises(diskgraph_runner.DiskgraphNoSpaceError) as caught:
        diskgraph_runner.run_ingest({}, output, initial_threads=1)

    assert str(output) in str(caught.value)
    assert "Free local capacity" in str(caught.value)
    assert "private path canary" not in str(caught.value)


def test_local_snapshot_paths_listing_verify_and_restore(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    durable_root = tmp_path / "durable"
    handle = DiskgraphSnapshotHandle("graph", "v1")
    store = LocalSnapshotStore(DiskgraphLocalConfig(str(local_root)))
    result = DiskgraphIngestResult.for_index(store.path(handle))
    marker = result.to_json()
    markers: dict[str, str] = {}

    def verify(path):
        assert path == store.path(handle)

    fake_diskgraph = SimpleNamespace(
        verify_snapshot_integrity=verify,
        read_success_marker=lambda path: markers.get(path),
        touch_success_marker=lambda path: None,
        write_success_marker=lambda path, value: markers.__setitem__(
            path, value),
    )
    monkeypatch.setitem(sys.modules, "diskgraph", fake_diskgraph)

    source = store.durable_path(handle, str(durable_root))

    def copy_tree(_source, destination):
        assert _source == source
        Path(destination).mkdir(parents=True)
        markers[destination] = marker

    restored = store.restore(
        handle,
        str(durable_root),
        exists=lambda path: path == source + "/_SUCCESS",
        copy_tree=copy_tree,
    )

    assert restored == result
    assert store.list_ready() == [handle]
    assert store.diskgraph_durable_path(
        handle, "abfss://container@account.dfs.core.windows.net/root"
    ) == "az://account/container/root/diskgraph_snapshots/graph/v1"


def test_checkpoint_stages_then_atomically_publishes(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = LocalSnapshotStore(DiskgraphLocalConfig(str(tmp_path / "local")))
    handle = DiskgraphSnapshotHandle("graph", "v2")
    local = store.path(handle)
    marker = DiskgraphIngestResult.for_index(local).to_json()
    markers = {local: marker}
    monkeypatch.setitem(
        sys.modules,
        "diskgraph",
        SimpleNamespace(
            verify_snapshot_integrity=lambda path: None,
            read_success_marker=lambda path: markers.get(path),
        ),
    )
    copied: list[tuple[str, str]] = []
    replaced: list[tuple[str, str]] = []

    def copy_tree(source: str, staging: str) -> None:
        copied.append((source, staging))
        markers[staging] = marker

    def replace_tree(staging: str, target: str) -> None:
        replaced.append((staging, target))
        markers[target] = markers.pop(staging)

    target = store.checkpoint(
        handle,
        str(tmp_path / "volume"),
        exists=lambda path: path.removesuffix("/_SUCCESS") in markers,
        copy_tree=copy_tree,
        replace_tree=replace_tree,
    )

    assert copied[0][0] == local
    assert copied[0][1].startswith(target + ".tmp-")
    assert replaced == [(copied[0][1], target)]


def test_verify_rejects_missing_success_after_integrity_pass(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_diskgraph = SimpleNamespace(
        verify_snapshot_integrity=lambda path: None,
        read_success_marker=lambda path: None,
    )
    monkeypatch.setitem(sys.modules, "diskgraph", fake_diskgraph)
    store = LocalSnapshotStore(DiskgraphLocalConfig(str(tmp_path)))

    with pytest.raises(DiskgraphSnapshotError, match="_SUCCESS"):
        store.verify(DiskgraphSnapshotHandle("graph", "v1"))


def test_result_json_is_stable() -> None:
    result = DiskgraphIngestResult("/i", "/c", "/s", "/k")
    assert DiskgraphIngestResult.from_json(result.to_json()) == result
    assert json.loads(result.to_json())["index_path"] == "/i"


def test_reject_cached_empty_required_time_table_without_stats(tmp_path):
    from kumo_core.diskgraph.snapshot import (
        DiskgraphSnapshotError,
        validate_cached_source_times,
    )
    metadata = {
        "node_types": [{
            "name": "events",
            "create_time_col": "ts",
            "shards": [{
                "min_id": 1,
                "max_id": 0
            }]
        }]
    }
    (tmp_path / "index_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(DiskgraphSnapshotError, match="correct the source"):
        validate_cached_source_times(str(tmp_path))


def test_cached_discarded_time_parse_failures_remain_allowed(tmp_path):
    from kumo_core.diskgraph.snapshot import validate_cached_source_times
    metadata = {
        "node_types": [{
            "name": "events",
            "create_time_col": "ts",
            "shards": [{
                "min_id": 0,
                "max_id": 1
            }]
        }]
    }
    stats = {
        "events": {
            "ts": {
                "parse_fail_count": 1
            },
            "optional_ts": {
                "parse_fail_count": 1
            }
        }
    }
    (tmp_path / "index_metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "col_stats.json").write_text(json.dumps(stats))
    validate_cached_source_times(str(tmp_path))


@pytest.mark.parametrize("operation", ["restore", "checkpoint"])
def test_snapshot_store_rejects_empty_required_time_table(
        monkeypatch, tmp_path, operation):
    import shutil

    store = LocalSnapshotStore(DiskgraphLocalConfig(str(tmp_path / "local")))
    handle = DiskgraphSnapshotHandle("graph", "v1")
    durable_root = str(tmp_path / "durable")
    source = Path(store.durable_path(handle, durable_root)
                  if operation == "restore" else store.path(handle))
    source.mkdir(parents=True)
    marker = DiskgraphIngestResult.for_index(str(source)).to_json()
    (source / "_SUCCESS").write_text(marker)
    (source / "index_metadata.json").write_text(json.dumps({
        "node_types": [{
            "name": "events", "create_time_col": "ts",
            "shards": [{"min_id": 1, "max_id": 0}],
        }],
    }))
    monkeypatch.setitem(sys.modules, "diskgraph", SimpleNamespace(
        verify_snapshot_integrity=lambda path: None,
        read_success_marker=lambda path: (Path(path) / "_SUCCESS").read_text(),
    ))
    with pytest.raises(DiskgraphSnapshotError, match="no usable rows"):
        if operation == "restore":
            store.restore(handle, durable_root, exists=lambda p: Path(p).exists(),
                          copy_tree=shutil.copytree)
        else:
            store.checkpoint(
                handle, durable_root, exists=lambda p: Path(p).exists(),
                copy_tree=shutil.copytree, replace_tree=shutil.move)
    if operation == "checkpoint":
        assert not Path(durable_root).exists()
    assert (source / "_SUCCESS").read_text() == marker


@pytest.mark.parametrize("invalid_first", [True, False])
def test_list_ready_skips_invalid_snapshot_without_hiding_valid(
        monkeypatch, tmp_path, invalid_first):
    store = LocalSnapshotStore(DiskgraphLocalConfig(str(tmp_path)))
    versions = ["invalid", "valid"] if invalid_first else ["valid", "invalid"]
    markers = {}
    for version in versions:
        path = Path(store.path(DiskgraphSnapshotHandle("graph", version)))
        path.mkdir(parents=True)
        markers[str(path)] = DiskgraphIngestResult.for_index(str(path)).to_json()
        (path / "index_metadata.json").write_text(json.dumps({
            "node_types": [{
                "name": "events", "create_time_col": "ts",
                "shards": [{"min_id": 1,
                            "max_id": 0 if version == "invalid" else 1}],
            }],
        }))
    touched = []
    monkeypatch.setitem(sys.modules, "diskgraph", SimpleNamespace(
        read_success_marker=lambda path: markers.get(path),
        touch_success_marker=touched.append,
        verify_snapshot_integrity=lambda path: None,
    ))
    assert store.list_ready() == [DiskgraphSnapshotHandle("graph", "valid")]
    assert touched == [store.path(DiskgraphSnapshotHandle("graph", "valid"))]
    with pytest.raises(DiskgraphSnapshotError, match="no usable rows"):
        store.verify(DiskgraphSnapshotHandle("graph", "invalid"))
    assert Path(store.path(DiskgraphSnapshotHandle("graph", "invalid"))).exists()
