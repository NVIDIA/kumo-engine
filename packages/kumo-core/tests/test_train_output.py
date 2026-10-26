# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextvars
import errno
import os
import threading
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from kumo_core.train.artifact_stats import (
    PREDICTION_ARTIFACT_STATS_MAX_ROW_GROUP_BYTES,
    PREDICTION_ARTIFACT_STATS_MAX_ROW_GROUP_ROWS,
)
from kumo_core.train.io import StreamingFrameWriter
from kumo_core.train.output import (
    ArrowPredOutput,
    PredOutput,
    StreamingPredOutputWriter,
)


def test_streaming_prediction_preserves_timestamp_precision(tmp_path):
    frame = pd.DataFrame({
        "ENTITY": [1],
        "TIMESTAMP": [pd.Timestamp("2026-07-22 01:02:03.123456789")],
        "PREDICTION": ["clicks"],
    })

    with StreamingPredOutputWriter(str(tmp_path), 0) as writer:
        writer.write(PredOutput(frame, None))

    paths = [path for path in tmp_path.rglob("*.parquet") if path.is_file()]
    assert len(paths) == 1
    schema = pq.read_schema(paths[0])
    assert schema.field("TIMESTAMP").type == pa.timestamp("ns")
    pd.testing.assert_frame_equal(pq.read_table(paths[0]).to_pandas(), frame)


def _read_parts(tmp_path, pattern="part-*.parquet"):
    paths = sorted(tmp_path.rglob(pattern))
    return pa.concat_tables([pq.read_table(path) for path in paths])


def test_arrow_prediction_matches_pandas_schema_and_values(tmp_path):
    expected = pd.DataFrame({
        "ENTITY": ["a", "a", "b"],
        "CLASS": ["x", None, "z"],
        "SCORE":
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "TIMESTAMP":
        pd.to_datetime([
            "2026-08-14 01:02:03.123456789",
            "2026-08-14 01:02:03.123456789",
            None,
        ]),
    })
    table = pa.Table.from_pandas(expected, preserve_index=False)
    with StreamingPredOutputWriter(str(tmp_path), 0) as writer:
        writer.write(ArrowPredOutput(table))

    actual = _read_parts(tmp_path).to_pandas()
    pd.testing.assert_frame_equal(actual, expected)
    assert _read_parts(tmp_path).schema.field("TIMESTAMP").type == pa.timestamp(
        "ns")


@pytest.mark.parametrize("kind", ["pandas", "arrow", "record-batch"])
@pytest.mark.parametrize("timezone", [None, "UTC", "America/Los_Angeles"])
@pytest.mark.parametrize("unit", ["us", "ns"])
def test_prediction_timestamp_round_trip(tmp_path, kind, timezone, unit):
    times = pd.to_datetime([
        "2026-07-22 01:02:03.123456001",
        "2026-07-22 01:02:03.123456999",
        None,
    ]).tz_localize(timezone).as_unit(unit)
    expected = pd.DataFrame({
        "ENTITY": [1, 1, 2],
        "TIMESTAMP": times,
        "ANCHOR_TIMESTAMP": times,
        "TARGET_PRED": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    })
    job_time = pd.Timestamp("2026-07-22 02:03:04.000000777", tz="UTC")
    with StreamingPredOutputWriter(
            str(tmp_path),
            0,
            output_extra_fields={
                "PREDICTIONS": ["ANCHOR_TIMESTAMP", "JOB_TIMESTAMP"]
            },
            job_start_time=job_time,
    ) as writer:
        for frame in (expected.iloc[:1], expected.iloc[1:]):
            if kind == "pandas":
                output = PredOutput(frame, None)
            else:
                table = pa.Table.from_pandas(frame, preserve_index=False)
                output = ArrowPredOutput(
                    table.to_batches()[0] if kind == "record-batch" else table)
            writer.write(output)

    expected["JOB_TIMESTAMP"] = job_time.tz_localize(None)
    actual = _read_parts(tmp_path)
    assert actual.schema.field("ANCHOR_TIMESTAMP").type == pa.timestamp(
        unit, tz=timezone)
    pd.testing.assert_frame_equal(actual.to_pandas(), expected)
    if unit == "ns":
        assert not actual.to_pandas().duplicated(
            ["ENTITY", "ANCHOR_TIMESTAMP"]).any()


@pytest.mark.parametrize("kind", ["pandas", "arrow"])
@pytest.mark.parametrize("offset_ns", [0, 1])
def test_prediction_stream_rejects_lossy_timestamp_schema_change(
        tmp_path, kind, offset_ns):
    anchor = pd.Timestamp("2026-07-22 01:02:03.123456")
    frames = [
        pd.DataFrame({
            "ANCHOR_TIMESTAMP": pd.Series([anchor], dtype="datetime64[us]")
        }),
        pd.DataFrame({
            "ANCHOR_TIMESTAMP": pd.Series(
                [anchor + pd.Timedelta(offset_ns, unit="ns")],
                dtype="datetime64[ns]",
            )
        }),
    ]
    context = (pytest.raises(pa.ArrowInvalid, match="lose data")
               if offset_ns else nullcontext())
    with context:
        with StreamingPredOutputWriter(str(tmp_path), 0) as writer:
            for frame in frames:
                output = (PredOutput(frame, None) if kind == "pandas" else
                          ArrowPredOutput(pa.Table.from_pandas(
                              frame, preserve_index=False)))
                writer.write(output)

    if offset_ns:
        assert not list(tmp_path.rglob("part-*.parquet"))
    else:
        assert _read_parts(tmp_path).column(
            "ANCHOR_TIMESTAMP").to_pylist() == [anchor, anchor]


def test_embeddings_only_writer_creates_no_prediction_artifact(tmp_path):
    with StreamingPredOutputWriter(str(tmp_path), 0,
                                   write_predictions=False) as writer:
        writer.write(
            ArrowPredOutput(
                pa.table({}),
                pa.table({
                    "ID": ["user_a"],
                    "TYPE": ["users"],
                    "EMBEDDING": [[1.0, 2.0]],
                }),
            ))

    assert not (tmp_path / "predictions.parquet").exists()
    assert (tmp_path / "embeddings.parquet" /
            "part-00000-00000.parquet").exists()


def test_embeddings_only_writer_rejects_prediction_rows(tmp_path):
    with StreamingPredOutputWriter(str(tmp_path), 0,
                                   write_predictions=False) as writer:
        with pytest.raises(ValueError, match="were not requested"):
            writer.write(ArrowPredOutput(pa.table({"CLASS": ["item_a"]})))

    assert not (tmp_path / "predictions.parquet").exists()


def test_streaming_arrow_writer_rotates_deterministic_parts(tmp_path):
    prefix = str(tmp_path / "part")
    with StreamingFrameWriter(
            prefix,
            max_file_size=32,
            max_buffer_size=32,
            index=False,
    ) as writer:
        writer.write(pa.table({"value": np.arange(10, dtype=np.int64)}))

    paths = sorted(tmp_path.glob("part-*.parquet"))
    assert [path.name for path in paths] == [
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
    ]
    assert _read_parts(tmp_path).column("value").to_pylist() == list(range(10))


@pytest.mark.parametrize(
    "table",
    [
        pa.table({
            "ID": [f"id-{row}" if row % 11 else None for row in range(1200)],
            "CLASS": [f"class-{row % 7}" for row in range(1200)],
            "SCORE":
            np.arange(1200, dtype=np.float32),
            "TIMESTAMP":
            pa.array(np.arange(1200, dtype=np.int64), type=pa.timestamp("us")),
        }),
        pa.table({
            "ID": [f"id-{row}" for row in range(1200)],
            "TYPE": ["entity"] * 1200,
            "EMBEDDING": [
                None if row % 17 == 0 else [float(row), None,
                                            float(row + 1)]
                for row in range(1200)
            ],
        }),
    ],
    ids=["string-null-timestamp", "list-null"],
)
def test_streaming_writer_stats_match_post_parquet_prefixes(tmp_path, table):
    writer = StreamingFrameWriter(
        str(tmp_path / "part"),
        max_file_size=2048,
        max_buffer_size=1024,
        max_row_group_rows=73,
        stats_sample_rows=1000,
        index=False,
    )
    writer.write(table.slice(0, 317))
    writer.write(table.slice(317, 883))
    with pytest.raises(RuntimeError, match="successful close"):
        _ = writer.stats
    writer.close()
    assert writer._peak_row_group_bytes <= writer.max_buffer_size

    paths = sorted(tmp_path.glob("part-*.parquet"))
    dataset = ds.dataset(paths, format="parquet")
    stats = writer.stats
    assert stats.num_rows == dataset.count_rows() == 1200
    assert stats.num_cols == len(dataset.schema.names)
    assert stats.sample_rows == 1000
    for rows in (1, 8, 72, 73, 317, 999, 1000):
        assert stats.sample_prefix_logical_bytes[rows] == dataset.head(
            rows).nbytes
    assert stats.physical_size_bytes == sum(path.stat().st_size
                                            for path in paths)
    assert stats.file_count == len(paths)


def test_stats_enabled_writer_enforces_row_group_memory_ceiling(tmp_path):
    writer = StreamingFrameWriter(
        str(tmp_path / "part"),
        max_file_size=1024 * 1024 * 1024,
        max_buffer_size=512 * 1024 * 1024,
        max_row_group_rows=4 * 1024 * 1024,
        stats_sample_rows=1000,
        index=False,
    )
    try:
        assert writer.max_buffer_size == (
            PREDICTION_ARTIFACT_STATS_MAX_ROW_GROUP_BYTES)
        assert writer.max_row_group_rows == (
            PREDICTION_ARTIFACT_STATS_MAX_ROW_GROUP_ROWS)
    finally:
        writer.abort()


def test_stats_disabled_writer_preserves_legacy_limits_and_no_metadata(
        tmp_path):
    writer = StreamingFrameWriter(
        str(tmp_path / "part"),
        max_file_size=4096,
        max_buffer_size=2048,
        max_row_group_rows=2048,
        index=False,
    )
    assert writer.max_buffer_size == 2048
    assert writer.max_row_group_rows == 2048
    assert writer.collect_stats is False
    writer.write(pa.table({"value": list(range(10))}))
    writer.close()
    with pytest.raises(RuntimeError, match="successful close"):
        _ = writer.stats


def test_stats_writer_flushes_before_one_wide_row_exceeds_byte_ceiling(
        tmp_path):
    writer = StreamingFrameWriter(
        str(tmp_path / "part"),
        max_file_size=4096,
        max_buffer_size=128,
        stats_sample_rows=2,
        index=False,
    )
    row = pa.table({"value": ["x" * 90]})
    assert row.nbytes == 94

    writer.write(row)
    writer.write(row)
    writer.close()

    assert writer._peak_row_group_bytes == row.nbytes
    assert writer._peak_row_group_bytes <= writer.max_buffer_size
    assert writer.stats.file_count == 1


def test_stats_writer_accepts_row_at_exact_byte_ceiling(tmp_path):
    writer = StreamingFrameWriter(
        str(tmp_path / "part"),
        max_file_size=4096,
        max_buffer_size=128,
        stats_sample_rows=1,
        index=False,
    )
    row = pa.table({"value": ["x" * 124]})
    assert row.nbytes == writer.max_buffer_size == 128

    writer.write(row)
    writer.close()

    assert writer._peak_row_group_bytes == writer.max_buffer_size
    assert writer.stats.file_count == 1


def test_stats_writer_rejects_row_one_byte_over_ceiling(tmp_path):
    writer = StreamingFrameWriter(
        str(tmp_path / "part"),
        max_file_size=4096,
        max_buffer_size=128,
        stats_sample_rows=1,
        index=False,
    )
    row = pa.table({"value": ["x" * 125]})
    assert row.nbytes == writer.max_buffer_size + 1 == 129

    try:
        with pytest.raises(ValueError,
                           match="single normalized prediction artifact row"):
            writer.write(row)
    finally:
        writer.abort()


@pytest.mark.parametrize(("values", "expected_bytes", "succeeds"), [
    ([list(range(31))], 128, True),
    ([list(range(32))], 132, False),
])
def test_stats_writer_bounds_nested_list_rows(tmp_path, values, expected_bytes,
                                              succeeds):
    writer = StreamingFrameWriter(
        str(tmp_path / "part"),
        max_file_size=4096,
        max_buffer_size=128,
        stats_sample_rows=1,
        index=False,
    )
    row = pa.table({
        "value": pa.array(values, type=pa.list_(pa.int32())),
    })
    assert row.nbytes == expected_bytes

    if succeeds:
        writer.write(row)
        writer.close()
        assert writer._peak_row_group_bytes == 128
    else:
        try:
            with pytest.raises(
                    ValueError,
                    match="single normalized prediction artifact row"):
                writer.write(row)
        finally:
            writer.abort()


def test_two_stats_writers_bound_aggregate_raw_row_groups(tmp_path):
    writers = [
        StreamingFrameWriter(
            str(tmp_path / f"artifact-{index}"),
            max_file_size=4096,
            max_buffer_size=128,
            stats_sample_rows=2,
            index=False,
        ) for index in range(2)
    ]
    row = pa.table({"value": ["x" * 90]})
    for writer in writers:
        writer.write(row)
        writer.write(row)
    for writer in writers:
        writer.close()

    assert all(writer._peak_row_group_bytes <= writer.max_buffer_size
               for writer in writers)
    assert sum(writer._peak_row_group_bytes for writer in writers) <= 256


def test_streaming_writer_coalesces_bounded_row_groups_and_recovers_size(
        tmp_path):
    batches = []
    for _ in range(64):
        rows = 4096
        batches.append(
            pa.table({
                "USER_ID": [f"user-{row % 1024}" for row in range(rows)],
                "CLASS": [f"item-{row % 128}" for row in range(rows)],
                "SCORE": np.arange(rows, dtype=np.float32),
            }))

    baseline_path = tmp_path / "per-batch-row-groups.parquet"
    baseline_writer = pq.ParquetWriter(baseline_path, batches[0].schema)
    for batch in batches:
        baseline_writer.write_table(batch)
    baseline_writer.close()

    output_dir = tmp_path / "coalesced"
    max_buffer_size = 16 * 1024 * 1024
    writer = StreamingFrameWriter(
        str(output_dir / "part"),
        max_file_size=1024 * 1024 * 1024,
        max_buffer_size=max_buffer_size,
        max_row_group_rows=4 * rows,
        index=False,
    )
    for batch in batches:
        writer.write(batch)
    writer.close()

    output_path = next(output_dir.glob("*.parquet"))
    baseline_metadata = pq.ParquetFile(baseline_path).metadata
    output_metadata = pq.ParquetFile(output_path).metadata
    assert baseline_metadata.num_row_groups == 64
    assert output_metadata.num_row_groups == 16
    assert [
        output_metadata.row_group(index).num_rows
        for index in range(output_metadata.num_row_groups)
    ] == [4 * rows] * 16
    assert writer._peak_row_group_bytes <= max_buffer_size
    assert output_path.stat().st_size < baseline_path.stat().st_size * 0.6
    assert pq.read_table(output_path).combine_chunks().equals(
        pq.read_table(baseline_path).combine_chunks())


def test_local_publish_falls_back_safely_across_filesystems(
        monkeypatch, tmp_path):
    source = tmp_path / "spool" / "part.parquet"
    target = tmp_path / "output" / "part.parquet"
    source.parent.mkdir()
    target.parent.mkdir()
    source.write_bytes(b"new parquet")
    target.write_bytes(b"old parquet")
    real_replace = os.replace

    def cross_device_once(src, dst):
        if os.fspath(src) == os.fspath(source):
            raise OSError(errno.EXDEV, "cross-device link")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", cross_device_once)
    StreamingFrameWriter._publish_local(source, target)

    assert target.read_bytes() == b"new parquet"
    assert not source.exists()
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_spool_root_is_configurable(monkeypatch, tmp_path):
    spool = tmp_path / "bounded-emptydir"
    monkeypatch.setenv("KUMO_BP_OUTPUT_SPOOL_ROOT", str(spool))
    writer = StreamingFrameWriter(
        str(tmp_path / "output" / "part"),
        max_file_size=32,
        max_buffer_size=16,
        index=False,
    )
    staging = writer._temp_dir
    assert staging.parent == spool
    writer.write(pa.table({"value": np.arange(4, dtype=np.int64)}))
    writer.close()
    assert not staging.exists()

    with pytest.raises(ValueError, match="local filesystem"):
        StreamingFrameWriter(
            str(tmp_path / "unused" / "part"),
            max_file_size=32,
            max_buffer_size=16,
            index=False,
            spool_root="s3://not-a-local-spool",
        )


def test_spool_root_defaults_to_diskgraph_data_root(monkeypatch, tmp_path):
    diskgraph_data_root = tmp_path / "localssd"
    monkeypatch.delenv("KUMO_BP_OUTPUT_SPOOL_ROOT", raising=False)
    monkeypatch.setenv("DISKGRAPH_DATA_ROOT", str(diskgraph_data_root))

    writer = StreamingFrameWriter(
        str(tmp_path / "output" / "part"),
        max_file_size=32,
        max_buffer_size=16,
        index=False,
    )

    staging = writer._temp_dir
    assert staging.parent == diskgraph_data_root / "bp-output-spool"
    writer.close()
    assert not staging.exists()


class _ControlledPublishWriter(StreamingFrameWriter):
    def __init__(self, *args, release: threading.Event, **kwargs):
        self.release = release
        self.started = threading.Event()
        self.active = 0
        self.max_active = 0
        self.active_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def _publish(self, local_path, output_path):
        try:
            with self.active_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            self.started.set()
            assert self.release.wait(timeout=5)
            os.unlink(local_path)
        finally:
            with self.active_lock:
                self.active -= 1
            self._disk_slots.release()


def test_slow_upload_applies_bounded_backpressure(tmp_path):
    release = threading.Event()
    writer = _ControlledPublishWriter(
        str(tmp_path / "part"),
        max_file_size=8,
        max_buffer_size=8,
        index=False,
        upload_workers=1,
        release=release,
    )
    row = pa.table({"value": np.array([1], dtype=np.int64)})
    writer.write(row)
    assert writer.started.wait(timeout=2)
    writer.write(row)
    deadline = time.monotonic() + 2
    while writer.files_written < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer.files_written == 2
    writer.write(row)
    time.sleep(0.05)  # serializer is waiting for a bounded local-file slot
    writer.write(row)
    blocked = threading.Event()

    def write_one_more():
        writer.write(row)
        blocked.set()

    thread = threading.Thread(target=write_one_more)
    thread.start()
    assert not blocked.wait(timeout=0.1)
    release.set()
    assert blocked.wait(timeout=2)
    thread.join()
    writer.close()
    assert writer._peak_row_group_bytes <= writer.max_buffer_size


def test_completed_parts_upload_concurrently(tmp_path):
    release = threading.Event()
    writer = _ControlledPublishWriter(
        str(tmp_path / "part"),
        max_file_size=8,
        max_buffer_size=32,
        index=False,
        upload_workers=2,
        release=release,
    )
    writer.write(pa.table({"value": np.arange(4, dtype=np.int64)}))
    deadline = time.monotonic() + 2
    while writer.max_active < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer.max_active == 2
    release.set()
    writer.close()
    assert not writer._serializer.is_alive()
    assert writer.active == 0


class _CredentialCheckingWriter(StreamingFrameWriter):
    def __init__(self, *args, credential, **kwargs):
        self.credential = credential
        self.serializer_credential = None
        self.publisher_credentials = []
        self.publisher_threads = set()
        self.publish_barrier = threading.Barrier(2)
        super().__init__(*args, **kwargs)

    def _serialize(self):
        self.serializer_credential = self.credential.get()
        super()._serialize()

    def _publish(self, local_path, output_path):
        self.publisher_credentials.append(self.credential.get())
        self.publisher_threads.add(threading.get_ident())
        self.publish_barrier.wait(timeout=5)
        super()._publish(local_path, output_path)


def test_scoped_filesystem_credential_reaches_all_background_threads(tmp_path):
    credential = contextvars.ContextVar("fake_filesystem_credential")
    token = credential.set("request-scoped-secret")
    try:
        writer = _CredentialCheckingWriter(
            str(tmp_path / "part"),
            max_file_size=8,
            max_buffer_size=32,
            index=False,
            upload_workers=2,
            credential=credential,
        )
    finally:
        credential.reset(token)

    writer.write(pa.table({"value": np.arange(4, dtype=np.int64)}))
    writer.close()

    assert writer.serializer_credential == "request-scoped-secret"
    assert writer.publisher_credentials == ["request-scoped-secret"] * 4
    assert len(writer.publisher_threads) == 2


class _FailingPublishWriter(StreamingFrameWriter):
    def __init__(self, *args, fail_at: int, **kwargs):
        self.fail_at = fail_at
        self.publish_count = 0
        self.publish_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def _publish(self, local_path, output_path):
        try:
            with self.publish_lock:
                index = self.publish_count
                self.publish_count += 1
            time.sleep(0.01)
            if index == self.fail_at:
                error = OSError("injected concurrent publisher failure")
                self._set_error(error)
                raise error
            os.unlink(local_path)
        finally:
            self._disk_slots.release()


def test_failing_publisher_unblocks_producer_and_shutdown(tmp_path):
    writer = _FailingPublishWriter(
        str(tmp_path / "part"),
        max_file_size=8,
        max_buffer_size=16,
        index=False,
        upload_workers=2,
        fail_at=2,
    )
    row = pa.table({"value": np.array([1], dtype=np.int64)})
    producer_error = []

    def produce():
        try:
            for _ in range(100):
                writer.write(row)
        except Exception as error:
            producer_error.append(error)

    producer = threading.Thread(target=produce)
    producer.start()
    producer.join(timeout=3)
    assert not producer.is_alive()
    assert producer_error
    with pytest.raises(RuntimeError, match="close Parquet output"):
        writer.close()
    assert not writer._serializer.is_alive()
    assert not writer._temp_dir.exists()


def test_abort_wakes_backpressured_producer_and_drains_slow_upload(tmp_path):
    release = threading.Event()
    writer = _ControlledPublishWriter(
        str(tmp_path / "part"),
        max_file_size=8,
        max_buffer_size=8,
        index=False,
        upload_workers=1,
        release=release,
    )
    row = pa.table({"value": np.array([1], dtype=np.int64)})
    producer_done = threading.Event()

    def produce():
        with pytest.raises(RuntimeError, match="aborted"):
            for _ in range(100):
                writer.write(row)
        producer_done.set()

    producer = threading.Thread(target=produce)
    producer.start()
    assert writer.started.wait(timeout=2)
    aborter = threading.Thread(target=writer.abort)
    aborter.start()
    assert producer_done.wait(timeout=2)
    release.set()
    aborter.join(timeout=2)
    producer.join(timeout=2)
    assert not aborter.is_alive()
    assert not producer.is_alive()
    assert not writer._serializer.is_alive()
    assert not writer._temp_dir.exists()


def test_async_upload_failure_propagates_and_cleans_staging(
        monkeypatch, tmp_path):
    from kumo_core.train import io as train_io

    real_open = train_io.open_file

    def fail_remote(path, mode, **kwargs):
        if path.startswith("memory://") and "w" in mode:
            raise OSError("injected upload failure")
        return real_open(path, mode, **kwargs)

    monkeypatch.setattr(train_io, "open_file", fail_remote)
    writer = StreamingFrameWriter(
        "memory://bp-output/part",
        max_file_size=8,
        max_buffer_size=8,
        index=False,
    )
    staging = writer._temp_dir
    table = pa.table({"value": np.array([1], dtype=np.int64)})
    writer.write(table)
    deadline = time.monotonic() + 2
    while writer._error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(RuntimeError, match="Asynchronous Parquet output"):
        writer.write(table)
    with pytest.raises(RuntimeError, match="close Parquet output"):
        writer.close()
    with pytest.raises(RuntimeError, match="successful close"):
        _ = writer.stats
    assert not staging.exists()
    assert not train_io.filesystem.exists(
        "memory://bp-output/part-00000.parquet")


def test_partition_retry_removes_only_stale_partition_parts(tmp_path):
    prediction_dir = tmp_path / "predictions.parquet"
    prediction_dir.mkdir()
    stale = prediction_dir / "part-00003-00000.parquet"
    stale_publish_temp = prediction_dir / \
        ".part-00003-00000.parquet.deadbeef.tmp"
    other = prediction_dir / "part-00004-00000.parquet"
    stale.touch()
    stale_publish_temp.touch()
    other.touch()
    with StreamingPredOutputWriter(str(tmp_path), 3):
        pass
    assert not stale.exists()
    assert not stale_publish_temp.exists()
    assert other.exists()


def test_abort_removes_parts_even_after_writer_was_closed(tmp_path):
    writer = StreamingPredOutputWriter(
        str(tmp_path),
        3,
        max_size_per_file=8,
        max_buffer_size=8,
    )
    writer.write(
        ArrowPredOutput(pa.table({"value": np.array([1], dtype=np.int64)})))
    writer.close()
    assert list(tmp_path.rglob("part-00003-*.parquet"))

    writer.abort()

    assert not list(tmp_path.rglob("part-00003-*.parquet"))


def test_embedding_close_failure_aborts_prediction_partition(
        monkeypatch, tmp_path):
    writer = StreamingPredOutputWriter(
        str(tmp_path),
        3,
        max_size_per_file=8,
        max_buffer_size=8,
    )
    writer.write(
        ArrowPredOutput(
            pa.table({"value": np.array([1], dtype=np.int64)}),
            pa.table({
                "ID": ["a"],
                "EMBEDDING": [[1.0]]
            }),
        ))
    assert writer._embedding_writer is not None

    def fail_close():
        raise OSError("injected embedding close failure")

    monkeypatch.setattr(writer._embedding_writer, "close", fail_close)

    with pytest.raises(OSError, match="embedding close failure"):
        writer.close()

    assert not list(tmp_path.rglob("part-00003-*.parquet"))
    assert not writer._prediction_writer._temp_dir.exists()
    assert not writer._embedding_writer._temp_dir.exists()


@pytest.mark.parametrize("invalid_id", [None, 3, 99])
@pytest.mark.parametrize("first_key", ["rhs-0", 1])
def test_ttng_lp_callback_builds_arrow_without_pandas_dataframe(
        monkeypatch, invalid_id, first_key):
    from kumo_core.train import diskgraph_nested_trainer as trainer_module

    dataframe_type = pd.DataFrame
    expected = pd.DataFrame({
        "USER_ID": ["lhs-a", "lhs-a", "lhs-b", "lhs-b"],
        "CLASS": [None, str(first_key), "rhs-2", None],
        "SCORE":
        np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        "EVENT_TIME":
        np.repeat(
            np.array([1_700_000_000_000_000_000, 1_700_000_001_000_000_000],
                     dtype=np.int64).astype("datetime64[ns]"),
            2),
    })

    def reject_dataframe(*args, **kwargs):
        raise AssertionError("TT-NG LP output must not materialize pandas")

    monkeypatch.setattr(trainer_module.pd, "DataFrame", reject_dataframe)
    context = (pytest.raises(RuntimeError, match="non-canonical RHS node ID")
               if invalid_id is not None else nullcontext())
    with context:
        output = trainer_module._build_ttng_lp_chunk_output(
            {
                "pkey":
                np.array(["lhs-a", "lhs-b"], dtype=object),
                "target":
                np.array([[1, 0], [2, -1 if invalid_id is None else invalid_id]
                          ], dtype=np.int64),
                "score":
                np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
                "time":
                np.array([1_700_000_000_000_000_000, 1_700_000_001_000_000_000],
                         dtype=np.int64),
            },
            rhs_node_ids=np.arange(3, dtype=np.int64),
            rhs_pk_arr=np.array([first_key, None, "rhs-2"], dtype=object),
            rhs_pk_mixed=isinstance(first_key, int),
            rhs_pk_type=pa.string(),
            lhs_fkey_name="USER_ID",
            time_col="EVENT_TIME",
            ignore_time_col=False,
            job_start_time=None,
            emit_predictions=True,
            emit_embeddings=False,
            lhs_table="users",
            rhs_table="items",
            embedding_extra_fields=set(),
            embedding_ids_as_string=True,
        )
    if invalid_id is not None:
        return

    assert isinstance(output, ArrowPredOutput)
    assert output.table.column_names == [
        "USER_ID", "CLASS", "SCORE", "EVENT_TIME"
    ]
    assert output.table.column("USER_ID").to_pylist() == [
        "lhs-a", "lhs-a", "lhs-b", "lhs-b"
    ]
    assert output.table.column("CLASS").to_pylist() == [
        None, str(first_key), "rhs-2", None
    ]
    assert output.table.schema.field("EVENT_TIME").type == pa.timestamp("ns")
    monkeypatch.setattr(trainer_module.pd, "DataFrame", dataframe_type)
    pd.testing.assert_frame_equal(output.table.to_pandas(), expected)


def test_arrow_string_conversion_preserves_null_and_mixed_semantics():
    from kumo_core.train import diskgraph_nested_trainer as trainer_module

    strings = np.array(["alpha", None, pd.NA, "omega"], dtype=object)
    assert trainer_module._arrow_string_array(strings).to_pylist() == [
        "alpha", None, None, "omega"
    ]

    integers = np.array([1, None, 3], dtype=object)
    assert trainer_module._arrow_string_array(integers).to_pylist() == [
        "1", None, "3"
    ]

    mixed = np.array([1, "two", None, np.nan, b"three"], dtype=object)
    assert trainer_module._arrow_string_array(mixed).to_pylist() == [
        "1", "two", None, None, "b'three'"
    ]


def test_ttng_lp_object_integer_class_matches_legacy_artifact(tmp_path):
    from kumo_core.train import diskgraph_nested_trainer as trainer_module

    arrow_output = trainer_module._build_ttng_lp_chunk_output(
        {
            "pkey": np.array(["lhs-a", "lhs-b", "lhs-c"], dtype=object),
            "target": np.array([[0], [1], [2]], dtype=np.int64),
            "score": np.array([[0.9], [0.8], [0.7]], dtype=np.float32),
        },
        rhs_node_ids=np.arange(3, dtype=np.int64),
        rhs_pk_arr=np.array([10, None, 30], dtype=object),
        rhs_pk_type=pa.int64(),
        lhs_fkey_name="USER_ID",
        time_col="EVENT_TIME",
        ignore_time_col=True,
        job_start_time=None,
        emit_predictions=True,
        emit_embeddings=False,
        lhs_table="users",
        rhs_table="items",
        embedding_extra_fields=set(),
        embedding_ids_as_string=False,
    )
    legacy_output = PredOutput(
        pd.DataFrame({
            "USER_ID": ["lhs-a", "lhs-b", "lhs-c"],
            "CLASS": pd.Series([10, None, 30], dtype=object),
            "SCORE": np.array([0.9, 0.8, 0.7], dtype=np.float32),
        }))

    legacy_dir = tmp_path / "legacy"
    arrow_dir = tmp_path / "arrow"
    legacy_output.save(str(legacy_dir))
    with StreamingPredOutputWriter(str(arrow_dir), 0) as writer:
        writer.write(arrow_output)

    legacy = _read_parts(legacy_dir / "predictions.parquet")
    arrow = _read_parts(arrow_dir / "predictions.parquet")
    assert arrow.schema.field("CLASS").type == pa.int64()
    assert arrow.schema.equals(legacy.schema, check_metadata=False)
    pd.testing.assert_frame_equal(arrow.to_pandas(), legacy.to_pandas())


def test_ttng_lp_arrow_embedding_matches_legacy_schema(tmp_path):
    from kumo_core.train import diskgraph_nested_trainer as trainer_module

    prediction = {
        "pkey":
        np.array(["lhs-a", "lhs-b"], dtype=object),
        "target":
        np.array([[0], [1]], dtype=np.int64),
        "score":
        np.array([[0.9], [0.8]], dtype=np.float32),
        "time":
        np.array([1_700_000_000_000_000_000, 1_700_000_001_000_000_000],
                 dtype=np.int64),
        "lhs_embedding":
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }
    output = trainer_module._build_ttng_lp_chunk_output(
        prediction,
        rhs_node_ids=np.arange(2, dtype=np.int64),
        rhs_pk_arr=np.array(["rhs-0", "rhs-1"]),
        rhs_pk_type=pa.string(),
        lhs_fkey_name="USER_ID",
        time_col="EVENT_TIME",
        ignore_time_col=False,
        job_start_time=None,
        emit_predictions=True,
        emit_embeddings=True,
        lhs_table="users",
        rhs_table="items",
        embedding_extra_fields={"ANCHOR_TIMESTAMP"},
        embedding_ids_as_string=True,
    )
    assert output.embedding is not None
    legacy = pd.DataFrame({
        "ID": ["lhs-a", "lhs-b"],
        "TYPE": ["users", "users"],
        "EMBEDDING": [row for row in prediction["lhs_embedding"]],
        "ANCHOR_TIMESTAMP":
        prediction["time"].astype("datetime64[ns]"),
    })
    legacy_schema = pa.Table.from_pandas(legacy, preserve_index=False).schema
    assert output.embedding.schema.equals(legacy_schema, check_metadata=False)

    with StreamingPredOutputWriter(str(tmp_path), 0) as writer:
        writer.write(output)
    embedding_path = next((tmp_path / "embeddings.parquet").glob("*.parquet"))
    assert pq.read_schema(embedding_path).field(
        "ANCHOR_TIMESTAMP").type == pa.timestamp("ns")


@pytest.mark.parametrize("targets", [
    [[0, 1], [2, -1]],
    [[-1, -1], [-1, -1]],
    [[0, 1], [2, 0]],
])
@pytest.mark.parametrize(("rhs_keys", "class_dtype"), [
    (["rhs-0", "rhs-1", "rhs-2"], None),
    ([2**60 + 1, 2**60 + 2, 2**60 + 3], pd.Int64Dtype()),
    ([2**63 + 1, 2**63 + 2, 2**63 + 3], pd.UInt64Dtype()),
])
def test_arrow_lp_artifacts_match_legacy_values_schema_and_metadata(
        tmp_path, targets, rhs_keys, class_dtype):
    import torch
    from kumo_core.train import diskgraph_nested_trainer as trainer_module
    from kumo_core.train.mapping import NODE_INDEX_COLUMN
    from kumo_core.train.trainer import create_lp_prediction_output

    time_values = np.array(
        [1_700_000_000_123_456_789, 1_700_000_001_987_654_321],
        dtype=np.int64,
    )
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    job_start = pd.Timestamp("2026-08-14T12:34:56Z")
    arrow_output = trainer_module._build_ttng_lp_chunk_output(
        {
            "pkey": np.array(["lhs-a", "lhs-b"], dtype=object),
            "target": np.array(targets, dtype=np.int64),
            "score": np.array([[0., 0.8], [0.7, 0.]], dtype=np.float32),
            "time": time_values,
            "lhs_embedding": embeddings,
        },
        rhs_node_ids=np.arange(3, dtype=np.int64),
        rhs_pk_arr=np.asarray(rhs_keys),
        rhs_pk_type=pa.infer_type(np.asarray(rhs_keys)),
        lhs_fkey_name="USER_ID",
        time_col="EVENT_TIME",
        ignore_time_col=False,
        job_start_time=job_start,
        emit_predictions=True,
        emit_embeddings=True,
        lhs_table="users",
        rhs_table="items",
        embedding_extra_fields={"ANCHOR_TIMESTAMP", "JOB_TIMESTAMP"},
        embedding_ids_as_string=True,
        class_metadata=(pa.Table.from_pandas(
            pd.DataFrame({'CLASS': pd.Series(
                dtype=class_dtype)}), preserve_index=False).schema.metadata
                        if class_dtype is not None else None),
    )
    target_tensor = torch.tensor(targets)
    legacy_frame = create_lp_prediction_output(
        outs={
            "pkey": np.array(["lhs-a", "lhs-b"]),
            "target": target_tensor,
            "score": torch.tensor([[0., 0.8], [0.7, 0.]]),
        },
        lhs_fkey_name="USER_ID",
        rhs_index_map=pd.DataFrame({NODE_INDEX_COLUMN: range(3)},
                                   index=rhs_keys),
    )
    assert target_tensor.tolist() == targets
    expected_classes = [
        rhs_keys[i] if i >= 0 else None for row in targets for i in row
    ]
    assert arrow_output.table.column("CLASS").to_pylist() == expected_classes
    assert arrow_output.table.column("SCORE").to_pylist()[0] == 0.
    legacy_frame["EVENT_TIME"] = np.repeat(time_values.astype("datetime64[ns]"),
                                           2)
    legacy_output = PredOutput(
        legacy_frame,
        pd.DataFrame({
            "ID": ["lhs-a", "lhs-b"],
            "TYPE": ["users", "users"],
            "EMBEDDING": [row for row in embeddings],
            "ANCHOR_TIMESTAMP": time_values.astype("datetime64[ns]"),
        }),
    )
    extra_fields = {
        "PREDICTIONS": ["JOB_TIMESTAMP"],
        "EMBEDDINGS": ["ANCHOR_TIMESTAMP", "JOB_TIMESTAMP"],
    }
    legacy_dir = tmp_path / "legacy"
    arrow_dir = tmp_path / "arrow"
    with StreamingPredOutputWriter(
            str(legacy_dir),
            0,
            output_extra_fields=extra_fields,
            job_start_time=job_start,
    ) as writer:
        writer.write(legacy_output)
        if all(i < 0 for row in targets for i in row):
            following = legacy_frame.copy()
            following["CLASS"] = pd.array([rhs_keys[0]] * 4,
                                          dtype=legacy_frame.CLASS.dtype)
            writer.write(PredOutput(following))
    with StreamingPredOutputWriter(
            str(arrow_dir),
            0,
            output_extra_fields=extra_fields,
            job_start_time=job_start,
    ) as writer:
        writer.write(arrow_output)
        if all(i < 0 for row in targets for i in row):
            writer.write(
                ArrowPredOutput(
                    arrow_output.table.set_column(
                        1, "CLASS",
                        pa.array(np.repeat(np.asarray(rhs_keys)[:1], 4)))))
            expected_classes += [rhs_keys[0]] * 4

    for artifact in ("predictions.parquet", "embeddings.parquet"):
        legacy = _read_parts(legacy_dir / artifact)
        arrow = _read_parts(arrow_dir / artifact)
        assert arrow.num_rows == legacy.num_rows
        assert arrow.schema.equals(legacy.schema, check_metadata=False)
        assert arrow.equals(legacy, check_metadata=False)
        if artifact == "predictions.parquet":
            assert arrow.column("CLASS").to_pylist() == expected_classes
            for directory in (legacy_dir, arrow_dir):
                classes = pd.read_parquet(directory / artifact)['CLASS']
                assert classes.dropna().tolist() == [
                    value for value in expected_classes if value is not None
                ]
                if class_dtype is not None:
                    assert classes.dtype == class_dtype
