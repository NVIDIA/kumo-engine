# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming Arrow batches to Parquet must lose no row and outgrow no bound."""

import threading
import time
from datetime import datetime, timedelta
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.arrow_parquet import (
    count_rows,
    materialize_arrow_streams,
    write_parquet_file,
)
from kumo_core.sources.errors import SchemaChangedError

MEGABYTE = 1024 * 1024
PROMPTLY = 5


def sample_table(rows: int = 300, start: int = 0) -> pa.Table:
    epoch = datetime(2026, 1, 1)
    return pa.table({
        "id":
        pa.array(
            [None if i % 7 == 0 else i for i in range(start, start + rows)],
            pa.int64()),
        "name":
        pa.array([
            None if i % 5 == 0 else f"name-{i}"
            for i in range(start, start + rows)
        ], pa.string()),
        "score":
        pa.array([i * 0.5 for i in range(start, start + rows)], pa.float64()),
        "active":
        pa.array([i % 2 == 0 for i in range(start, start + rows)], pa.bool_()),
        "seen_at":
        pa.array(
            [epoch + timedelta(seconds=i) for i in range(start, start + rows)],
            pa.timestamp("us")),
    })


def skewed_table(rows: int = 200, wide_every: int = 25) -> pa.Table:
    return pa.table({
        "id":
        pa.array(range(rows), pa.int64()),
        "payload":
        pa.array([
            "x" * (500_000 if i % wide_every == 0 else 8) for i in range(rows)
        ], pa.string()),
    })


def destination(tmp_path, name: str = "orders"):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    return directory


def materialize(batches, directory, **limits):
    return materialize_arrow_streams([batches], str(directory), **limits)


def shards_of(result) -> list[pa.Table]:
    return [pq.read_table(path) for path in result.paths]


def read_back(result) -> pa.Table:
    return pa.concat_tables(shards_of(result))


class TestEveryRowArrives:
    def test_batches_arrive_in_order_with_values_types_and_nulls_intact(
            self, tmp_path):
        batches = [sample_table(100, start=n * 100) for n in range(3)]
        expected = pa.concat_tables(batches)
        assert expected.column("name").null_count > 0

        result = materialize(batches, destination(tmp_path))

        assert result.rows == expected.num_rows
        assert result.uncompressed_bytes == sum(b.nbytes for b in batches)
        assert read_back(result).equals(expected)

    def test_no_row_is_lost_when_the_row_bound_forces_many_shards(
            self, tmp_path):
        table = sample_table()

        result = materialize([table], destination(tmp_path),
                             max_rows_per_shard=7)

        assert len(result.paths) > 1
        assert read_back(result).equals(table)


class TestByteBound:
    def test_a_skewed_table_is_bounded_without_being_split_to_single_rows(
            self, tmp_path):
        table = skewed_table()
        assert table.nbytes > 3 * MEGABYTE

        result = materialize([table], destination(tmp_path),
                             max_uncompressed_bytes_per_shard=MEGABYTE)

        shards = shards_of(result)
        assert len(shards) > 1
        assert any(shard.num_rows > 1 for shard in shards)
        for shard in shards:
            assert shard.num_rows == 1 or shard.nbytes <= MEGABYTE
        assert read_back(result).equals(table)

    def test_a_row_larger_than_the_byte_bound_is_emitted_alone(self, tmp_path):
        table = pa.table({
            "id":
            pa.array([0, 1, 2], pa.int64()),
            "payload":
            pa.array(["small", "x" * (4 * MEGABYTE), "small"]),
        })

        result = materialize([table], destination(tmp_path),
                             max_uncompressed_bytes_per_shard=MEGABYTE)

        oversized = [
            shard for shard in shards_of(result) if shard.nbytes > MEGABYTE
        ]
        assert [shard.num_rows for shard in oversized] == [1]
        assert read_back(result).equals(table)


class TestSchemaSurvival:
    def test_an_empty_result_still_writes_the_schema_and_only_the_schema(
            self, tmp_path):
        empty = sample_table(0)
        real = sample_table(50)

        alone = materialize([empty], destination(tmp_path, "alone"))
        beside = materialize([empty, real, empty],
                             destination(tmp_path, "beside"))

        assert (alone.rows, len(alone.paths)) == (0, 1)
        written = pq.read_table(alone.paths[0])
        assert written.num_rows == 0
        assert written.schema.equals(empty.schema)
        assert read_back(beside).equals(real)
        assert all(shard.num_rows > 0 for shard in shards_of(beside))

    def test_a_schema_change_between_batches_is_refused(self, tmp_path):
        """Each shard is written alone, so a change yields a mixed directory."""
        first = pa.table({"id": pa.array([1], pa.int64())})
        second = pa.table({
            "id": pa.array([2], pa.int64()),
            "extra": pa.array(["x"], pa.string())
        })
        with pytest.raises(SchemaChangedError):
            materialize([first, second], destination(tmp_path, "out"))


class TestDestination:
    def test_a_destination_the_caller_never_prepared_is_refused(self, tmp_path):
        """``run_prepare`` empties it, so its absence means the read is
        writing somewhere nothing will be ingested from."""
        with pytest.raises(ValueError):
            materialize([sample_table()], tmp_path / "never-made")

    @pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
    def test_an_interrupted_write_leaves_no_destination_behind(
            self, tmp_path, failure):
        directory = destination(tmp_path)

        def batches():
            yield sample_table()
            raise failure()

        with pytest.raises(failure):
            materialize(batches(), directory)

        assert not directory.exists()


class TestBatchKinds:
    def test_a_record_batch_stream_matches_an_equivalent_table_stream(
            self, tmp_path):
        table = sample_table(100)
        record_batches = table.to_batches(max_chunksize=30)
        assert len(record_batches) > 1

        streamed = materialize(record_batches,
                               destination(tmp_path, "from-batches"))

        assert streamed.rows == table.num_rows
        assert read_back(streamed).equals(table)

    def test_a_non_arrow_batch_is_refused_and_writes_nothing(self, tmp_path):
        directory = destination(tmp_path)

        with pytest.raises(TypeError):
            materialize([[{"id": 1}]], directory)

        assert not directory.exists()


@pytest.mark.parametrize("limits", [{
    "max_rows_per_shard": 0
}, {
    "max_uncompressed_bytes_per_shard": 0
}])
def test_an_impossible_shard_limit_is_refused(tmp_path, limits):
    with pytest.raises(ValueError):
        materialize([sample_table()], destination(tmp_path), **limits)


@pytest.mark.parametrize("rows,bound,files", [(500, None, 1), (10, 3, 4)])
def test_small_batches_coalesce_but_never_past_the_row_bound(
        tmp_path, rows, bound, files):
    batches = [pa.table({"id": pa.array([i], pa.int64())}) for i in range(rows)]
    limit = {"max_rows_per_shard": bound} if bound else {}
    result = materialize(batches, destination(tmp_path, "out"), **limit)
    assert result.rows == rows
    assert len(result.paths) == files


def test_the_rows_written_are_counted_from_the_footers(tmp_path):
    result = materialize([sample_table(120)], destination(tmp_path),
                         max_rows_per_shard=50)
    assert len(result.paths) == 3
    assert count_rows(result.paths) == 120
    assert count_rows(()) == 0


class TestParallelStreams:
    @pytest.mark.parametrize("ceiling", [None, 1, 4])
    def test_streams_carry_the_same_rows_and_schema_as_one_producer(
            self, tmp_path, ceiling):
        batches = [sample_table(100, start=n * 100) for n in range(6)]

        parallel = str(destination(tmp_path, f"parallel-{ceiling}"))
        one = materialize(batches, destination(tmp_path, f"single-{ceiling}"))
        many = materialize_arrow_streams(
            [batches[:2], batches[2:4], batches[4:]], parallel,
            worker_ceiling=ceiling)

        assert (many.rows, many.uncompressed_bytes) == (one.rows,
                                                        one.uncompressed_bytes)
        assert read_back(many).equals(read_back(one))
        assert read_back(many).schema.equals(read_back(one).schema)

    def test_every_part_is_its_own_file_and_no_stream_overwrites_another(
            self, tmp_path):
        directory = destination(tmp_path)
        streams = [[sample_table(50, start=n * 50)] for n in range(4)]

        result = materialize_arrow_streams(streams, str(directory),
                                           max_rows_per_shard=20)

        assert len(result.paths) > len(streams)
        assert len(set(result.paths)) == len(result.paths)
        assert sorted(result.paths) == sorted(
            str(path) for path in directory.iterdir())

    def test_a_failing_stream_propagates_its_own_error_and_leaves_nothing(
            self, tmp_path):
        directory = destination(tmp_path)

        def failing():
            yield sample_table(10)
            raise RuntimeError("read stream was reset")

        def slow():
            for n in range(8):
                time.sleep(0.005)
                yield sample_table(10, start=n * 10)

        with pytest.raises(RuntimeError, match="read stream was reset"):
            materialize_arrow_streams(
                [slow(), failing(), slow()], str(directory))

        assert not directory.exists()

    def test_a_late_failure_is_not_held_behind_a_stream_still_producing(
            self, tmp_path, monkeypatch):
        """Collected in submission order, this failure is not seen until the
        blocked stream ahead of it ends, and every stream queued behind it is
        drained in the meantime -- the warehouse serving results nobody will
        read, to reach an answer already known."""
        from kumo_core.sources import cpu

        # Exercise two concurrent producers even on a one-CPU developer
        # machine. A worker ceiling alone cannot raise the detected budget.
        monkeypatch.setattr(cpu, "cpu_budget", lambda: 2)
        directory = destination(tmp_path)
        released = threading.Event()
        drained = []

        def blocked():
            yield sample_table(10)
            released.wait(timeout=10)
            yield sample_table(10, start=10)

        def failing():
            yield sample_table(10)
            raise RuntimeError("read stream was reset")

        def queued(index):
            drained.append(index)
            yield sample_table(10, start=index * 10)

        started = time.monotonic()
        try:
            with pytest.raises(RuntimeError, match="read stream was reset"):
                materialize_arrow_streams(
                    [blocked(),
                     failing(), *(queued(n) for n in range(8))], str(directory),
                    worker_ceiling=2)
            assert time.monotonic() - started < PROMPTLY
            assert len(drained) < 8
            assert not directory.exists()
        finally:
            released.set()

    def test_streams_that_disagree_on_schema_are_refused(self, tmp_path):
        """One stream alone cannot see a change another stream made."""
        directory = destination(tmp_path)
        first = pa.table({"id": pa.array([1], pa.int64())})
        renamed = pa.table({"id_": pa.array([2], pa.int64())})

        with pytest.raises(SchemaChangedError):
            materialize_arrow_streams([[first], [renamed]], str(directory))

        assert not directory.exists()

    def test_empty_streams_keep_the_schema_and_no_streams_write_nothing(
            self, tmp_path):
        empty = sample_table(0)

        schema_only = materialize_arrow_streams(
            [[empty], [empty]], str(destination(tmp_path, "empty")))
        nothing = materialize_arrow_streams([],
                                            str(destination(tmp_path, "none")))

        assert (schema_only.rows, len(schema_only.paths)) == (0, 1)
        assert pq.read_table(schema_only.paths[0]).schema.equals(empty.schema)
        assert (nothing.rows, nothing.paths) == (0, ())


@pytest.mark.parametrize("rows", [0, 17, 1_048_577])
def test_single_file_writes_zstd_with_dictionaries(rows):
    table = pa.table({"value": pa.repeat(pa.scalar("value"), rows)})
    previous, shared = BytesIO(), BytesIO()
    pq.write_table(table, previous, compression="zstd", compression_level=1,
                   use_dictionary=True)
    assert write_parquet_file((table, ), shared) == rows
    assert not shared.closed
    assert shared.getvalue() == previous.getvalue()


def test_single_file_writes_before_requesting_the_next_batch():
    output = BytesIO()
    batch = pa.record_batch({"id": [1, 2]})

    def batches():
        yield batch
        assert output.tell() > 4
        yield pa.Table.from_batches([batch])

    assert write_parquet_file(batches(), output) == 4
    assert pq.read_table(pa.BufferReader(output.getvalue())).to_pydict() == {
        "id": [1, 2, 1, 2]
    }


@pytest.mark.parametrize("supply_schema", [False, True])
def test_single_file_empty_iterator_only_writes_with_schema(supply_schema):
    output = BytesIO()
    schema = pa.schema([("id", pa.int64())])
    assert write_parquet_file((), output,
                              schema=schema if supply_schema else None) == 0
    assert not output.closed
    if supply_schema:
        table = pq.read_table(pa.BufferReader(output.getvalue()))
        assert table.schema == schema
        assert table.num_rows == 0
    else:
        assert output.getvalue() == b""


@pytest.mark.parametrize("failure", ["source", "schema"])
def test_single_file_finalizes_writer_without_owning_output(failure):
    output = BytesIO()

    def batches():
        yield pa.record_batch({"id": [1, 2]})
        if failure == "source":
            raise RuntimeError("source unavailable")
        yield pa.record_batch({"other": [1, 2]})

    with pytest.raises((RuntimeError, ValueError)):
        write_parquet_file(batches(), output)
    assert not output.closed
    assert pq.read_table(pa.BufferReader(output.getvalue())).to_pydict() == {
        "id": [1, 2]
    }
