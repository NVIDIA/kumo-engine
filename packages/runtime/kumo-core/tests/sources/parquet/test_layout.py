# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Table boundaries and resolution, which decide what a name even means.

Ported from the hosted file connector, so these cases are the contract two
implementations have to agree on rather than this one's preferences.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.sources import layout
from kumo_core.sources.errors import PermanentSourceError, SourceNotFoundError
from kumo_core.sources.layout import staging_dir
from pyarrow.fs import FileInfo, FileType, LocalFileSystem

FRAME = pa.table({"id": pa.array([1, 2, 3], pa.int64())})


def write(root, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if relative.endswith(".parquet"):
        pq.write_table(FRAME, path)
    else:
        path.write_text("id\n1\n2\n3\n", encoding="utf-8")


def relatives(root) -> list[str]:
    return layout.scan(LocalFileSystem(), str(root))


def tables(root) -> list[str]:
    return layout.discover_tables(relatives(root))


def test_a_flat_file_is_a_table_named_without_its_suffix(tmp_path):
    write(tmp_path, "customer.parquet")
    write(tmp_path, "stock.csv")
    assert tables(tmp_path) == ["customer", "stock"]


def test_a_directory_of_parts_is_one_table(tmp_path):
    write(tmp_path, "events/part-0.parquet")
    write(tmp_path, "events/part-1.parquet")
    assert tables(tmp_path) == ["events"]


def test_a_nested_directory_keeps_its_whole_relative_name(tmp_path):
    write(tmp_path, "warehouse/gold/events/part-0.parquet")
    assert tables(tmp_path) == ["warehouse/gold/events"]


def test_hive_partitions_collapse_to_the_root_above_them(tmp_path):
    write(tmp_path, "events/dt=2026-01-01/region=eu/part-0.parquet")
    write(tmp_path, "events/dt=2026-01-02/region=us/part-0.parquet")
    assert tables(tmp_path) == ["events"]


def test_spark_and_dot_prefixed_paths_are_skipped(tmp_path):
    write(tmp_path, "events/part-0.parquet")
    (tmp_path / "events" / "_SUCCESS").write_text("", encoding="utf-8")
    write(tmp_path, "_staging/part-0.parquet")
    write(tmp_path, ".backup/part-0.parquet")
    assert tables(tmp_path) == ["events"]


def test_a_directory_holding_another_dataset_is_not_itself_a_table(tmp_path):
    """The leaf wins, and the parts beside it are left unaddressable: a
    directory that is both a dataset and a parent of one is ambiguous, and the
    hosted connector resolves it this way."""
    write(tmp_path, "events/part-0.parquet")
    write(tmp_path, "events/nested/part-0.parquet")
    assert tables(tmp_path) == ["events/nested"]


def test_a_directory_mixing_formats_is_not_a_table(tmp_path):
    write(tmp_path, "mixed/part-0.parquet")
    write(tmp_path, "mixed/part-1.csv")
    assert tables(tmp_path) == []


def test_discovery_is_sorted(tmp_path):
    for name in ("zeta", "alpha", "middle"):
        write(tmp_path, f"{name}/part-0.parquet")
    assert tables(tmp_path) == ["alpha", "middle", "zeta"]


def test_missing_root_advice_follows_the_filesystem(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(SourceNotFoundError) as error:
        layout.scan(LocalFileSystem(), str(missing))
    message = str(error.value)
    assert str(missing) in message
    assert "local directory" in message
    assert "read and traverse" in message
    assert "bucket" not in message and "prefix" not in message

    class MissingObjectStore:
        def get_file_info(self, path):
            return FileInfo(path, FileType.NotFound)

    with pytest.raises(SourceNotFoundError,
                       match="Check the bucket and prefix"):
        layout.scan(MissingObjectStore(), "bucket/missing")


def test_more_objects_than_the_scan_ceiling_asks_for_a_narrower_prefix(
        tmp_path):
    for index in range(21):
        write(tmp_path, f"events/part-{index}.parquet")
    with pytest.raises(PermanentSourceError, match="narrower parent"):
        layout.scan(LocalFileSystem(), str(tmp_path), max_entries=20)


def test_parts_are_not_spent_against_the_table_ceiling(tmp_path):
    for table in range(20):
        for part in range(15):
            write(tmp_path, f"t{table}/part-{part}.parquet")
    found = layout.scan(LocalFileSystem(), str(tmp_path),
                        max_entries=layout.MAX_SCANNED_ENTRIES)
    assert len(layout.discover_tables(found)) == 20


def test_a_mixed_directory_is_refused_when_it_is_named(tmp_path):
    write(tmp_path, "mixed/part-0.parquet")
    write(tmp_path, "mixed/part-1.csv")
    with pytest.raises(PermanentSourceError, match="mix CSV and Parquet"):
        layout.resolve(LocalFileSystem(), str(tmp_path), "mixed")


def test_nested_csv_is_refused_and_names_what_is_selectable(tmp_path):
    write(tmp_path, "raw/inner/part-0.csv")
    with pytest.raises(PermanentSourceError) as error:
        layout.resolve(LocalFileSystem(), str(tmp_path), "raw")
    assert "nested sub-directories" in str(error.value)
    assert "raw/inner" in str(error.value)


def test_nested_parquet_without_hive_segments_is_refused(tmp_path):
    write(tmp_path, "raw/inner/part-0.parquet")
    with pytest.raises(PermanentSourceError, match="key=value"):
        layout.resolve(LocalFileSystem(), str(tmp_path), "raw")


def test_a_bare_name_resolves_to_its_directory(tmp_path):
    write(tmp_path, "events/part-0.parquet")
    path, kind = layout.resolve(LocalFileSystem(), str(tmp_path), "events")
    assert path == f"{tmp_path}/events/"
    assert kind == layout.PARQUET


def test_parquet_is_probed_before_csv(tmp_path):
    write(tmp_path, "events.csv")
    write(tmp_path, "events.parquet")
    path, kind = layout.resolve(LocalFileSystem(), str(tmp_path), "events")
    assert path.endswith("events.parquet")
    assert kind == layout.PARQUET


def test_a_name_that_already_carries_its_suffix_is_not_doubled(tmp_path):
    write(tmp_path, "events.parquet")
    path, kind = layout.resolve(LocalFileSystem(), str(tmp_path),
                                "events.parquet")
    assert path == f"{tmp_path}/events.parquet"
    assert kind == layout.PARQUET


def test_a_missing_table_is_reported_as_missing(tmp_path):
    with pytest.raises(SourceNotFoundError):
        layout.resolve(LocalFileSystem(), str(tmp_path), "absent")


def test_a_missing_table_named_with_a_suffix_suggests_dropping_it(tmp_path):
    with pytest.raises(SourceNotFoundError, match="remove the suffix"):
        layout.resolve(LocalFileSystem(), str(tmp_path), "absent.parquet")


def test_the_root_keeps_the_separator_it_came_with():
    assert layout._join("/data/root", "events") == "/data/root/events"
    assert layout._join("bucket/prefix", "events") == "bucket/prefix/events"
    assert layout._join("bucket", "a/b.parquet") == "bucket/a/b.parquet"
    assert layout._join("", "bucket/key") == "bucket/key"
    assert layout._join("bucket/prefix/", "") == "bucket/prefix"


def test_hidden_markers_never_reach_the_object_set(tmp_path):
    """The hidden file carries a readable suffix, or the format filter would
    drop it first and the hidden rule would never be reached."""
    write(tmp_path, "events/part-0.parquet")
    write(tmp_path, "events/_staging.parquet")
    write(tmp_path, "events/.checkpoint.parquet")
    found = layout.objects(LocalFileSystem(), f"{tmp_path}/events/")
    assert [entry.path
            for entry in found] == [f"{tmp_path}/events/part-0.parquet"]


def test_the_verdict_is_reached_from_relative_names_alone():
    """The hosted connector lists an object store through a credential broker
    this package cannot import, so it passes the names it listed and nothing
    else; a filesystem argument here would put the rule back on both sides."""
    assert layout.classify("events", ["part-0.parquet"]) == layout.PARQUET
    assert layout.classify("events", ["part-0.csv"]) == layout.CSV
    assert layout.classify("events",
                           ["dt=2026-01-01/part-0.parquet"]) == layout.PARQUET
    assert layout.classify("events", ["notes.txt"]) is None
    assert layout.classify("events", []) is None


def test_the_verdict_refuses_what_neither_reader_supports():
    with pytest.raises(PermanentSourceError, match="mix CSV and Parquet"):
        layout.classify("events", ["a.csv", "b.parquet"])
    with pytest.raises(PermanentSourceError, match="nested sub-directories"):
        layout.classify("events", ["inner/a.csv"])
    with pytest.raises(PermanentSourceError, match="key=value"):
        layout.classify("events", ["inner/a.parquet"])


def test_the_verdict_names_what_is_selectable_below_a_refused_path():
    with pytest.raises(PermanentSourceError) as error:
        layout.classify("raw", ["inner/a.csv"], "raw")
    assert "raw/inner" in str(error.value)


def test_only_a_name_carrying_a_separator_sits_below_the_root():
    assert not layout.nested(["part-0.parquet", "part-1.parquet"])
    assert layout.nested(["dt=2026-01-01/part-0.parquet"])
    assert not layout.nested([])


# --- naming the directory a read empties -----------------------------------


def test_a_name_that_is_not_one_path_component_cannot_address_a_path():
    root = Path("/scratch/graph-1")

    assert staging_dir(root, "ORDERS") == root / "ORDERS"
    for hostile in ("/etc/kumo", "../../escape", "..", ".", "a/../../b",
                    "\x00"):
        destination = staging_dir(root, hostile)
        assert destination.parent == root, hostile
        assert destination.resolve().is_relative_to(root.resolve()), hostile


def test_two_names_that_differ_keep_two_directories():
    root = Path("/scratch/graph-1")
    names = ["a/b", "a_b", "/a/b", "../a/b", "events.v2", "v2"]

    assert len({staging_dir(root, name) for name in names}) == len(names)
    # A legitimately dotted identifier is already one component, so it stays
    # readable rather than being hashed.
    assert staging_dir(root, "events.v2") == root / "events.v2"


def test_a_destination_that_escapes_its_root_is_refused(tmp_path):
    root = tmp_path / "scratch"
    root.mkdir()
    (tmp_path / "victim").mkdir()
    (root / "ORDERS").symlink_to(tmp_path / "victim")

    with pytest.raises(ValueError, match="outside"):
        staging_dir(root, "ORDERS")
