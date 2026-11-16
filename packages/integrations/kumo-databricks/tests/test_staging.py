# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unload Databricks runs, and the objects it leaves in the volume."""

import os
import threading

import pytest
from fakes import FakeCursor, FakeFiles, FakeWorkspaceClient
from kumo_core.sources.errors import PermanentSourceError
from kumo_databricks.sql import build_projection
from kumo_databricks.staging import clear_volume, download, list_parquet, stage
from kumoapi.typing import Dtype

SOURCE = "`samples`.`tpch`.`customer`"
LOCATION = "/Volumes/example_catalog/example_schema/vol/out"
COLUMNS = {
    "c_custkey": Dtype.int64,
    "c_name": Dtype.string,
    "c_acctbal": Dtype.float,
}
PARQUET = b"PAR1" + b"body"
PATIENCE = 30
PARTS = ("part-00000-tid-1.c000.snappy.parquet",
         "part-00001-tid-1.c000.snappy.parquet")


def volume(extra=None):
    objects = {
        f"{LOCATION}/_SUCCESS": b"",
        f"{LOCATION}/_committed_1": b"{}",
        f"{LOCATION}/_started_1": b"",
        f"{LOCATION}/{PARTS[0]}": PARQUET,
        f"{LOCATION}/{PARTS[1]}": PARQUET + b"x",
    }
    objects.update(extra or {})
    return FakeFiles(objects)


@pytest.mark.parametrize("dtype,declared,expression", [
    (Dtype.float, "decimal(18,2)", "`c_acctbal`"),
    (Dtype.float, "double", "`c_acctbal`"),
    (Dtype.float, None, "`c_acctbal`"),
    (Dtype.string, "decimal(38,2)", "`c_acctbal`"),
    (Dtype.float, "decimal(38,2)",
     "CAST(`c_acctbal` AS DOUBLE) AS `c_acctbal`"),
    (Dtype.int64, "decimal(38,2)",
     "CAST(`c_acctbal` AS BIGINT) AS `c_acctbal`"),
    (Dtype.float, "DECIMAL(38,2)",
     "CAST(`c_acctbal` AS DOUBLE) AS `c_acctbal`"),
    (Dtype.float, " decimal(19, 0) ",
     "CAST(`c_acctbal` AS DOUBLE) AS `c_acctbal`"),
])
def test_only_a_decimal_too_wide_for_64_bits_is_cast_at_the_source(
        dtype, declared, expression):
    types = {"c_acctbal": declared} if declared else {}
    assert build_projection({"c_acctbal": dtype},
                            source_types=types) == (expression)


def test_a_location_containing_a_quote_is_refused():
    with pytest.raises(ValueError):
        stage(FakeCursor(), location="/Volumes/o'ut", projection="`c_name`",
              source=SOURCE)


def test_listing_returns_the_parquet_and_never_the_commit_markers():
    files = list_parquet(FakeWorkspaceClient(volume()), LOCATION)
    assert [(os.path.basename(f.path), f.size)
            for f in files] == [(PARTS[0], len(PARQUET)),
                                (PARTS[1], len(PARQUET) + 1)]


@pytest.mark.parametrize("objects", [{f"{LOCATION}/_SUCCESS": b""}, {}])
def test_a_directory_holding_no_parquet_is_refused(objects):
    with pytest.raises(PermanentSourceError):
        list_parquet(FakeWorkspaceClient(FakeFiles(objects)), LOCATION)


def test_clearing_removes_the_commit_markers_along_with_the_parquet():
    files = volume()
    clear_volume(FakeWorkspaceClient(files), LOCATION)
    assert files.objects == {}
    clear_volume(FakeWorkspaceClient(FakeFiles()), LOCATION)


def test_clear_deletes_in_parallel():
    started = threading.Barrier(2, timeout=PATIENCE)
    files = FakeFiles({f"{LOCATION}/{part}": PARQUET
                       for part in PARTS},
                      on_delete=lambda path: started.wait())
    clear_volume(FakeWorkspaceClient(files), LOCATION)
    assert files.objects == {}


def test_clear_tolerates_missing_file():
    files = volume()
    files.on_delete = files.objects.pop
    clear_volume(FakeWorkspaceClient(files), LOCATION)
    assert files.objects == {}


def test_a_download_writes_every_staged_object(tmp_path):
    client = FakeWorkspaceClient(volume())
    paths = download(client, list_parquet(client, LOCATION), str(tmp_path))
    assert [os.path.basename(path) for path in paths] == list(PARTS)
    assert [os.path.getsize(path)
            for path in paths] == [len(PARQUET), len(PARQUET) + 1]
