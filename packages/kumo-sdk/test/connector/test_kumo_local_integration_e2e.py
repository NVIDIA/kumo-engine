# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kumo SDK — Local Upload Integration E2E (real backend, local data)

Success:
  - Parquet single file
  - Parquet multi-file directory (with file_type hint)
  - CSV single file
  - CSV multi-file directory (with file_type hint)
  - CSV header auto-rename (invalid header chars → sanitized)
  - Parquet many-shards (opt-in with KUMO_HEAVY=1)

Failure (assert on message or accept backend 500 for now):
  - Parquet zero rows (single file)
  - Mixed extensions in a directory (parquet + csv) with a file_type hint
  - CSV empty file
  - CSV header-only (currently xfail until strict zero-row check lands)

Usage:
  export RUN_KUMO_LOCAL_INTEG=1
  export KUMO_API_URL="http://localhost:10002/api"
  export KUMO_API_TOKEN="kumo:YOUR_TOKEN"
  # Optional:
  export KUMO_KEEP=1
  export KUMO_HEAVY=1
  export KUMO_BASE_DIR=/path/to/tmp
  pytest -s -vv test/connector/test_kumo_local_integration_e2e.py
"""

import csv
import io
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kumoai_studio as kumo
from kumoai_studio.connector import FileUploadConnector
from kumoai_studio.connector.utils import upload_table
from kumoai_studio.graph import Table

RUN = os.getenv("RUN_KUMO_LOCAL_INTEG") == "1"
API_URL = os.getenv("KUMO_API_URL")
API_TOKEN = os.getenv("KUMO_API_TOKEN")
KEEP = os.getenv("KUMO_KEEP", "0") == "1"
HEAVY = os.getenv("KUMO_HEAVY", "0") == "1"
BASE_DIR = os.getenv("KUMO_BASE_DIR")

pytestmark = pytest.mark.skipif(
    not (RUN and API_URL and API_TOKEN),
    reason="Set RUN_KUMO_LOCAL_INTEG=1, KUMO_API_URL, "
    "and KUMO_API_TOKEN to run.")

# ---------------- helpers ----------------


def _uniq_name(prefix: str) -> str:
    # Only underscores/alphanum; no hyphens; keep reasonably short
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    name = f"{prefix}_{ts}_{suffix}"
    # Enforce a conservative max length (many backends have limits)
    return name[:60]


def _parquet_bytes(tbl: pa.Table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(tbl, buf)
    return buf.getvalue()


def _csv_bytes(rows: List[List[str]], delimiter=",", bom=False,
               newline="\n") -> bytes:
    b = io.BytesIO()
    if bom:
        b.write(b"\xef\xbb\xbf")
    t = io.TextIOWrapper(b, encoding="utf-8", newline=newline)
    w = csv.writer(t, delimiter=delimiter, lineterminator=newline,
                   quoting=csv.QUOTE_MINIMAL)
    for r in rows:
        w.writerow(r)
    t.flush()
    return b.getvalue()


def _register(name: str, file_type_hint: str):
    conn = FileUploadConnector(file_type=file_type_hint)
    src = conn[name]
    tbl = Table.from_source_table(src)
    tbl.infer_metadata()
    tbl.save(name)


# ---------------- fixtures ----------------


@pytest.fixture(scope="session", autouse=True)
def _kumo_init():
    kumo.init(API_URL, API_TOKEN)
    yield


@pytest.fixture
def local_root():
    if BASE_DIR:
        root = os.path.join(BASE_DIR, _uniq_name("kumo_local_integ"))
        os.makedirs(root, exist_ok=True)
        try:
            yield root
        finally:
            if not KEEP:
                shutil.rmtree(root, ignore_errors=True)
            else:
                print(f"[keep] Local test data kept at: {root}")
    else:
        d = tempfile.TemporaryDirectory(prefix=_uniq_name("kumo_local_integ") +
                                        "_")
        try:
            yield d.name
        finally:
            if KEEP:
                print(f"[keep] Local test data kept at: {d.name}")
                d.cleanup = lambda: None  # type: ignore
            else:
                d.cleanup()


# ---------------- SUCCESS CASES ----------------


def test_local_parquet_single_success(local_root):
    p = os.path.join(local_root, "parquet_single_ok.parquet")
    tbl = pa.table({
        "id": pa.array([1, 2, 3], type=pa.int32()),
        "name": pa.array(["a", "b", "c"], type=pa.string())
    })
    with open(p, "wb") as f:
        f.write(_parquet_bytes(tbl))

    name = _uniq_name("parquet_single_ok")
    upload_table(name=name, path=p, file_type="parquet")
    _register(name, "parquet")


@pytest.mark.xfail(
    reason="Directory upload (parquet) for local paths not implemented yet.")
def test_local_parquet_multi_success(local_root):
    d = os.path.join(local_root, "parquet_multi_ok")
    os.makedirs(d, exist_ok=True)
    schema = pa.schema([("id", pa.int32()), ("name", pa.string())])
    a = pa.table(
        {
            "id": pa.array(list(range(10)), type=pa.int32()),
            "name": pa.array([f"a{i}" for i in range(10)], type=pa.string())
        }, schema=schema)
    b = pa.table(
        {
            "id": pa.array(list(range(10, 20)), type=pa.int32()),
            "name": pa.array([f"b{i}"
                              for i in range(10, 20)], type=pa.string())
        }, schema=schema)
    with open(os.path.join(d, "a.parquet"), "wb") as f:
        f.write(_parquet_bytes(a))
    with open(os.path.join(d, "b.parquet"), "wb") as f:
        f.write(_parquet_bytes(b))

    name = _uniq_name("parquet_multi_ok")
    # IMPORTANT: pass file_type for directory uploads
    upload_table(name=name, path=d, file_type="parquet")
    _register(name, "parquet")


def test_local_csv_single_success(local_root):
    p = os.path.join(local_root, "csv_single_ok.csv")
    with open(p, "wb") as f:
        f.write(_csv_bytes([["id", "name"], ["1", "a"], ["2", "b"]]))

    name = _uniq_name("csv_single_ok")
    upload_table(name=name, path=p, file_type="csv")
    _register(name, "csv")


@pytest.mark.xfail(
    reason="Directory upload (csv) for local paths not implemented yet.")
def test_local_csv_multi_success(local_root):
    d = os.path.join(local_root, "csv_multi_ok")
    os.makedirs(d, exist_ok=True)
    rows = [["id", "name"]] + [[str(i), f"user_{i}"] for i in range(10)]
    for i in range(2):
        with open(os.path.join(d, f"part-{i:04d}.csv"), "wb") as f:
            f.write(_csv_bytes(rows))

    name = _uniq_name("csv_multi_ok")
    # IMPORTANT: pass file_type for directory uploads
    upload_table(name=name, path=d, file_type="csv")
    _register(name, "csv")


def test_local_csv_header_auto_rename_success(local_root):
    p = os.path.join(local_root, "csv_header_auto_rename.csv")
    with open(p, "wb") as f:
        f.write(
            _csv_bytes([["First Name", "Age(yrs)", "Email@Address"],
                        ["Alice", "25", "a@test.com"]]))

    name = _uniq_name("csv_header_auto_rename")
    upload_table(name=name, path=p, file_type="csv")
    _register(name, "csv")


@pytest.mark.skipif(not HEAVY, reason="Enable big test with KUMO_HEAVY=1")
def test_local_parquet_big_many_shards_success(local_root):
    d = os.path.join(local_root, "parquet_big_ok")
    os.makedirs(d, exist_ok=True)
    shard = pa.table({
        "id":
        pa.array(list(range(1000)), type=pa.int32()),
        "name":
        pa.array([f"u{i}" for i in range(1000)], type=pa.string())
    })
    data = _parquet_bytes(shard)
    for i in range(20):  # moderate by default; increase when needed
        with open(os.path.join(d, f"part-{i:05d}.parquet"), "wb") as f:
            f.write(data)

    name = _uniq_name("parquet_big_ok")
    upload_table(name=name, path=d, file_type="parquet")
    _register(name, "parquet")


# ---------------- FAILURE CASES ----------------


def test_local_parquet_zero_rows_fail(local_root):
    p = os.path.join(local_root, "parquet_zero_rows.parquet")
    zero = pa.table({"id": pa.array([], type=pa.int32())})
    with open(p, "wb") as f:
        f.write(_parquet_bytes(zero))

    # Accept client-side validation or backend 500 for now
    with pytest.raises(Exception) as ei:
        upload_table(name=_uniq_name("parquet_zero_rows"), path=p,
                     file_type="parquet")
    msg = str(ei.value)
    assert re.search(r"(zero|row|empty|internal server error)", msg,
                     flags=re.I)


@pytest.mark.xfail(reason="Mixed-type enforcement for local "
                   "directory uploads not implemented yet.")
def test_local_parquet_mixed_ext_dir_fail(local_root):
    d = os.path.join(local_root, "parquet_mixed_ext")
    os.makedirs(d, exist_ok=True)
    good = pa.table({
        "id": pa.array([1], type=pa.int32()),
        "name": pa.array(["a"], type=pa.string())
    })
    with open(os.path.join(d, "x.parquet"), "wb") as f:
        f.write(_parquet_bytes(good))
    with open(os.path.join(d, "notes.csv"), "w", encoding="utf-8") as f:
        f.write("n,txt\n1,hello\n")

    # Hint parquet to trigger mixed-type enforcement messaging
    with pytest.raises(Exception) as ei:
        upload_table(name=_uniq_name("parquet_mixed_ext"), path=d,
                     file_type="parquet")
    msg = str(ei.value)
    # Match either our explicit message or a generic mixed-type hint
    assert re.search(
        r"(Directory also contains CSV files|Mixed CSV and Parquet)", msg,
        flags=re.I)


def test_local_csv_empty_fail(local_root):
    p = os.path.join(local_root, "csv_empty.csv")
    with open(p, "wb") as f:
        f.write(b"")

    with pytest.raises(Exception) as ei:
        upload_table(name=_uniq_name("csv_empty"), path=p, file_type="csv")
    # Accept pandas' EmptyDataError or our delimiter detection errors
    msg = str(ei.value)
    assert re.search(r"(empty|delimiter|auto[- ]?detect|no columns to parse)",
                     msg, flags=re.I)
