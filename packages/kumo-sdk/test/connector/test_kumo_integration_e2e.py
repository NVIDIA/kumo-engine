# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kumo Integration E2E — per-test, self-contained datasets
(GENERATE -> UPLOAD -> REGISTER)

Summary of test cases
Success:
  - Parquet single file
  - Parquet multi-file directory
  - CSV single file
  - CSV multi-file directory
  - CSV header auto-rename (invalid header chars -> sanitized without
    duplicates)
  - Parquet many shards “big” dataset (opt-in with KUMO_HEAVY=1)

Failure (assert on error message):
  - Parquet with unsupported dtype (struct)
  - Parquet schema mismatch across shards (e.g., int vs float)
  - Parquet zero rows (single file)
  - Mixed extensions in a directory (parquet + csv)
  - CSV header only (no data rows)
  - CSV empty file
  - CSV header mismatch across shards
  - CSV .csv.gz in a directory (opt-in with KUMO_INCLUDE_GZ=1)

Usage:
  export RUN_KUMO_INTEG=1
  export KUMO_API_URL="https://kumo.example.com/api"
  export KUMO_API_TOKEN="kumo:YOUR_TOKEN"
  export KUMO_DEST="s3://your-bucket/kumo-integ"   # or gs://..., abfs://...
  # Optional:
  export KUMO_HEAVY=1        # enable big many-shard parquet success test
  export KUMO_INCLUDE_GZ=1   # enable .csv.gz failure test
  export KUMO_KEEP=1         # keep generated data (disable cleanup)
  pytest -s -vv test/connector/test_kumo_integration_e2e.py
"""

import csv
import io
import os
import re
import uuid
import warnings
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kumoai_studio as kumo
from kumoai_studio.connector import FileUploadConnector
# ---- silence noisy GCS ADC warnings during collection/run ----
from kumoai_studio.connector.utils import upload_table
from kumoai_studio.graph import Table

pytestmark = [
    pytest.mark.filterwarnings("ignore:Your application has authenticated "
                               "using end user credentials.*:UserWarning")
]
warnings.filterwarnings(
    "ignore",
    message=r"Your application has authenticated using end user credentials.*",
    category=UserWarning,
)

# ---- imports ----

# Kumo SDK

# ---- config & skip-by-default ----
RUN = os.getenv("RUN_KUMO_INTEG") == "1"
API_URL = os.getenv("KUMO_API_URL")
API_TOKEN = os.getenv("KUMO_API_TOKEN")
DEST = os.getenv("KUMO_DEST", "")
HEAVY = os.getenv("KUMO_HEAVY", "0") == "1"
INCLUDE_GZ = os.getenv("KUMO_INCLUDE_GZ", "0") == "1"
CLEANUP = os.getenv("KUMO_KEEP", "0") != "1"  # default: cleanup enabled

pytestmark.append(
    pytest.mark.skipif(
        not RUN, reason="Set RUN_KUMO_INTEG=1, KUMO_API_URL, "
        "KUMO_API_TOKEN, KUMO_DEST to run."))

# ---------------- helpers ----------------


def _uniq_root() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"kumo_integ_{ts}-{uuid.uuid4().hex[:6]}/"


def _fs_for(url: str) -> Any:
    import fsspec
    return fsspec.filesystem(url.split("://", 1)[0])


def _ensure_dir(fs, path: str) -> str:
    if not path.endswith("/"):
        path += "/"
    try:
        fs.mkdir(path)
    except Exception:
        pass
    return path


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


def _detect_file_type(fs, path: str) -> Optional[str]:
    if not path.endswith("/"):
        lo = path.lower()
        if lo.endswith(".csv"):
            return "csv"
        if lo.endswith(".parquet"):
            return "parquet"
        return None
    # directory: peek children
    try:
        items = fs.ls(path, detail=True)
    except Exception:
        return None
    csvs = [
        it["name"] for it in items
        if str(it.get("name", "")).lower().endswith(".csv")
    ]
    pars = [
        it["name"] for it in items
        if str(it.get("name", "")).lower().endswith(".parquet")
    ]
    if csvs and not pars:
        return "csv"
    if pars and not csvs:
        return "parquet"
    return None


def _register(name: str, hint: Optional[str]):
    conn = FileUploadConnector(file_type=(hint or "parquet"))
    src = conn[name]
    tbl = Table.from_source_table(src)
    tbl.infer_metadata()
    tbl.save(name)


# ---------------- fixtures ----------------


@pytest.fixture(scope="session")
def kumo_env():
    assert API_URL and API_TOKEN and DEST, ("KUMO_API_URL, KUMO_API_TOKEN, "
                                            "KUMO_DEST must be set")
    kumo.init(API_URL, API_TOKEN)
    return {"api": API_URL}


@pytest.fixture
def cloud_scratch():
    """Yields (fs, root_prefix) for the test to write under.
    Cleans up the entire prefix on teardown unless KUMO_KEEP=1.
    """
    root = DEST.rstrip("/") + "/" + _uniq_root()
    fs = _fs_for(root)
    try:
        yield fs, root
    finally:
        if CLEANUP:
            try:
                fs.rm(root, recursive=True)
            except Exception:
                pass
        else:
            print(f"[keep] Leaving data at: {root}")


# --------------- SUCCESS CASES ---------------


def test_parquet_single_success(kumo_env, cloud_scratch):
    fs, base = cloud_scratch
    tbl = pa.table({
        "id": pa.array([1, 2, 3], type=pa.int32()),
        "name": pa.array(["a", "b", "c"], type=pa.string())
    })
    p = base + "parquet_single_ok.parquet"
    with fs.open(p, "wb") as f:
        f.write(_parquet_bytes(tbl))
    name = "parquet_single_ok"
    upload_table(name=name, path=p, file_type=_detect_file_type(fs, p))
    _register(name, "parquet")


def test_parquet_multi_success(kumo_env, cloud_scratch):
    fs, root = cloud_scratch
    base = _ensure_dir(fs, root + "parquet_multi_ok/")
    schema = pa.schema([("id", pa.int32()), ("name", pa.string())])
    a = pa.table(
        {
            "id": pa.array(list(range(100)), type=pa.int32()),
            "name": pa.array([f"a{i}" for i in range(100)], type=pa.string())
        }, schema=schema)
    b = pa.table(
        {
            "id": pa.array(list(range(100, 200)), type=pa.int32()),
            "name": pa.array([f"b{i}"
                              for i in range(100, 200)], type=pa.string())
        }, schema=schema)
    for fn, t in [("a.parquet", a), ("b.parquet", b)]:
        with fs.open(base + fn, "wb") as f:
            f.write(_parquet_bytes(t))
    name = "parquet_multi_ok"
    upload_table(name=name, path=base, file_type="parquet")
    _register(name, "parquet")


def test_csv_single_success(kumo_env, cloud_scratch):
    fs, base = cloud_scratch
    p = base + "csv_single_ok.csv"
    with fs.open(p, "wb") as f:
        f.write(_csv_bytes([["id", "name"], ["1", "a"], ["2", "b"]]))
    name = "csv_single_ok"
    upload_table(name=name, path=p, file_type="csv")
    _register(name, "csv")


def test_csv_multi_success(kumo_env, cloud_scratch):
    fs, root = cloud_scratch
    base = _ensure_dir(fs, root + "csv_multi_ok/")
    rows = [["id", "name"]] + [[str(i), f"user_{i}"] for i in range(50)]
    for i in range(2):
        with fs.open(f"{base}part-{i:04d}.csv", "wb") as f:
            f.write(_csv_bytes(rows))
    name = "csv_multi_ok"
    upload_table(name=name, path=base, file_type="csv")
    _register(name, "csv")


def test_csv_header_auto_rename_success(kumo_env, cloud_scratch):
    fs, base = cloud_scratch
    p = base + "csv_header_auto_rename.csv"
    with fs.open(p, "wb") as f:
        f.write(
            _csv_bytes([["First Name", "Age(yrs)", "Email@Address"],
                        ["Alice", "25", "a@test.com"]]))
    name = "csv_header_auto_rename"
    upload_table(name=name, path=p, file_type="csv")
    _register(name, "csv")


@pytest.mark.skipif(not HEAVY, reason="Enable big test with KUMO_HEAVY=1")
def test_parquet_big_many_shards_success(kumo_env, cloud_scratch):
    fs, root = cloud_scratch
    base = _ensure_dir(fs, root + "parquet_big_ok/")
    # Many small shards to simulate a large dataset without huge cost.
    shard = pa.table({
        "id":
        pa.array(list(range(100)), type=pa.int32()),
        "name":
        pa.array([f"u{i}" for i in range(1000)], type=pa.string())
    })
    data = _parquet_bytes(shard)
    for i in range(100 if HEAVY else 10):
        with fs.open(f"{base}part-{i:05d}.parquet", "wb") as f:
            f.write(data)
    name = "parquet_big_ok"
    upload_table(name=name, path=base, file_type="parquet")
    _register(name, "parquet")


def test_parquet_all_supported_types_success(kumo_env, cloud_scratch):
    """Happy-path: single Parquet file with all supported dtypes
                (minus dictionary types).
    Allowed: bool, signed ints, floats, (regular) string, date32/date64,
             epoch-based timestamps, (regular) binary, decimal (<= precision),
             list of {signed int, float}.
    Disallowed (intentionally omitted here): unsigned ints, null-typed columns,
             time32/64, duration, list<string>, list<bool>, large_* types.
    """
    fs, base = cloud_scratch
    p = base + "parquet_all_supported_types.parquet"

    tbl = pa.table({
        # primitives (signed only)
        "b":
        pa.array([True, False, True], type=pa.bool_()),
        "i8":
        pa.array([1, -2, 3], type=pa.int8()),
        "i16":
        pa.array([100, -200, 300], type=pa.int16()),
        "i32":
        pa.array([1_000, -2_000, 3_000], type=pa.int32()),
        "i64":
        pa.array([10_000, -20_000, 30_000], type=pa.int64()),
        "f32":
        pa.array([1.5, 2.5, 3.5], type=pa.float32()),
        "f64":
        pa.array([100.0, 200.25, 300.5], type=pa.float64()),
        "s":
        pa.array(["a", "b", "c"], type=pa.string()),
        # dates
        "d32":
        pa.array([1, 2, 3], type=pa.date32()),
        "d64":
        pa.array([86_400_000, 172_800_000, 259_200_000], type=pa.date64()),
        # epoch-based timestamps (all units)
        "ts_s":
        pa.array([1, 2, 3], type=pa.timestamp("s")),
        "ts_ms":
        pa.array([1_000, 2_000, 3_000], type=pa.timestamp("ms")),
        "ts_us":
        pa.array([1_000_000, 2_000_000, 3_000_000], type=pa.timestamp("us")),
        "ts_ns":
        pa.array([1_000_000_000, 2_000_000_000, 3_000_000_000],
                 type=pa.timestamp("ns")),
        # regular binary
        "bin":
        pa.array([b"a", b"\x00b", b"ccc"], type=pa.binary()),
        # decimal within precision/scale limits
        "dec_18_4":
        pa.array(
            [Decimal("1234.5678"),
             Decimal("0.0001"),
             Decimal("-7.1250")],
            type=pa.decimal128(18, 4),
        ),
        # lists of allowed primitives (NOT string/bool; NOT large_list)
        "list_i32":
        pa.array([[1, 2], [], [3]], type=pa.list_(pa.int32())),
        "list_f64":
        pa.array([[1.0, 2.5], [3.14], []], type=pa.list_(pa.float64())),
    })

    with fs.open(p, "wb") as f:
        buf = io.BytesIO()
        pq.write_table(tbl, buf)
        f.write(buf.getbuffer())

    name = "parquet_all_supported_types"
    upload_table(name=name, path=p, file_type="parquet")
    _register(name, "parquet")


# --------------- FAILURE CASES (assert message) ---------------


def test_parquet_bad_type_struct_fail(kumo_env, cloud_scratch):
    fs, root = cloud_scratch
    base = _ensure_dir(fs, root + "parquet_bad_type_struct/")
    good = pa.table({
        "id": pa.array([1], type=pa.int32()),
        "name": pa.array(["a"], type=pa.string())
    })
    bad = pa.table({
        "id":
        pa.array([1], type=pa.int32()),
        "val":
        pa.array([{
            "x": 1
        }], type=pa.struct([("x", pa.int32())]))
    })
    with fs.open(base + "good.parquet", "wb") as f:
        f.write(_parquet_bytes(good))
    with fs.open(base + "bad_struct.parquet", "wb") as f:
        f.write(_parquet_bytes(bad))
    with pytest.raises(Exception) as ei:
        upload_table(name="parquet_bad_type_struct", path=base,
                     file_type="parquet")
    assert re.search(r"(Unsupported Parquet|not supported|struct)",
                     str(ei.value), flags=re.I)


def test_parquet_schema_mismatch_fail(kumo_env, cloud_scratch):
    fs, root = cloud_scratch
    base = _ensure_dir(fs, root + "parquet_schema_mismatch/")
    t1 = pa.table({
        "id": pa.array([1, 2], type=pa.int32()),
        "val": pa.array([10, 11], type=pa.int32())
    })
    t2 = pa.table({
        "id": pa.array([3, 4], type=pa.int32()),
        "val": pa.array([10.5, 11.5], type=pa.float64())
    })
    for fn, t in [("part-0000.parquet", t1), ("part-0001.parquet", t2)]:
        with fs.open(base + fn, "wb") as f:
            f.write(_parquet_bytes(t))
    with pytest.raises(Exception) as ei:
        upload_table(name="parquet_schema_mismatch", path=base,
                     file_type="parquet")
    assert re.search(r"(schema|mismatch|float)", str(ei.value), flags=re.I)


def test_parquet_zero_rows_fail(kumo_env, cloud_scratch):
    fs, base = cloud_scratch
    p = base + "parquet_zero_rows.parquet"
    zero = pa.table({"id": pa.array([], type=pa.int32())})
    with fs.open(p, "wb") as f:
        f.write(_parquet_bytes(zero))
    with pytest.raises(Exception) as ei:
        upload_table(name="parquet_zero_rows", path=p, file_type="parquet")
    assert re.search(r"(zero|row|empty)", str(ei.value), flags=re.I)


def test_parquet_mixed_ext_dir_fail(kumo_env, cloud_scratch):
    fs, root = cloud_scratch
    base = _ensure_dir(fs, root + "parquet_mixed_ext/")
    good = pa.table({
        "id": pa.array([1], type=pa.int32()),
        "name": pa.array(["a"], type=pa.string())
    })
    with fs.open(base + "x.parquet", "wb") as f:
        f.write(_parquet_bytes(good))
    with fs.open(base + "notes.csv", "w", encoding="utf-8") as f:
        f.write("n,txt\n1,hello\n")
    with pytest.raises(Exception) as ei:
        upload_table(name="parquet_mixed_ext", path=base, file_type=None)
    assert re.search(r"(Mixed CSV and Parquet|remove|file_type)",
                     str(ei.value), flags=re.I)


def test_csv_header_only_fail(kumo_env, cloud_scratch):
    fs, base = cloud_scratch
    p = base + "csv_header_only.csv"
    with fs.open(p, "wb") as f:
        f.write(_csv_bytes([["id", "name"]]))
    with pytest.raises(Exception) as ei:
        upload_table(name="csv_header_only", path=p, file_type="csv")
    assert re.search(r"(zero|row|no data)", str(ei.value), flags=re.I)


def test_csv_empty_fail(kumo_env, cloud_scratch):
    fs, base = cloud_scratch
    p = base + "csv_empty.csv"
    with fs.open(p, "wb") as f:
        f.write(b"")
    with pytest.raises(Exception) as ei:
        upload_table(name="csv_empty", path=p, file_type="csv")
    assert re.search(r"(empty|delimiter|auto[- ]?detect)", str(ei.value),
                     flags=re.I)


def test_csv_header_mismatch_dir_fail(kumo_env, cloud_scratch):
    fs, root = cloud_scratch
    base = _ensure_dir(fs, root + "csv_header_mismatch/")
    with fs.open(base + "a.csv", "wb") as f:
        f.write(_csv_bytes([["id", "name"], ["1", "a"]]))
    with fs.open(base + "b.csv", "wb") as f:
        f.write(_csv_bytes([["name", "id"], ["b", "2"]]))
    with pytest.raises(Exception) as ei:
        upload_table(name="csv_header_mismatch", path=base, file_type="csv")
    assert re.search(r"(header|mismatch|column)", str(ei.value), flags=re.I)


@pytest.mark.skipif(not INCLUDE_GZ, reason="Enable with KUMO_INCLUDE_GZ=1")
def test_csv_gz_fail(kumo_env, cloud_scratch):
    import gzip
    fs, root = cloud_scratch
    base = _ensure_dir(fs, root + "csv_gz/")
    with fs.open(base + "part-0000.csv.gz", "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb") as z:
            z.write(_csv_bytes([["id", "name"], ["1", "a"]]))
    with pytest.raises(Exception) as ei:
        upload_table(name="csv_gz", path=base, file_type="csv")
    assert re.search(r"(gz|gzip|not supported)", str(ei.value), flags=re.I)
