# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import multiprocessing as mp

import pyarrow as pa
import pyarrow.parquet as pq


def _run_all_null_timestamp_convert(parquet_dir, result_queue):
    from diskgraph import DiskgraphConfigConverter

    class ReentrantHandler(logging.Handler):
        def emit(self, record):
            # Exercise the pyo3_log/tracing_log path that re-enters Python
            # logging while the Rust converter emits all-null timestamp
            # warnings.
            record.getMessage()

    logging.getLogger().addHandler(ReentrantHandler())

    config = {
        "name": "all_null_timestamp_graph",
        "tables": [{
            "table_name": "dim_product",
            "data_source": "parquet_upload_connector",
            "source_name": "dim_product",
            "data_version": "v1",
            "pkey": {
                "name": "product_sk",
                "dtype": "int64",
                "stype": "ID",
            },
            "cols": [{
                "name": "retire_dt",
                "dtype": "date",
                "stype": "timestamp",
            }],
        }],
    }

    DiskgraphConfigConverter(
        config,
        "s3://unused/data-persist",
        {"dim_product": "v1"},
        source_paths={"dim_product": str(parquet_dir)},
    ).convert()
    result_queue.put("ok")


def test_convert_kumo_config_does_not_deadlock_on_python_logging(tmp_path):
    table = pa.table({
        "product_sk": pa.array([1, 2, 3], type=pa.int64()),
        "retire_dt": pa.array([None, None, None], type=pa.timestamp("ns")),
    })
    parquet_dir = tmp_path / "dim_product"
    parquet_dir.mkdir()
    pq.write_table(table, parquet_dir / "part-0.parquet")

    queue = mp.Queue()
    process = mp.Process(
        target=_run_all_null_timestamp_convert,
        args=(parquet_dir, queue),
    )
    process.start()
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        assert False, "convert_kumo_config deadlocked while emitting logs"

    assert process.exitcode == 0
    assert queue.get(timeout=1) == "ok"
