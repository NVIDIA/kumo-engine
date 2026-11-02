# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the nightly reader with generated Parquet outside routine CPU CI."""

import argparse
import importlib.util
import json
import math
import resource
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def run(output, rows):
    source = Path(__file__).parents[1] / 'scripts/nightly_check.py'
    spec = importlib.util.spec_from_file_location('nightly_reader', source)
    nightly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nightly)
    source_sha256 = nightly.digest(source)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'predictions.parquet'
    schema = pa.schema([('ROW_ID', pa.int64()), ('ENTITY', pa.int64()),
                        ('CLASS', pa.int64()), ('SCORE', pa.float64())])
    entities = math.ceil(rows / 500)
    factor = 48271
    while math.gcd(factor, entities) != 1:
        factor += 2
    started = time.monotonic()
    with pq.ParquetWriter(inputs, schema) as writer:
        for start in range(0, rows, 65536):
            ids = np.arange(start, min(start + 65536, rows), dtype=np.int64)
            writer.write_table(
                pa.Table.from_arrays([
                    ids, ((ids // 500) * factor) % entities, ids % 500,
                    ((ids * 48271) % 1000000) / 1000000
                ], schema=schema))
    generation_seconds = time.monotonic() - started
    peak_disk_bytes = 0
    done = threading.Event()

    def sample():
        nonlocal peak_disk_bytes
        while not done.is_set():
            size = 0
            for path in output.rglob('*'):
                try:
                    if path.is_file():
                        size += path.stat().st_blocks * 512
                except FileNotFoundError:
                    pass
            peak_disk_bytes = max(peak_disk_bytes, size)
            done.wait(1)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    started = time.monotonic()
    count = checksum = 0
    previous = None
    try:
        for frame in nightly.ordered_batches([str(inputs)], output / 'sort'):
            entity, score, item = (frame[column].to_numpy()
                                   for column in ('ENTITY', 'SCORE', 'CLASS'))
            assert np.all((entity[1:] > entity[:-1]) | (
                (entity[1:] == entity[:-1]) & ((score[1:] < score[:-1]) | (
                    (score[1:] == score[:-1]) & (item[1:] >= item[:-1])))))
            first = (int(entity[0]), -float(score[0]), int(item[0]))
            assert previous is None or previous <= first
            previous = (int(entity[-1]), -float(score[-1]), int(item[-1]))
            count += len(frame)
            checksum += int(frame.ROW_ID.sum())
    finally:
        seconds = time.monotonic() - started
        done.set()
        worker.join()
    assert count == rows and checksum == rows * (rows - 1) // 2
    assert not list((output / 'sort').iterdir())
    result = dict(
        rows=count, reader_seconds=seconds,
        microseconds_per_row=seconds * 1000000 / rows,
        generation_seconds=generation_seconds,
        peak_process_rss_bytes=resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss * 1024,
        peak_owned_disk_bytes=peak_disk_bytes,
        input_bytes=inputs.stat().st_size, source_sha256=source_sha256,
        pyarrow=pa.__version__,
        evidence='Synthetic Parquet; CPU only; no model qualification',
        order_count_checksum_and_cleanup='PASS')
    nightly.save(output / 'measurement.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=8000000)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error('--rows must be positive')
    print(json.dumps(run(args.output, args.rows)))
