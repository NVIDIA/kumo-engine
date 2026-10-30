# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read retained predictions with an explicit CPU sort budget."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def ordered_batches(files, scratch, *, batch_rows=65536):
    """Keep the frozen Parquet ordering and Arrow-to-pandas stream unchanged."""
    import duckdb

    files = list(map(str, files))
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(
        status='STARTED', files=files, rows=0, batches=0, batch_rows=batch_rows,
        started_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        duckdb_version=duckdb.__version__,
        requested_config=dict(memory_limit='2GB', threads=1,
                              temp_directory=str(scratch)))
    try:
        with duckdb.connect(config=receipt['requested_config']) as connection:
            actual = connection.execute(
                "SELECT current_setting('memory_limit'), current_setting('threads')"
            ).fetchone()
            receipt.update(actual_memory_limit=actual[0],
                           actual_threads=actual[1])
            reader = connection.read_parquet(list(files)).order(
                'ENTITY, SCORE DESC, CLASS ASC').fetch_record_batch(batch_rows)
            for batch in reader:
                frame = batch.to_pandas()
                receipt['rows'] += len(frame)
                receipt['batches'] += 1
                yield frame
        receipt['status'] = 'COMPLETE'
    except GeneratorExit:
        receipt['status'] = 'STOPPED_BEFORE_COMPLETION'
        raise
    except BaseException as error:
        receipt.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic() - started,
                       finished_at=datetime.now(timezone.utc).isoformat())
        (scratch / 'reader-receipt.json'
         ).write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
