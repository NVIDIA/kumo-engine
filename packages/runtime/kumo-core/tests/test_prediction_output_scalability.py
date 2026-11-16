# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No-inference regression for output completion beyond available row memory."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _memory_probe(root, repetitions):
    import resource

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumo_core.prediction import \
        publish_scored_output as _publish_scored_output
    from kumo_core.train.artifact_manifest import (
        load_prediction_selection,
        resolve_prediction_selection_artifact,
    )
    from kumo_core.train.artifact_stats import PredictionArtifactStatsCollector
    from kumo_core.train.io import StreamingFrameWriter
    from kumo_core.types import PredictionArtifactType

    root = Path(root)
    generation = root / '_attempts' / 'runtime-00000' / 'attempt-1'
    kind = PredictionArtifactType.PREDICTIONS
    batch = pa.table(
        {f'value_{i}': np.arange(4096, dtype=np.float64)
         for i in range(64)})
    # Bound extra virtual memory after imports/Arrow's runtime initialization.
    # The large case has 1 GiB of decoded numeric rows against 512 MiB headroom.
    pa.total_allocated_bytes()
    current_vm = int(
        next(line.split()[1]
             for line in Path('/proc/self/status').read_text().splitlines()
             if line.startswith('VmSize:'))) * 1024
    headroom = 512 << 20
    resource.setrlimit(resource.RLIMIT_AS,
                       (current_vm + headroom, current_vm + headroom))
    collector = PredictionArtifactStatsCollector(str(generation), (kind, ))
    writer = StreamingFrameWriter(
        str(generation / kind.output_file_name / 'part-00000'),
        max_file_size=8 << 20,
        max_buffer_size=4 << 20,
        index=False,
        max_row_group_rows=4096,
        stats_sample_rows=1000,
        spool_root=root / 'spool',
    )
    with writer:
        for _ in range(repetitions):
            writer.write(batch)
    collector.add_partition(0, {kind: writer.stats})

    def forbidden(*args, **kwargs):
        raise AssertionError(
            'output completion attempted whole-result collection')

    patch = pytest.MonkeyPatch()
    patch.setattr(pd, 'read_parquet', forbidden)
    patch.setattr(pd, 'concat', forbidden)
    patch.setattr(pq, 'read_table', forbidden)
    reference = _publish_scored_output(
        collector,
        output_dir=str(root),
        run_id='bp-memory',
        attempt_id='attempt-1',
        graph_snapshot_id='snapshot',
        source_config_sha256='1' * 64,
        runtime_config_sha256='2' * 64,
    )
    selection = load_prediction_selection(reference)
    artifact = resolve_prediction_selection_artifact(reference, kind)
    assert selection.artifacts[0].num_rows == repetitions * batch.num_rows
    assert sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in artifact.files) == repetitions * batch.num_rows
    from kumo_core.object_store_sink import ObjectStoreSinkBackend
    from pyarrow.fs import LocalFileSystem

    class LocalSink(ObjectStoreSinkBackend):
        scheme = 'file'
        provider = 'test'

        def _destination(self, connector, coordinate):
            return LocalFileSystem(), coordinate

    target = root / 'exported'
    LocalSink().write_files(None,
                            str(target) + '/', artifact.files,
                            export_id='memory-export')
    assert (target / '_SUCCESS').exists()
    assert sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in target.glob('*.parquet')) == repetitions * batch.num_rows
    print(
        json.dumps(
            dict(
                rows=repetitions * batch.num_rows,
                logical_bytes=repetitions * batch.nbytes,
                extra_memory_limit=headroom, peak_rss_bytes=int(
                    next(line.split()[1]
                         for line in Path('/proc/self/status').read_text().
                         splitlines() if line.startswith('VmHWM:'))) * 1024,
                files=len(artifact.files))))


@pytest.mark.skipif(
    importlib.util.find_spec('kumo_core') is None,
    reason='kumo_core is not installed')
@pytest.mark.skipif(sys.platform != 'linux',
                    reason='uses Linux RLIMIT_AS and /proc')
def test_completion_is_bounded_by_batches_not_total_output(tmp_path):
    reports = []
    for repetitions in (32, 512):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(tmp_path / str(repetitions)),
                str(repetitions)
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
            env={
                **os.environ, 'OPENBLAS_NUM_THREADS': '1',
                'OMP_NUM_THREADS': '1',
                'MALLOC_ARENA_MAX': '2'
            },
        )
        reports.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert reports[1]['logical_bytes'] > reports[1]['extra_memory_limit']
    assert reports[1]['peak_rss_bytes'] - reports[0][
        'peak_rss_bytes'] < 128 << 20
    print(json.dumps(reports))


if __name__ == '__main__':
    _memory_probe(sys.argv[1], int(sys.argv[2]))
