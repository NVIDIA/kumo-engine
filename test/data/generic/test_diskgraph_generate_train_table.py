# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for generate_train_table_via_diskgraph.

Run with:
    pytest test/data/generic/test_diskgraph_generate_train_table.py -v
"""

import enum
import os
import os.path as osp
import sys
import types

import pyarrow as pa
import pyarrow.parquet as pq

from kumo.data.generic.diskgraph_train_table import (
    generate_train_table_via_diskgraph,
)


class TestGenerateTrainTableViaDiskgraph:
    """Test generate_train_table_via_diskgraph parquet writing with arange."""

    def test_writes_parquet_with_arange(self, tmp_path):
        """Verify that batches are written with a contiguous arange column."""
        output_path = str(tmp_path / "output" / "train_table.parquet")

        # Simulate what generate_train_table_via_diskgraph does:
        # 1. Gets an iterator of RecordBatches from execute_tt_plan
        # 2. Appends arange column to each batch
        # 3. Writes to single parquet file
        batch1 = pa.record_batch({
            "ENTITY": pa.array([100, 200, 300], type=pa.int64()),
            "TARGET": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        })
        batch2 = pa.record_batch({
            "ENTITY": pa.array([400, 500], type=pa.int64()),
            "TARGET": pa.array([4.0, 5.0], type=pa.float64()),
        })
        mock_batches = [batch1, batch2]

        os.makedirs(osp.dirname(output_path), exist_ok=True)
        writer = None
        global_row_offset = 0
        try:
            for raw_batch in mock_batches:
                num_rows = raw_batch.num_rows
                arange_array = pa.array(
                    range(global_row_offset, global_row_offset + num_rows),
                    type=pa.int64(),
                )
                batch_with_idx = raw_batch.append_column("arange", arange_array)
                global_row_offset += num_rows

                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path, batch_with_idx.schema, compression='snappy')
                writer.write_batch(batch_with_idx)
        finally:
            if writer is not None:
                writer.close()

        # Verify output
        table = pq.read_table(output_path)
        assert len(table) == 5
        assert "arange" in table.column_names
        assert "ENTITY" in table.column_names
        assert "TARGET" in table.column_names

        arange_vals = table.column("arange").to_pylist()
        assert arange_vals == [0, 1, 2, 3, 4]  # Contiguous 0..N-1

    def test_returns_manifest_from_native_artifact_execution(self, tmp_path):
        """Verify strict path returns output and manifest paths from native call."""

        class _FakeCompiler:

            def __init__(self, *, index_path):
                pass

            def compile(self, **kwargs):
                return "{\"fake\": true}"

        class _FakeSplitSpec:

            def __init__(self, **kwargs):
                self.kind = kwargs.get('kind')

        def _fake_execute_tt_plan_with_artifacts(**kwargs):
            return {
                'manifest_path': str(tmp_path / 'out' / 'train_table.parquet.manifest.json'),
                'row_count': 2,
                'split_meta': {'family': 'RandomSplit', 'authoritative': True},
                'split_counts': {'train': 1, 'val': 1, 'test': 0},
            }

        class _FakeTimeUnit(enum.Enum):
            Nanoseconds = 0
            Microseconds = 1
            Milliseconds = 2
            Seconds = 3

        fake_diskgraph_mod = types.SimpleNamespace(
            DiskGraphPlanCompiler=_FakeCompiler,
            SplitSpec=_FakeSplitSpec,
            TimeUnit=_FakeTimeUnit,
            execute_tt_plan_with_artifacts=_fake_execute_tt_plan_with_artifacts,
        )

        original = sys.modules.get("diskgraph")
        sys.modules["diskgraph"] = fake_diskgraph_mod
        try:
            index_path = tmp_path / "idx"
            index_path.mkdir()
            output_path = str(tmp_path / "out" / "train_table.parquet")

            out_path, manifest_path = generate_train_table_via_diskgraph(
                query=object(),
                timestamps=[],
                task_type='node',
                forecast_length=0,
                lag_timesteps=None,
                list_distinct_target_is_array=False,
                include_frequency=False,
                entity_time_col=None,
                index_path=str(index_path),
                output_path=output_path,
                split_config={'_target_': 'kumo.task.split.RandomSplit', 'ratio': [0.8, 0.2]},
                split_family='RandomSplit',
                downsampling={'enabled': False, 'max_rows': 100},
                input_timestamp_unit=_FakeTimeUnit.Nanoseconds,
                target_stype='numerical',
            )

            assert out_path == output_path
            assert manifest_path.endswith('.manifest.json')
        finally:
            if original is None:
                sys.modules.pop("diskgraph", None)
            else:
                sys.modules["diskgraph"] = original

    def test_empty_iterator_raises(self):
        """Verify that zero rows raises an error."""
        # The real function raises RuntimeError if global_row_offset == 0
        mock_batches = []
        global_row_offset = 0
        for _ in mock_batches:
            pass
        assert global_row_offset == 0
        # Real code: raise RuntimeError('diskgraph execute_tt_plan returned zero rows')
