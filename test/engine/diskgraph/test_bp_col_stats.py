# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Kumo's DiskGraph batch-prediction statistics contract."""
import json
import os
import os.path as osp
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from io_utils import fsutil


def _stub_col(name, stype, dtype):
    class _S:
        def __init__(self, value):
            self.value = value

    class _Col:
        def __init__(self, name, stype, dtype):
            self.name = name
            self.stype = _S(stype)
            self.dtype = _S(dtype)

    return _Col(name, stype, dtype)


def _stub_table_def(cols):
    class _TD:
        def __init__(self):
            self.cols = cols

    return _TD()


class TestToKumoColSpecs:
    """``_to_kumo_col_specs`` is now a thin projection — the real Kumo
    ``Stype``/``Dtype`` → diskgraph type translation happens on the
    diskgraph side via ``map_kumo_stype``/``map_kumo_dtype``. This
    module confirms the projection passes through raw value strings and skips
    unresolved columns consistently with the normal stats pipeline.
    """
    def test_projects_raw_value_strings(self):
        from kumo.diskgraph.service import _to_kumo_col_specs
        td = _stub_table_def([
            _stub_col('entity', 'ID', 'string'),
            _stub_col('PREDICTED', 'categorical', 'int'),
            _stub_col('CLASS', 'categorical', 'string'),
            _stub_col('SCORE', 'numerical', 'float'),
        ])
        assert _to_kumo_col_specs(td) == [
            {
                'name': 'entity',
                'stype': 'ID',
                'dtype': 'string'
            },
            {
                'name': 'PREDICTED',
                'stype': 'categorical',
                'dtype': 'int'
            },
            {
                'name': 'CLASS',
                'stype': 'categorical',
                'dtype': 'string'
            },
            {
                'name': 'SCORE',
                'stype': 'numerical',
                'dtype': 'float'
            },
        ]

    def test_unsupported_columns_are_skipped(self):
        """``Optional[Stype]`` is None on synthesized columns sometimes;
        normal stats skip those columns rather than failing the table."""
        from kumo.diskgraph.service import _to_kumo_col_specs

        class _NoneCol:
            name = 'x'
            stype = None
            dtype = None

        td = _stub_table_def([_NoneCol()])
        assert _to_kumo_col_specs(td) == []

    def test_projects_authoritative_integer_timestamp_unit(self):
        from kumo.diskgraph.service import _to_kumo_col_specs

        col = _stub_col('event_time', 'timestamp', 'int')
        col.timestamp_format = type('_Unit', (), {'value': 'ms'})()

        assert _to_kumo_col_specs(_stub_table_def([col])) == [{
            'name':
            'event_time',
            'stype':
            'timestamp',
            'dtype':
            'int',
            'timestamp_unit':
            'ms',
        }]

    def test_projects_authoritative_integer_timestamp_format(self):
        from kumo.diskgraph.service import _to_kumo_col_specs

        col = _stub_col('event_date', 'timestamp', 'int')
        col.timestamp_format = '%Y%m%d'

        assert _to_kumo_col_specs(_stub_table_def([col])) == [{
            'name':
            'event_date',
            'stype':
            'timestamp',
            'dtype':
            'int',
            'timestamp_format':
            '%Y%m%d',
        }]


class TestListParquetFiles:
    """``_list_parquet_files`` handles the three BP-output writer shapes:
    single file, directory of part-*.parquet, and missing.
    """
    def test_single_file(self):
        from rest.backends.data_backend.data_backend import _list_parquet_files
        with tempfile.TemporaryDirectory() as d:
            p = osp.join(d, 'predictions.parquet')
            fsutil.write_text(p, '')  # empty file is fine for path resolution
            assert _list_parquet_files(p) == [p]

    def test_directory_of_parts(self):
        from rest.backends.data_backend.data_backend import _list_parquet_files
        with tempfile.TemporaryDirectory() as d:
            outdir = osp.join(d, 'predictions.parquet')
            os.makedirs(outdir)
            files = [
                'part-00000.parquet', 'part-00001.parquet', 'part-00002.parquet'
            ]
            for fn in files:
                fsutil.write_text(osp.join(outdir, fn), '')
            # Also a non-parquet file that must be skipped.
            fsutil.write_text(osp.join(outdir, '_SUCCESS'), '')
            got = _list_parquet_files(outdir)
            assert got == [osp.join(outdir, fn) for fn in sorted(files)]

    def test_missing_returns_empty(self):
        from rest.backends.data_backend.data_backend import _list_parquet_files
        assert _list_parquet_files(
            '/nonexistent/path/predictions.parquet') == []


def test_physical_parquet_types_normalize_to_logical_bp_specs(tmp_path):
    from diskgraph import compute_col_stats_for_parquet

    first_path = tmp_path / 'part-00000.parquet'
    second_path = tmp_path / 'part-00001.parquet'
    pq.write_table(
        pa.table({
            'PREDICTED': pa.array([True, False]),
            'CLASS': pa.array([1, 2], type=pa.int64()),
            'ENTITY': pa.array([10, 20], type=pa.int32()),
            'SCORE': pa.array([0.2, 0.8], type=pa.float32()),
        }),
        first_path,
    )
    pq.write_table(
        pa.table({
            'SCORE': pa.array([0.4, 0.6], type=pa.float64()),
            'ENTITY': pa.array([30, 40], type=pa.uint64()),
            'CLASS': pa.array([1, 2], type=pa.uint16()),
            'PREDICTED': pa.array([False, True]),
        }),
        second_path,
    )
    specs = [
        {
            'name': 'ENTITY',
            'stype': 'ID',
            'dtype': 'string'
        },
        {
            'name': 'PREDICTED',
            'stype': 'categorical',
            'dtype': 'int'
        },
        {
            'name': 'CLASS',
            'stype': 'categorical',
            'dtype': 'string'
        },
        {
            'name': 'SCORE',
            'stype': 'numerical',
            'dtype': 'float'
        },
    ]

    full_json, _ = compute_col_stats_for_parquet(
        parquet_paths=[str(first_path), str(second_path)],
        kumo_column_specs_json=json.dumps(specs),
        data_version='v1',
    )
    stats = json.loads(full_json)

    assert stats['ENTITY__kumo.ID__COUNT__False']['value'] == 4
    predicted_counts = stats[
        'PREDICTED__kumo.categorical__CATEGORY_COUNTS__False']['value']
    assert dict(predicted_counts) == {0: 2, 1: 2}
    class_counts = stats['CLASS__kumo.categorical__CATEGORY_COUNTS__False'][
        'value']
    assert dict(class_counts) == {'1': 2, '2': 2}
    by_class = stats['SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False'][
        'value']
    assert set(by_class) == {'1', '2'}
    assert by_class['1']['mean'] == pytest.approx(0.3)
    assert by_class['2']['mean'] == pytest.approx(0.7)

    # The reader uses this same request to decide whether BP statistics are
    # available. Assert the current DiskGraph producer satisfies it exactly;
    # a requested-but-never-emitted stat otherwise leaves the API polling
    # forever even though the activity completed successfully.
    from collections import defaultdict

    from kumoapi.source_table import S3RelativeSource
    from kumoapi.typing import Dtype, Stype

    from kumo.data.cache.exact_stats import parse_col_stat_from_key
    from rest.config.file_type import FileType
    from rest.config.table_v2 import Column, S3SourceTable, TableDefinition
    from rest.utils.table import ColStatsSet
    from statedb.config.common import ConnectorID

    connector_id = ConnectorID('connector_' + '1' * 32)
    table_def = TableDefinition(
        cols=tuple(
            Column(
                name=spec['name'],
                stype=Stype(spec['stype']),
                dtype=Dtype(spec['dtype']),
            ) for spec in specs),
        source_table=S3SourceTable(
            file_type=FileType.PARQUET,
            connector_id=connector_id,
            source=S3RelativeSource(path='predictions'),
        ),
        pkey='ENTITY',
    )
    present = defaultdict(set)
    for key in stats:
        name, stat_type, approx = parse_col_stat_from_key(key)
        if not approx:
            present[name].add(stat_type)

    request = ColStatsSet.BP.get_stats_types(table_def)
    for column, required in request.items():
        assert set(required) <= present[column.name], (
            column.name,
            set(required) - present[column.name],
        )


class TestSummaryStatsByClass:
    """The diskgraph BP path emits ``SUMMARY_STATS_BY_CLASS`` for the SCORE
    column of multiclass/ranking jobs: SCORE grouped by CLASS into
    ``{class: {min, max, mean}}`` (the post-cutover replacement for Spark's
    group-by). These tests pin the two kumo-side contracts that make the
    diskgraph-produced value light up the ``/prediction_stats`` endpoint:
    the ``__``-key parses back to the right ``ColStatType``, and a dict-valued
    stat surfaces as ``ColumnStats.bp_stats``.
    """
    def test_by_class_key_parses_to_colstattype(self):
        """The Rust side emits ``SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS
        __False``; the cache reader must map it back to the enum, with a
        dict value kept intact (``Stats.value`` is ``Any``)."""
        from kumo_core.types import ColStatType

        from kumo.data.cache.exact_stats import parse_col_stat_from_key

        col, stat, approx = parse_col_stat_from_key(
            'SCORE__kumo.numerical__SUMMARY_STATS_BY_CLASS__False')
        assert col == 'SCORE'
        assert stat == ColStatType.SUMMARY_STATS_BY_CLASS
        assert approx is False

    def test_dict_value_surfaces_as_bp_stats(self):
        """A numerical column whose stats dict carries
        ``SUMMARY_STATS_BY_CLASS`` (a ``{class: {min,max,mean}}`` dict) must
        surface that dict verbatim as ``ColumnStats.bp_stats``."""
        from kumo_core.types import ColStatType
        from kumoapi.typing import Stype

        from kumo.stats.stats import Stats
        from rest.config.table import ColumnStats

        by_class = {
            'fraud': {
                'min': 0.2,
                'max': 0.9,
                'mean': 0.5333333333333333
            },
            'legit': {
                'min': 0.1,
                'max': 0.8,
                'mean': 0.4666666666666667
            },
        }
        # The numerical branch of ``from_stats_dict`` reads COUNT, NUM_NA,
        # MEAN, MIN, MAX, QUANTILES (3-list) plus the by-class dict.
        stats_dict = {
            ColStatType.COUNT: Stats(6, False),
            ColStatType.NUM_NA: Stats(0, False),
            ColStatType.MEAN: Stats(0.5, False),
            ColStatType.MIN: Stats(0.1, False),
            ColStatType.MAX: Stats(0.9, False),
            ColStatType.QUANTILES: Stats([0.2, 0.5, 0.8], False),
            ColStatType.SUMMARY_STATS_BY_CLASS: Stats(by_class, False),
        }
        out = ColumnStats.from_stats_dict(Stype.numerical, stats_dict)
        assert out.bp_stats == by_class

    def test_no_by_class_yields_none(self):
        """Absent the stat (regression/binary, or any column the diskgraph
        side didn't group), ``bp_stats`` stays ``None`` — no regression vs.
        pre-cutover behavior."""
        from kumo_core.types import ColStatType
        from kumoapi.typing import Stype

        from kumo.stats.stats import Stats
        from rest.config.table import ColumnStats

        stats_dict = {
            ColStatType.COUNT: Stats(6, False),
            ColStatType.NUM_NA: Stats(0, False),
            ColStatType.MEAN: Stats(0.5, False),
            ColStatType.MIN: Stats(0.1, False),
            ColStatType.MAX: Stats(0.9, False),
            ColStatType.QUANTILES: Stats([0.2, 0.5, 0.8], False),
        }
        out = ColumnStats.from_stats_dict(Stype.numerical, stats_dict)
        assert out.bp_stats is None
