# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the explicit-``manifest_path`` seed path of
:meth:`kumo.task.task.Task.seed_train_table_col_stats_from_manifest`.

Background
----------
Under TT-NG (FK-free on-disk DiskGraph), BINARY_CLASSIFICATION /
MULTICLASS node-prediction training failed with
``KeyError: CATEGORY_COUNTS`` because the train-table *target* column's
stats were never seeded into the backend col-stats cache. The TT-NG
train-table generator (``generate_train_table.py``) does NOT write the
``{train_table_path}.manifest.json`` sidecar the legacy seeder resolves,
so the seeder no-op'd. ttgen DOES compute the stat, but it lives in the
producer/gen-traintable manifest at
``<dg>/train_tables/<tt_id>/manifest.json`` instead.

The approved fix gives
:meth:`Task.seed_train_table_col_stats_from_manifest` an optional
``manifest_path`` argument. When supplied, the seeder reads that explicit
producer manifest directly (bypassing the connector ``_get_path`` resolver
and the sidecar derivation). When ``None``, the legacy behavior is
unchanged.

These tests mirror the established patterns in
``test/task/test_seed_train_table_col_stats_smoke.py`` (Task construction
over a ``PandasConnector``) and
``test/engine/test_materialize_task_dataset.py`` (patching
``get_validated_artifact_bundle``).
"""
import json
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
from kumo_core.types import ColStatType
from kumoapi.typing import Dtype, Stype

from kumo.connector.pandas_connector import PandasConnector
from kumo.graph.graph import Graph
from kumo.task.task import Task

# The seeder imports ``get_validated_artifact_bundle`` lazily from this
# module, so patching it at the source is what intercepts the call.
_BUNDLE_PATCH_TARGET = (
    'kumo.diskgraph.diskgraph_native_tt_artifacts.get_validated_artifact_bundle')


def _make_bc_task():
    """Build a minimal BC node-prediction Task with a categorical
    ``Index`` target column named ``TARGET`` over a ``PandasConnector``."""
    graph_df = pd.DataFrame({
        'USER_ID': np.arange(20),
        'COUNTRY': ['US', 'CA'] * 10,
    })
    connector = PandasConnector({'USERS': graph_df})
    graph = Graph(name='bc_seed_graph')
    graph.connect(connector)
    graph.add('USERS', pkey=dict(name='USER_ID', dtype=Dtype.int))
    graph.infer_metadata()

    # The train-table target is a boolean/categorical label. We declare
    # it as a categorical ``Index`` target named ``TARGET`` (the logical
    # slot-0 name the producer's ``TARGET1`` aliases to).
    train_df = pd.DataFrame({
        'USER_ID': np.arange(20),
        'TARGET': ([1, 0] * 10),
    })
    task = Task(
        graph,
        task_type='binary_classification',
        train_table_name='TRAINING_TARGET',
        target_col=dict(name='TARGET', stype=Stype.categorical,
                        dtype=Dtype.int),
        fkeys=[{
            'name': 'USER_ID',
            'dst_table': 'USERS',
        }],
        data_source=train_df,
        train_table_data_version='bc-seed-v1',
        ignore_encoders=True,
        task_name='bc_seed_task',
    )
    return task


def _make_regression_task():
    graph_df = pd.DataFrame({
        'USER_ID': np.arange(20),
        'COUNTRY': ['US', 'CA'] * 10,
    })
    connector = PandasConnector({'USERS': graph_df})
    graph = Graph(name='regression_seed_graph')
    graph.connect(connector)
    graph.add('USERS', pkey=dict(name='USER_ID', dtype=Dtype.int))
    graph.infer_metadata()

    train_df = pd.DataFrame({
        'USER_ID': np.arange(20),
        'TARGET': np.arange(20),
    })
    return Task(
        graph,
        task_type='regression',
        train_table_name='TRAINING_TARGET',
        target_col=dict(name='TARGET', stype=Stype.numerical, dtype=Dtype.int),
        fkeys=[{
            'name': 'USER_ID',
            'dst_table': 'USERS',
        }],
        data_source=train_df,
        train_table_data_version='regression-seed-v1',
        ignore_encoders=True,
        task_name='regression_seed_task',
    )


def _seed_basic_target_stats(task):
    """Seed the basic exact/approx col-stats the categorical target encoder
    needs beyond CATEGORY_COUNTS (COUNT / NUM_NA / NUM_UNIQUE / NA_FRACTION),
    mirroring what DiskGraph ingest + materialization populate into the backend
    col-stats cache in the LIVE path. In production these come from
    ``compute_exact_stats`` over the materialized train table; here we seed them
    directly so the encoder-stat fetch exercises the real "everything present
    after ingestion + materialization" contract while these unit tests focus on
    the CATEGORY_COUNTS manifest/sidecar seeding path."""
    from kumo.data.backend_cache import get_active_cache
    from kumo.stats.stats import Stats

    target_col = task.train_table.target_col
    col_cfg = target_col.base_config()
    data_version = task.train_table.data_version
    cache = get_active_cache()
    basic = {
        ColStatType.COUNT: (Stats(value=20, approx=False,
                                  data_version=data_version, num_entries=20)),
        ColStatType.NUM_NA: (Stats(value=0, approx=False,
                                   data_version=data_version, num_entries=20)),
        ColStatType.NUM_UNIQUE:
        (Stats(value=2, approx=False, data_version=data_version,
               num_entries=20)),
        ColStatType.NA_FRACTION: (Stats(value=0.0, approx=True,
                                        data_version=data_version,
                                        num_entries=20)),
    }
    cache.add_col_stats(task.train_table.fingerprint(), {col_cfg: basic})
    cache.save_col_stats_cache(task.train_table.fingerprint(), data_version)


def _fake_producer_bundle():
    """A bundle shaped like the output of
    ``get_validated_artifact_bundle`` over the TT-NG producer manifest:
    ``parsed_train_table_col_stats.stats`` keyed by diskgraph's canonical
    physical column names (``TARGET1`` here)."""
    return {
        'parsed_train_table_col_stats': {
            'version': 1,
            'kind': 'tt_train_table_col_stats',
            'stats_format': 'kumo_col_stats_v1',
            'stats': {
                'TARGET1__kumo.categorical__CATEGORY_COUNTS__False': {
                    'value': [[0, 1520], [1, 103]],
                    'num_entries': 1623,
                },
            },
        },
    }


def test_encoder_stats_cache_miss_fails_loud_without_connector_fallback():
    """TT-NG fails loud on a col-stats cache miss: there is intentionally no
    manifest re-probe / whole-table connector-read fallback. When the backend
    cache was never seeded (ingest + materialization did not run), the
    strict-path encoder-stat fetch must raise a precise ``RuntimeError`` naming
    the table, column, and missing stat types -- NOT silently re-read the
    connector to paper over the upstream gap."""
    task = _make_regression_task()
    assert task.train_table.target_col is not None

    # A connector that *could* satisfy a legacy whole-table read; the point is
    # that the strict path must NOT consult it -- it must fail loud instead.
    class _NativeTrainTableConnector:
        name = 'pandas'

        def read_table(self, table_name, cols):  # pragma: no cover
            raise AssertionError(
                'connector read_table must not be consulted on a col-stats '
                'cache miss under the TT-NG strict path')

    task.train_table._connector = _NativeTrainTableConnector()

    with pytest.raises(RuntimeError) as excinfo:
        task._get_encoder_stats_from_cache_or_fail(task.train_table.target_col)
    msg = str(excinfo.value)
    assert 'TRAINING_TARGET' in msg
    assert 'TARGET' in msg
    assert 'missing' in msg.lower()
    assert 'no manifest or connector fallback' in msg.lower()


def test_seed_from_explicit_producer_manifest_populates_category_counts(
    tmp_path, ):
    """The TT-NG fix: passing ``manifest_path`` seeds the TARGET column's
    CATEGORY_COUNTS into the cache and a subsequent encoder-stat fetch
    populates the Index encoder ``_stats``."""
    task = _make_bc_task()
    assert task.train_table.target_col is not None
    assert task.train_table.target_col.name == 'TARGET'

    # The producer manifest must exist on disk so the seeder's
    # ``fsutil.exists`` guard passes; its contents are irrelevant because
    # ``get_validated_artifact_bundle`` is patched to return the parsed
    # bundle directly (the established pattern in
    # test/engine/test_materialize_task_dataset.py).
    manifest_path = str(tmp_path / 'train_tables' / 'tt-id' / 'manifest.json')
    import os
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump({'version': 1}, f)

    with patch(_BUNDLE_PATCH_TARGET, return_value=_fake_producer_bundle()):
        seeded = task.seed_train_table_col_stats_from_manifest(
            manifest_path=manifest_path)
    assert seeded is True

    # Basic exact/approx stats (COUNT / NUM_NA / NUM_UNIQUE / NA_FRACTION) come
    # from compute_exact_stats over the materialized table in the live path; in
    # this unit test seed them the same way DG ingest/materialization populates
    # the backend cache, so the encoder-stat fetch sees the full contract.
    _seed_basic_target_stats(task)

    # The col-stats cache now holds CATEGORY_COUNTS for the TARGET column
    # (the producer key ``TARGET1`` aliases to logical ``TARGET``).
    target_stats = task._get_encoder_stats_from_cache_or_fail(
        task.train_table.target_col)
    assert ColStatType.CATEGORY_COUNTS in target_stats

    # And a full encoder materialization populates the Index encoder's
    # ``_stats[CATEGORY_COUNTS]`` (the exact path that previously raised
    # ``KeyError: CATEGORY_COUNTS`` in TargetEncoder.from_encoder). The
    # BC/MULTICLASS target uses an ``Index`` encoder whose only required
    # stat is CATEGORY_COUNTS; attach it (the task is built with
    # ``ignore_encoders=True`` so encoder inference is skipped) and verify
    # ``fetch_and_set_encoder_stats`` (via ``materialize_encoders``)
    # pulls the seeded counts out of the cache onto the encoder.
    from kumo.graph.column import Index
    from kumo.graph.utils import materialize_encoders
    task.train_table.target_col.encoder = Index()
    assert ColStatType.CATEGORY_COUNTS in (
        task.train_table.target_col.encoder.required_stats)
    _encoders, target = materialize_encoders(task.train_table)
    assert target is not None
    assert ColStatType.CATEGORY_COUNTS in target.encoder._stats


def test_explicit_missing_manifest_returns_false_without_raising(tmp_path):
    """An explicit ``manifest_path`` that does not exist is a best-effort
    miss (returns False), not an error."""
    task = _make_bc_task()
    missing = str(tmp_path / 'does' / 'not' / 'exist' / 'manifest.json')
    assert task.seed_train_table_col_stats_from_manifest(
        manifest_path=missing) is False


def test_lp_shape_missing_block_degrades_to_false(tmp_path):
    """FK-free LP: the producer omits the ``train_table_col_stats`` block
    when the payload is empty (LP has no scalar target; train-table cols are
    populated only at materialize). With an explicit producer manifest whose
    bundle lacks the block, the seeder must degrade to False — NOT raise —
    because there is nothing to seed. Guards the core LP path that the fix
    newly routes through this code."""
    task = _make_bc_task()
    manifest_path = str(tmp_path / 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({'version': 1}, f)

    tt_type = type(task.train_table)
    with patch(_BUNDLE_PATCH_TARGET, return_value={}), \
         patch.object(tt_type, 'target_col', new_callable=PropertyMock,
                      return_value=None), \
         patch.object(tt_type, 'cols', new_callable=PropertyMock,
                      return_value=[]):
        seeded = task.seed_train_table_col_stats_from_manifest(
            manifest_path=manifest_path)
    assert seeded is False


def test_node_pred_missing_block_still_raises(tmp_path):
    """Node prediction (target_col present): a missing
    ``train_table_col_stats`` block means the categorical-target stat the
    encoder needs is gone — the seeder must raise loudly, not degrade."""
    task = _make_bc_task()
    manifest_path = str(tmp_path / 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({'version': 1}, f)

    with patch(_BUNDLE_PATCH_TARGET, return_value={}):
        with pytest.raises(RuntimeError, match='parsed train_table_col_stats'):
            task.seed_train_table_col_stats_from_manifest(
                manifest_path=manifest_path)
