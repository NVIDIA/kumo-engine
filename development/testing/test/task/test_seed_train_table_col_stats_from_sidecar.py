# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for seeding train-table col stats from the DURABLE
``train_table_col_stats.json`` sidecar DiskGraph writes during TT-NG
gen/ingest.

Background
----------
TT-NG node-prediction (BINARY_CLASSIFICATION / MULTICLASS) training failed
with ``KeyError: CATEGORY_COUNTS`` because the target column's stats were
never seeded into the backend col-stats cache.

The producer (DiskGraph) writes a self-describing, durable
``train_table_col_stats.json`` sidecar (kumo schema
``tt_train_table_col_stats`` v1, ``stats_format: kumo_col_stats_v1``)
beside the ingested train-table extension AND in the persisted gen→ingest
staging bundle. Master's seeder, however, reads col-stats only through the
*transient gen* ``manifest.json`` bundle via
:func:`get_validated_artifact_bundle`, which validates EVERY artifact the
gen manifest references (train_table.path, stats.path, ...). On the PQL
node-pred path those sibling artifacts point at the consumed scratch
staging that no longer exists, so ``get_validated_artifact_bundle`` raises,
the seeder swallows it (best-effort ``return False``), and CATEGORY_COUNTS
is never seeded -> downstream ``KeyError``.

The fix reads the durable sidecar DIRECTLY (via
:func:`parse_train_table_col_stats_artifact`, which validates ONLY the
sidecar), independent of the gen manifest bundle. These tests pin that
behavior: they construct ONLY the durable sidecar (no gen manifest) and
assert CATEGORY_COUNTS seeds. On master there is no sidecar-reading entry
point, so they fail.
"""
import json
import os

import numpy as np
import pandas as pd
from kumo_core.types import ColStatType
from kumoapi.typing import Dtype, Stype

from kumo.connector.pandas_connector import PandasConnector
from kumo.graph.column import Column, ForeignKey, PrimaryKey
from kumo.graph.graph import Graph
from kumo.task.task import Task


def _make_bc_task():
    """Minimal BC node-prediction Task with a categorical ``TARGET``."""
    graph_df = pd.DataFrame({
        'USER_ID': np.arange(20),
        'COUNTRY': ['US', 'CA'] * 10,
    })
    connector = PandasConnector({'USERS': graph_df})
    graph = Graph(name='sidecar_bc_graph')
    graph.connect(connector)
    graph.add('USERS', pkey=PrimaryKey(name='USER_ID', dtype=Dtype.int))
    graph.infer_metadata()

    train_df = pd.DataFrame({
        'USER_ID': np.arange(20),
        'TARGET': ([1, 0] * 10),
    })
    return Task(
        graph,
        task_type='binary_classification',
        train_table_name='TRAINING_TARGET',
        target_col=Column(name='TARGET', stype=Stype.categorical,
                          dtype=Dtype.int),
        fkeys=[ForeignKey(name='USER_ID', dst_table='USERS')],
        data_source=train_df,
        train_table_data_version='sidecar-bc-v1',
        ignore_encoders=True,
        task_name='sidecar_bc_task',
    )


def _seed_basic_target_stats(task):
    """Seed the basic exact/approx col-stats the categorical target encoder
    needs beyond CATEGORY_COUNTS (COUNT / NUM_NA / NUM_UNIQUE / NA_FRACTION),
    mirroring what DiskGraph ingest + materialization populate into the backend
    col-stats cache in the LIVE path. In production these come from
    ``compute_exact_stats`` over the materialized train table; here we seed them
    directly so the encoder-stat fetch exercises the real "everything present
    after ingestion + materialization" contract while this test focuses on the
    CATEGORY_COUNTS sidecar seeding path."""
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


def _write_durable_sidecar(path: str) -> None:
    """Write a ``train_table_col_stats.json`` in the EXACT shape the DG
    producer emits (``kumo_contract_bridge::write_train_table_col_stats_
    artifact`` -> ``parse_train_table_col_stats_artifact``)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'version': 1,
        'kind': 'tt_train_table_col_stats',
        'stats_format': 'kumo_col_stats_v1',
        'stats': {
            'TARGET1__kumo.categorical__CATEGORY_COUNTS__False': {
                'value': [[0, 1520], [1, 103]],
                'approx': False,
                'data_version': 'sidecar-bc-v1',
                'num_entries': 1623,
            },
        },
    }
    with open(path, 'w') as f:
        json.dump(payload, f)


def test_seed_from_durable_sidecar_populates_category_counts(tmp_path):
    """The fix: pointing the seeder at the durable
    ``train_table_col_stats.json`` sidecar (no gen manifest bundle present)
    seeds CATEGORY_COUNTS for the TARGET column and a subsequent encoder
    stat fetch resolves. On master there is no sidecar entry point, so this
    fails."""
    task = _make_bc_task()
    assert task.train_table.target_col is not None

    sidecar = str(tmp_path / 'train_tables' / 'tt-id' /
                  'train_table_col_stats.json')
    _write_durable_sidecar(sidecar)

    seeded = task.seed_train_table_col_stats_from_sidecar(sidecar)
    assert seeded is True

    # Basic exact/approx stats (COUNT / NUM_NA / NUM_UNIQUE / NA_FRACTION) come
    # from compute_exact_stats over the materialized table in the live path; in
    # this unit test seed them the same way DG ingest/materialization populates
    # the backend cache, so the encoder-stat fetch sees the full contract.
    _seed_basic_target_stats(task)

    target_stats = task._get_encoder_stats_from_cache_or_fail(
        task.train_table.target_col)
    assert ColStatType.CATEGORY_COUNTS in target_stats


def test_seed_from_durable_sidecar_drives_encoder_build(tmp_path):
    """Full Index-encoder materialization populates
    ``encoder._stats[CATEGORY_COUNTS]`` from the sidecar-seeded cache --
    the exact path that previously raised ``KeyError: CATEGORY_COUNTS`` in
    ``TargetEncoder.from_encoder``."""
    task = _make_bc_task()
    sidecar = str(tmp_path / 'train_table_col_stats.json')
    _write_durable_sidecar(sidecar)

    assert task.seed_train_table_col_stats_from_sidecar(sidecar) is True

    from kumoapi.encoder import Index

    from kumo.graph.utils import materialize_encoders
    target_col = task.train_table.target_col
    assert target_col is not None
    target_col.encoder = Index()
    _encoders, target = materialize_encoders(task.train_table)
    assert target is not None
    assert ColStatType.CATEGORY_COUNTS in target.encoder._stats


def test_seed_from_missing_sidecar_returns_false(tmp_path):
    """A missing durable sidecar is a best-effort miss (returns False),
    not an error -- mirrors the manifest seeder's missing-file contract."""
    task = _make_bc_task()
    missing = str(tmp_path / 'nope' / 'train_table_col_stats.json')
    assert task.seed_train_table_col_stats_from_sidecar(missing) is False


def test_seed_from_durable_sidecar_is_idempotent(tmp_path):
    """The seeded latch flips once; a second call is a no-op (False)."""
    task = _make_bc_task()
    sidecar = str(tmp_path / 'train_table_col_stats.json')
    _write_durable_sidecar(sidecar)
    assert task.seed_train_table_col_stats_from_sidecar(sidecar) is True
    assert task.seed_train_table_col_stats_from_sidecar(sidecar) is False
