# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TDD unit tests for the TT-NG batch-prediction durable gen->ingest staging
bridge (distributed-correctness fix).

These cover the gen-side durable staging persist (C-gen) and the materialize-side
re-ingest-from-durable bridge (C-mat). The diskgraph wheel
(``execute_tt_gen`` / ``execute_tt_ingest``) is stubbed so the tests exercise the
kumo-side durability plumbing, not the Rust.
"""
from __future__ import annotations

import importlib
import os
import os.path as osp
import sys
from test.data.generic.ttng_v3_test_bundle import write_v3_bundle
from types import SimpleNamespace

bp = importlib.import_module('kumo.data.generic.diskgraph_batch_pred')
stg = importlib.import_module('kumo.data.generic.diskgraph_staging')
# Import the bundle-validation module BEFORE any test stubs
# ``sys.modules['diskgraph']`` (real top-level ``from diskgraph import
# TimeUnit``); avoids an order-dependent ImportError.
importlib.import_module('kumo.data.generic.diskgraph_train_table')


# --------------------------------------------------------------------------- #
# Test 1: durable staging path shape (parallels the train helper test).
# --------------------------------------------------------------------------- #
def test_ttng_persisted_pred_staging_dir_path_shape():
    assert (stg.ttng_persisted_pred_staging_dir('/home/pred_table') ==
            '/home/pred_table.ttng_staging')
    # Trailing slash is normalized so it does not yield a double-suffix.
    assert (stg.ttng_persisted_pred_staging_dir('/home/pred_table/') ==
            '/home/pred_table.ttng_staging')


# --------------------------------------------------------------------------- #
# Test 2: gen writes the durable staging bundle (C-gen).
# --------------------------------------------------------------------------- #
def _stub_execute_tt_gen_writing_bundle(index_path, tt_id, *, rows=3):
    """Return a fake ``execute_tt_gen`` that writes a realistic staging bundle
    (a real DG-schema tt.parquet + tt_ingest_spec.json + manifest.json) into the
    ephemeral staging dir and reports ``rows`` rows. ``tt.parquet`` is a genuine
    parquet (slot-0 ENTITY1/TARGET1) so the serving-table replay in
    ``_gen_kept_staging_ttng`` can read it back."""
    staging = stg.ttng_staging_dir(index_path, tt_id)

    def _fake(plan_json, idx, tid):
        d = stg.ttng_staging_dir(idx, tid)
        os.makedirs(d, exist_ok=True)
        write_v3_bundle(osp.dirname(d), tid, rows=rows)
        return {'row_count': rows}

    return _fake, staging


def _stub_read_tt_parquet_batches(path):
    """Real batch reader for the stubbed ``diskgraph`` module: the serving-table
    replay iterates the bundle parquet via ``read_tt_parquet_batches``."""
    import pyarrow.parquet as pq
    yield from pq.ParquetDataset(path).read().to_batches()


def test_gen_kept_staging_persists_durable_bundle(tmp_path, monkeypatch):
    index_path = str(tmp_path / 'index')
    output_root = str(tmp_path / 'home' / 'pred_table')
    tt_id = stg.predict_staging_tt_id('bp-job')

    fake_gen, ephemeral = _stub_execute_tt_gen_writing_bundle(index_path, tt_id)
    diskgraph_mod = SimpleNamespace(
        execute_tt_gen=fake_gen,
        read_tt_parquet_batches=_stub_read_tt_parquet_batches)
    monkeypatch.setitem(sys.modules, 'diskgraph', diskgraph_mod)

    from kumo_core.types import TaskType
    rows, parts = bp._gen_kept_staging_ttng(
        '{}', index_path, tt_id=tt_id, output_root=output_root,
        task_type=TaskType.BINARY_CLASSIFICATION)

    assert (rows, parts) == (3, 1)
    durable = stg.ttng_persisted_pred_staging_dir(output_root)
    from kumo.data.generic.diskgraph_train_table import (
        resolve_durable_ttng_bundle,
    )
    published = resolve_durable_ttng_bundle(durable)
    # The WHOLE bundle must be persisted durably, not just tt.parquet.
    assert osp.isdir(osp.join(published, 'tt.parquet'))
    for name in ('tt_ingest_spec.json', 'manifest.json'):
        assert osp.isfile(osp.join(published, name)), name
    # The ephemeral copy is still kept (materialize on the same pod can use it).
    assert osp.isdir(osp.join(ephemeral, 'tt.parquet'))
    # And the customer-facing serving pred table is materialized at the pred
    # table path so GET /genpredtable_jobs/{id}/table_data can surface it.
    import pandas as pd
    serving = osp.join(output_root, 'part-00000.parquet')
    assert osp.isfile(serving)
    serving_df = pd.read_parquet(serving)
    # Node-pred serving schema: ENTITY (renamed), TIMESTAMP, arange; TARGET dropped.
    assert set(serving_df.columns) == {'ENTITY', 'TIMESTAMP', 'arange'}
    assert len(serving_df) == 3


def test_gen_kept_staging_skips_durable_without_output_root(
        tmp_path, monkeypatch):
    """Legacy / unit callers that pass no ``output_root`` keep only the
    ephemeral staging (back-compat: no durable copy attempted)."""
    index_path = str(tmp_path / 'index')
    tt_id = stg.predict_staging_tt_id('bp-job')
    fake_gen, ephemeral = _stub_execute_tt_gen_writing_bundle(index_path, tt_id)
    monkeypatch.setitem(sys.modules, 'diskgraph',
                        SimpleNamespace(execute_tt_gen=fake_gen))

    rows, _ = bp._gen_kept_staging_ttng('{}', index_path, tt_id=tt_id)
    assert rows == 3
    assert osp.isdir(osp.join(ephemeral, 'tt.parquet'))
    # No durable dir created anywhere under home.
    assert not osp.exists(str(tmp_path / 'home'))


# --------------------------------------------------------------------------- #
# Test 5: is_tt_extension_complete predicate.
# --------------------------------------------------------------------------- #
def test_is_tt_extension_complete_predicate(tmp_path):
    ext = str(tmp_path / 'ext')
    os.makedirs(ext, exist_ok=True)
    assert not stg.is_tt_extension_complete(ext)  # sentinel missing
    with open(osp.join(ext, 'diskgraph_manifest.json'), 'w') as f:
        f.write('{}')
    assert stg.is_tt_extension_complete(ext)


# --------------------------------------------------------------------------- #
# Test 6: PredictionConfig carries the durable staging fields, back-compat.
# --------------------------------------------------------------------------- #
def test_prediction_config_durable_staging_fields_roundtrip(tmp_path):
    from kumo_core.job_config import PredictionConfig

    from io_utils.json import load_json, save_json

    cfg = PredictionConfig(
        home_dir='/h',
        train_dir='/t',
        task_name='task',
        model_dir='/m',
        output_dir='/o',
        return_embeddings=False,
        return_predictions=True,
        pred_table_path='/pred',
        diskgraph_pred_staging_dir='/pred.ttng_staging',
        diskgraph_pred_target_tt_id='pred_bp-job',
    )
    path = str(tmp_path / 'pred_config.json')
    save_json(cfg, path)
    loaded = load_json(path, PredictionConfig)
    assert loaded.diskgraph_pred_staging_dir == '/pred.ttng_staging'
    # The SOURCE staging is located by path (durable_staging_dir); its id is
    # read from the bundle spec, so only the TARGET extension tt_id (BP-job) is
    # carried on the config, and it round-trips.
    assert loaded.diskgraph_pred_target_tt_id == 'pred_bp-job'


def test_prediction_config_durable_staging_defaults_are_none():
    """Legacy / non-DG configs leave the new fields None (byte-identical)."""
    from kumo_core.job_config import PredictionConfig
    cfg = PredictionConfig(home_dir='/h', train_dir='/t', task_name='task',
                           model_dir='/m', output_dir='/o',
                           return_embeddings=False, return_predictions=True,
                           pred_table_path='/pred')
    assert cfg.diskgraph_pred_staging_dir is None
    assert cfg.diskgraph_pred_target_tt_id is None


# --------------------------------------------------------------------------- #
# Orphan-FK skip counters on the PREDICT ingest shape: same chokepoint as
# train (``ensure_stage_ingested``), just ``predict_stage_name='predict'``.
# A skipped predict seed means a missing prediction row — warn with
# kind='predict', and re-check reused extensions from the persisted manifest.
# --------------------------------------------------------------------------- #
def _write_durable_pred_staging(durable_dir):
    """Complete no-split predict bundle: tt.parquet + spec (split_stages=None)."""
    source_dir = write_v3_bundle(
        osp.dirname(durable_dir), 'pred-gen-job', rows=3)
    from kumo.data.generic.diskgraph_train_table import (
        persist_durable_ttng_bundle,
    )
    persist_durable_ttng_bundle(source_dir, durable_dir)


def _stub_pred_ingest_with_skips(monkeypatch, *, membership=0, rows=10):
    import json

    def _fake_ingest(**kwargs):
        tt_id = kwargs['tt_id']
        ext = osp.join(kwargs['index_path'], 'train_tables', tt_id)
        os.makedirs(ext, exist_ok=True)
        result = {
            'tt_extension_path': ext,
            'key_failures': {
                'parse_failures': 0,
                'lookup_failures': 0,
                'membership_failures': membership,
                'samples': [],
                'by_context': {
                    'dst_lookup:predict_seeds_ENTITY1_to_users_id': {
                        'parse_failures': 0,
                        'lookup_failures': 0,
                        'membership_failures': membership,
                    },
                },
            },
            'task_dataset': {
                'stages': {
                    'predict': {
                        'row_count': rows
                    }
                }
            },
        }
        with open(osp.join(ext, 'diskgraph_manifest.json'), 'w') as f:
            json.dump(result, f)
        return result

    monkeypatch.setitem(sys.modules, 'diskgraph',
                        SimpleNamespace(execute_tt_ingest=_fake_ingest))


def test_predict_orphan_skips_warn(tmp_path, monkeypatch, caplog):
    index_path = str(tmp_path / 'index')
    durable = str(tmp_path / 'home' / 'pred_table.ttng_staging')
    _write_durable_pred_staging(durable)
    _stub_pred_ingest_with_skips(monkeypatch, membership=4, rows=10)
    with caplog.at_level('WARNING'):
        stg.ensure_stage_ingested(index_path=index_path,
                                  target_tt_id='pred-job-1',
                                  staging_dir=durable,
                                  predict_stage_name='predict')
    warnings = [
        r.getMessage() for r in caplog.records
        if 'seed FK reference(s)' in r.getMessage()
    ]
    assert len(warnings) == 1
    assert warnings[0].startswith('TT-NG predict:')
    assert '~40.0%' in warnings[0]


def test_predict_orphan_skips_rechecked_on_reused_extension(
        tmp_path, monkeypatch, caplog):
    index_path = str(tmp_path / 'index')
    durable = str(tmp_path / 'home' / 'pred_table.ttng_staging')
    _write_durable_pred_staging(durable)
    _stub_pred_ingest_with_skips(monkeypatch, membership=6, rows=10)
    with caplog.at_level('WARNING'):
        stg.ensure_stage_ingested(index_path=index_path,
                                  target_tt_id='pred-job-2',
                                  staging_dir=durable,
                                  predict_stage_name='predict')
    caplog.clear()

    # Reused already-published extension: the manifest re-check must fire the
    # knob for later consumers, not just the ingesting job.
    monkeypatch.setenv('KUMO_TTNG_ORPHAN_FAIL_PCT', '50')
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match='reused extension'):
        stg.ensure_stage_ingested(index_path=index_path,
                                  target_tt_id='pred-job-2',
                                  staging_dir=durable,
                                  predict_stage_name='predict')
