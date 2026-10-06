# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TT-NG training durable gen->ingest staging bridge.

Cover ``ensure_stage_ingested`` in its train shape (``predict_stage_name=None``
-> split train/val/test ingest). ``execute_tt_ingest`` is stubbed.
"""
from __future__ import annotations

import importlib
import json
import os
import os.path as osp
import sys
from test.data.generic.ttng_v3_test_bundle import write_v3_bundle
from types import SimpleNamespace

import pytest

bp = importlib.import_module('kumo.data.generic.diskgraph_staging')
# Import the bundle-validation module BEFORE any test stubs
# ``sys.modules['diskgraph']``: it does a real ``from diskgraph import
# TimeUnit`` at module top, so importing it under the stub fails with an
# order-dependent ImportError.
ttng = importlib.import_module('kumo.data.generic.diskgraph_train_table')


def _write_durable_train_staging(durable_dir):
    """Write a complete TT-NG *train* staging bundle: tt.parquet +
    tt_ingest_spec.json declaring split_stages with per-stage part files."""
    parts = ('train-0.parquet', 'val-0.parquet', 'test-0.parquet')
    producer_dir = write_v3_bundle(
        osp.dirname(durable_dir),
        'gen-tt-job',
        split_stages=[
            {'stage': 'train', 'parts': ['train-0.parquet']},
            {'stage': 'val', 'parts': ['val-0.parquet']},
            {'stage': 'test', 'parts': ['test-0.parquet']},
        ],
        declared_parts=parts,
    )
    ttng.persist_durable_ttng_bundle(producer_dir, durable_dir)


def _write_complete_train_extension(index_path, tt_id):
    """A COMPLETE extension carries the ingest-published sentinel."""
    ext = osp.join(index_path, 'train_tables', tt_id)
    os.makedirs(ext, exist_ok=True)
    with open(osp.join(ext, 'diskgraph_manifest.json'), 'w') as f:
        f.write('{}')
    return ext


# --------------------------------------------------------------------------- #
# ensure_stage_ingested (train shape: predict_stage_name=None) — 4 cases.
# --------------------------------------------------------------------------- #
def test_ensure_stage_ingested_train_reingests_when_absent(tmp_path,
                                                           monkeypatch):
    index_path = str(tmp_path / 'index')
    durable = str(tmp_path / 'home' / 'train_table.ttng_staging')
    _write_durable_train_staging(durable)

    calls = []

    def _fake_ingest(**kwargs):
        calls.append(kwargs)
        tt_id = kwargs['tt_id']
        _write_complete_train_extension(kwargs['index_path'], tt_id)
        return {'tt_extension_path':
                osp.join(kwargs['index_path'], 'train_tables', tt_id)}

    monkeypatch.setitem(sys.modules, 'diskgraph',
                        SimpleNamespace(execute_tt_ingest=_fake_ingest))

    ext = bp.ensure_stage_ingested(
        index_path=index_path, target_tt_id='gen-tt-job',
        staging_dir=durable, predict_stage_name=None)

    assert ext == osp.join(index_path, 'train_tables', 'gen-tt-job')
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs['index_path'] == index_path
    assert kwargs['tt_id'] == 'gen-tt-job'
    # Train ingest passes no predict_stage_name (split train/val/test ingest).
    assert 'predict_stage_name' not in kwargs
    # The durable bundle was copied into the index-local staging dir.
    assert osp.isdir(osp.join(kwargs['staging_dir'], 'tt.parquet'))


def test_ensure_stage_ingested_accepts_selected_bundle_without_pointer(
        tmp_path, monkeypatch):
    index_path = str(tmp_path / 'index')
    durable = str(tmp_path / 'home' / 'train_table.ttng_staging')
    _write_durable_train_staging(durable)
    selected = ttng.resolve_durable_ttng_bundle(durable)
    calls = []

    def _fake_ingest(**kwargs):
        calls.append(kwargs)
        _write_complete_train_extension(kwargs['index_path'], kwargs['tt_id'])
        return {
            'tt_extension_path':
            osp.join(kwargs['index_path'], 'train_tables', kwargs['tt_id']),
        }

    monkeypatch.setitem(sys.modules, 'diskgraph',
                        SimpleNamespace(execute_tt_ingest=_fake_ingest))

    ext = bp.ensure_stage_ingested(
        index_path=index_path,
        target_tt_id='gen-tt-job',
        staging_dir=selected,
    )

    assert ext == osp.join(index_path, 'train_tables', 'gen-tt-job')
    assert len(calls) == 1
    assert osp.isdir(osp.join(calls[0]['staging_dir'], 'tt.parquet'))


def test_ensure_stage_ingested_train_noop_when_present(tmp_path, monkeypatch):
    index_path = str(tmp_path / 'index')
    durable = str(tmp_path / 'home' / 'train_table.ttng_staging')
    _write_durable_train_staging(durable)
    # COMPLETE extension already present -> idempotent no-op.
    _write_complete_train_extension(index_path, 'gen-tt-job')

    calls = []
    monkeypatch.setitem(
        sys.modules, 'diskgraph',
        SimpleNamespace(execute_tt_ingest=lambda **k: calls.append(k)))

    ext = bp.ensure_stage_ingested(
        index_path=index_path, target_tt_id='gen-tt-job',
        staging_dir=durable, predict_stage_name=None)
    assert ext == osp.join(index_path, 'train_tables', 'gen-tt-job')
    assert calls == []  # no re-ingest


def test_ensure_stage_ingested_train_reingests_when_extension_partial(
        tmp_path, monkeypatch):
    """A snapshot restore / interrupted publish can leave the train extension
    dir without its final ``diskgraph_manifest.json``; the helper must delete
    the partial dir and re-ingest rather than skip on bare dir-existence."""
    index_path = str(tmp_path / 'index')
    durable = str(tmp_path / 'home' / 'train_table.ttng_staging')
    _write_durable_train_staging(durable)

    partial = osp.join(index_path, 'train_tables', 'gen-tt-job')
    os.makedirs(partial, exist_ok=True)
    with open(osp.join(partial, 'manifest.json'), 'w') as f:
        f.write('{"stale": true}')  # sidecar present, sentinel missing

    calls = []

    def _fake_ingest(**kwargs):
        calls.append(kwargs['tt_id'])
        _write_complete_train_extension(kwargs['index_path'], kwargs['tt_id'])
        return {}

    monkeypatch.setitem(sys.modules, 'diskgraph',
                        SimpleNamespace(execute_tt_ingest=_fake_ingest))

    ext = bp.ensure_stage_ingested(
        index_path=index_path, target_tt_id='gen-tt-job',
        staging_dir=durable, predict_stage_name=None)

    assert calls == ['gen-tt-job']  # partial dir triggered a re-ingest
    assert not osp.exists(osp.join(ext, 'manifest.json'))  # stale wiped
    assert osp.isfile(osp.join(ext, 'diskgraph_manifest.json'))


def test_ensure_stage_ingested_train_raises_when_staging_missing(tmp_path):
    index_path = str(tmp_path / 'index')
    durable = str(tmp_path / 'home' / 'train_table.ttng_staging')  # not created
    with pytest.raises(RuntimeError, match='staging bundle is missing'):
        bp.ensure_stage_ingested(
            index_path=index_path, target_tt_id='gen-tt-job',
            staging_dir=durable, predict_stage_name=None)


def test_ensure_stage_ingested_train_is_idempotent_on_rerun(tmp_path,
                                                            monkeypatch):
    """Calling twice (pod retry) re-ingests at most once."""
    index_path = str(tmp_path / 'index')
    durable = str(tmp_path / 'home' / 'train_table.ttng_staging')
    _write_durable_train_staging(durable)

    calls = []

    def _fake_ingest(**kwargs):
        calls.append(kwargs['tt_id'])
        _write_complete_train_extension(kwargs['index_path'], kwargs['tt_id'])
        return {}

    monkeypatch.setitem(sys.modules, 'diskgraph',
                        SimpleNamespace(execute_tt_ingest=_fake_ingest))

    bp.ensure_stage_ingested(
        index_path=index_path, target_tt_id='gen-tt-job',
        staging_dir=durable, predict_stage_name=None)
    bp.ensure_stage_ingested(
        index_path=index_path, target_tt_id='gen-tt-job',
        staging_dir=durable, predict_stage_name=None)
    assert calls == ['gen-tt-job']  # exactly one ingest across both calls


# --------------------------------------------------------------------------- #
# Orphan-FK skip counters: ingest membership-verifies seed FKs against the
# training graph and reports skips under ``key_failures``; the helper must
# warn (escalating >=50%), re-check reused extensions from the persisted
# manifest, and hard-fail only via KUMO_TTNG_ORPHAN_FAIL_PCT.
# --------------------------------------------------------------------------- #
def _ingest_with_skips(monkeypatch, *, parse=0, lookup=0, membership=0,
                       row_counts=()):
    """Stub ``execute_tt_ingest`` returning skip counters + stage row counts,
    persisting the same shape into the extension's manifest (as the real
    ingest does) so the reused-extension re-check path is exercised too."""

    def _fake_ingest(**kwargs):
        tt_id = kwargs['tt_id']
        ext = _write_complete_train_extension(kwargs['index_path'], tt_id)
        stages = {
            f'stage{i}': {'row_count': n} for i, n in enumerate(row_counts)
        }
        result = {
            'tt_extension_path':
            osp.join(kwargs['index_path'], 'train_tables', tt_id),
            'key_failures': {
                'parse_failures': parse,
                'lookup_failures': lookup,
                'membership_failures': membership,
                'samples': ['dst_lookup:train_seeds_ENTITY1_to_users_id: 50'],
                'by_context': {
                    'dst_lookup:train_seeds_ENTITY1_to_users_id': {
                        'parse_failures': parse,
                        'lookup_failures': lookup,
                        'membership_failures': membership,
                    },
                },
            },
            'task_dataset': {'stages': stages},
        }
        with open(osp.join(ext, 'diskgraph_manifest.json'), 'w') as f:
            json.dump(result, f)
        return result

    monkeypatch.setitem(sys.modules, 'diskgraph',
                        SimpleNamespace(execute_tt_ingest=_fake_ingest))


def _run_train_ingest(tmp_path):
    index_path = str(tmp_path / 'index')
    durable = str(tmp_path / 'home' / 'train_table.ttng_staging')
    _write_durable_train_staging(durable)
    return bp.ensure_stage_ingested(
        index_path=index_path, target_tt_id='gen-tt-job',
        staging_dir=durable, predict_stage_name=None)


def test_orphan_skips_zero_stays_silent(tmp_path, monkeypatch, caplog):
    _ingest_with_skips(monkeypatch, lookup=0, membership=0,
                       row_counts=(80, 10, 10))
    with caplog.at_level('WARNING'):
        _run_train_ingest(tmp_path)
    assert not [r for r in caplog.records if 'skipped' in r.getMessage()]


def test_orphan_skips_missing_counters_tolerated(tmp_path, monkeypatch,
                                                 caplog):
    """An ingest result without ``key_failures`` (older wheel) must not warn
    or crash."""

    def _fake_ingest(**kwargs):
        _write_complete_train_extension(kwargs['index_path'], kwargs['tt_id'])
        return {}

    monkeypatch.setitem(sys.modules, 'diskgraph',
                        SimpleNamespace(execute_tt_ingest=_fake_ingest))
    with caplog.at_level('WARNING'):
        _run_train_ingest(tmp_path)
    assert not [r for r in caplog.records if 'skipped' in r.getMessage()]


def test_orphan_skips_warn_below_escalation(tmp_path, monkeypatch, caplog):
    # 5 skipped / 100 rows = 5% -> plain warning, not the cross-graph wording.
    _ingest_with_skips(monkeypatch, lookup=2, membership=3,
                       row_counts=(80, 10, 10))
    with caplog.at_level('WARNING'):
        _run_train_ingest(tmp_path)
    warnings = [r.getMessage() for r in caplog.records
                if 'skipped' in r.getMessage()]
    assert len(warnings) == 1
    assert '5 seed FK reference(s)' in warnings[0]
    assert '~5.0%' in warnings[0]
    assert 'by_context=' in warnings[0]
    assert 'different graph' not in warnings[0]


def test_orphan_skips_escalates_wording_at_50pct(tmp_path, monkeypatch,
                                                 caplog):
    # 60 skipped / 100 rows -> escalated "does not belong to this graph".
    _ingest_with_skips(monkeypatch, lookup=0, membership=60,
                       row_counts=(80, 10, 10))
    with caplog.at_level('WARNING'):
        _run_train_ingest(tmp_path)
    warnings = [r.getMessage() for r in caplog.records
                if 'skipped' in r.getMessage()]
    assert len(warnings) == 1
    assert 'generated against a different graph' in warnings[0]


def test_orphan_skips_hard_fail_env(tmp_path, monkeypatch):
    _ingest_with_skips(monkeypatch, lookup=0, membership=60,
                       row_counts=(80, 10, 10))
    monkeypatch.setenv('KUMO_TTNG_ORPHAN_FAIL_PCT', '50')
    with pytest.raises(RuntimeError, match='KUMO_TTNG_ORPHAN_FAIL_PCT'):
        _run_train_ingest(tmp_path)


def test_orphan_skips_hard_fail_env_below_threshold_warns_only(
        tmp_path, monkeypatch, caplog):
    _ingest_with_skips(monkeypatch, lookup=5, membership=0,
                       row_counts=(80, 10, 10))
    monkeypatch.setenv('KUMO_TTNG_ORPHAN_FAIL_PCT', '50')
    with caplog.at_level('WARNING'):
        _run_train_ingest(tmp_path)  # 5% < 50% -> no raise
    assert [r for r in caplog.records if 'skipped' in r.getMessage()]


def test_orphan_skips_invalid_env_ignored(tmp_path, monkeypatch, caplog):
    _ingest_with_skips(monkeypatch, lookup=0, membership=60,
                       row_counts=(80, 10, 10))
    monkeypatch.setenv('KUMO_TTNG_ORPHAN_FAIL_PCT', 'not-a-number')
    with caplog.at_level('WARNING'):
        _run_train_ingest(tmp_path)  # invalid knob -> warn-only, no raise
    messages = [r.getMessage() for r in caplog.records]
    assert any('Ignoring invalid' in m for m in messages)
    assert any('skipped' in m for m in messages)


def test_orphan_skips_parse_failures_counted(tmp_path, monkeypatch, caplog):
    """PK-key-type mismatch surfaces as parse failures — must not be silent."""
    _ingest_with_skips(monkeypatch, parse=100, row_counts=(80, 10, 10))
    with caplog.at_level('WARNING'):
        _run_train_ingest(tmp_path)
    warnings = [r.getMessage() for r in caplog.records
                if 'skipped' in r.getMessage()]
    assert len(warnings) == 1
    assert '100 seed FK reference(s)' in warnings[0]
    assert 'different graph' in warnings[0]  # 100% -> escalated


def test_orphan_skips_pct_clamped_for_lp_references(tmp_path, monkeypatch,
                                                    caplog):
    """LP rows carry one ref per target-list element, so skips can exceed
    rows; the reported percent is clamped at 100."""
    _ingest_with_skips(monkeypatch, lookup=350, row_counts=(80, 10, 10))
    with caplog.at_level('WARNING'):
        _run_train_ingest(tmp_path)
    warnings = [r.getMessage() for r in caplog.records
                if 'skipped' in r.getMessage()]
    assert len(warnings) == 1
    assert '~100.0%' in warnings[0]
    assert '350 seed FK reference(s)' in warnings[0]


def test_orphan_skips_unknown_denominator(tmp_path, monkeypatch, caplog):
    """Counters without stage row counts: warn loudly, say the denominator
    is unknown, and treat as escalated."""
    _ingest_with_skips(monkeypatch, membership=3, row_counts=())
    with caplog.at_level('WARNING'):
        _run_train_ingest(tmp_path)
    warnings = [r.getMessage() for r in caplog.records
                if 'skipped' in r.getMessage()]
    assert len(warnings) == 1
    assert 'unknown number of seed rows' in warnings[0]


def test_orphan_skips_escalates_at_exact_boundary(tmp_path, monkeypatch,
                                                  caplog):
    _ingest_with_skips(monkeypatch, membership=50, row_counts=(80, 10, 10))
    with caplog.at_level('WARNING'):
        _run_train_ingest(tmp_path)  # exactly 50.0%
    warnings = [r.getMessage() for r in caplog.records
                if 'skipped' in r.getMessage()]
    assert 'different graph' in warnings[0]


def test_orphan_skips_malformed_counters_nonfatal(tmp_path, monkeypatch,
                                                  caplog):
    """Diagnostics parsing must never kill the job (wheel skew)."""

    def _fake_ingest(**kwargs):
        _write_complete_train_extension(kwargs['index_path'], kwargs['tt_id'])
        return {'key_failures': 'garbage-not-a-dict'}

    monkeypatch.setitem(sys.modules, 'diskgraph',
                        SimpleNamespace(execute_tt_ingest=_fake_ingest))
    monkeypatch.setenv('KUMO_TTNG_ORPHAN_FAIL_PCT', '0')
    with caplog.at_level('WARNING'):
        _run_train_ingest(tmp_path)  # no raise despite fail-on-any knob
    assert any('could not parse ingest key_failures' in r.getMessage()
               for r in caplog.records)


def test_orphan_skips_hard_fail_message_carries_detail(tmp_path, monkeypatch):
    _ingest_with_skips(monkeypatch, membership=60, row_counts=(80, 10, 10))
    monkeypatch.setenv('KUMO_TTNG_ORPHAN_FAIL_PCT', '50')
    with pytest.raises(RuntimeError,
                       match='did not resolve to a node of the training'):
        _run_train_ingest(tmp_path)


def test_orphan_skips_rechecked_on_reused_extension(tmp_path, monkeypatch,
                                                    caplog):
    """The extension publishes BEFORE the counter check, so a retry / a later
    job on the same graph+tt_id takes the skip-re-ingest path. The persisted
    manifest must re-trigger the same warn/fail — the first job must not be
    the only one that ever sees the skips."""
    _ingest_with_skips(monkeypatch, membership=60, row_counts=(80, 10, 10))
    monkeypatch.setenv('KUMO_TTNG_ORPHAN_FAIL_PCT', '50')
    with pytest.raises(RuntimeError, match='KUMO_TTNG_ORPHAN_FAIL_PCT'):
        _run_train_ingest(tmp_path)  # ingests, publishes, then hard-fails

    # Second run: extension present-and-complete -> no re-ingest, but the
    # manifest re-check must fail again, not silently succeed.
    with pytest.raises(RuntimeError, match='reused extension'):
        _run_train_ingest(tmp_path)

    # Warn-only mode on the reused extension still warns from the manifest.
    monkeypatch.delenv('KUMO_TTNG_ORPHAN_FAIL_PCT')
    with caplog.at_level('WARNING'):
        ext = _run_train_ingest(tmp_path)
    assert ext.endswith('gen-tt-job')
    reused = [r.getMessage() for r in caplog.records
              if 'reused extension' in r.getMessage()]
    assert reused and 'skipped' in reused[0]
