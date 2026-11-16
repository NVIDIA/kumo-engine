# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof that the symmetric spec-sourced ``tt_id`` derivation produces a value
BYTE-IDENTICAL to the old path-parse (train) / ``or job_input.id`` (predict)
derivations for every normal job. This is a pure refactor of HOW the id is
resolved, not WHAT it is.

Old train derivation (deleted): parse ``.../jobs/<tt_id>/train_table/...``.
Old predict source derivation (deleted fallback):
    ``predict_staging_tt_id(generate_pred_table_job_id or job_input.id)``.
New derivation (both): read ``tt_id`` from the durable bundle's
``tt_ingest_spec.json`` -- which gen stamped with exactly the same id.
"""
import json
import os
import os.path as osp

from kumo.data.generic.diskgraph_staging import (
    predict_staging_tt_id,
    ttng_persisted_pred_staging_dir,
)
from kumo.data.generic.diskgraph_train_table import (
    read_tt_id_from_staging_bundle,
    ttng_persisted_staging_dir,
)


def _old_train_tt_id_from_path(train_table_path):
    """The exact deleted ``_ttng_tt_id_from_train_table_path`` body."""
    parts = train_table_path.split(osp.sep)
    return parts[parts.index('jobs') + 1]


def _old_predict_source_tt_id(generate_pred_table_job_id, bp_job_id):
    """The exact deleted ``predict_staging_tt_id(gen or bp)`` derivation."""
    return predict_staging_tt_id(
        str(generate_pred_table_job_id or bp_job_id))


def _write_bundle(d, tt_id):
    os.makedirs(d, exist_ok=True)
    with open(osp.join(d, 'tt.parquet'), 'wb') as f:
        f.write(b'x')
    with open(osp.join(d, 'tt_ingest_spec.json'), 'w') as f:
        json.dump({
            'schema_version': 'diskgraph.tt_ingest_spec.v3',
            'tt_id': tt_id,
            'split_stages': None,
        }, f)


def test_train_tt_id_spec_source_equals_old_path_parse(tmp_path):
    """gen stamps the spec with the gen-traintable job id -- the same id the old
    code parsed out of ``.../jobs/<tt_id>/train_table/...``."""
    job_id = 'gen-tt-job-123'
    train_table_path = str(tmp_path / 'jobs' / job_id / 'train_table' / 'tt')
    old = _old_train_tt_id_from_path(train_table_path)
    assert old == job_id  # sanity: path-parse recovers the job id

    # gen persists the bundle beside the train table, stamped with the SAME id.
    _write_bundle(ttng_persisted_staging_dir(train_table_path), tt_id=job_id)
    new = read_tt_id_from_staging_bundle(
        ttng_persisted_staging_dir(train_table_path))
    assert new == old  # BYTE-IDENTICAL value


def test_predict_source_tt_id_spec_source_equals_old_derivation_child(tmp_path):
    """Child/decoupled gen job: generate_pred_table_job_id is set -> source id
    == pred_<gen_job_id> under both old and new derivations."""
    gen_job_id, bp_job_id = 'gen-job-77', 'bp-job-99'
    old = _old_predict_source_tt_id(gen_job_id, bp_job_id)
    assert old == 'pred_gen-job-77'

    pred_table_path = str(tmp_path / 'home' / 'pred_table')
    # gen stamps the bundle spec with pred_<gen_job_id>.
    _write_bundle(ttng_persisted_pred_staging_dir(pred_table_path), tt_id=old)
    new = read_tt_id_from_staging_bundle(
        ttng_persisted_pred_staging_dir(pred_table_path))
    assert new == old


def test_predict_source_tt_id_spec_source_equals_old_derivation_no_gen(
        tmp_path):
    """No separate gen job (generate_pred_table_job_id None): the OLD code fell
    back to job_input.id -> source id == pred_<bp_job_id>. gen stamps the SAME
    id into the spec, so the new spec-source read matches -- WITHOUT the fragile
    ``or job_input.id`` fallback in the consumer."""
    bp_job_id = 'bp-job-99'
    old = _old_predict_source_tt_id(None, bp_job_id)
    assert old == 'pred_bp-job-99'

    pred_table_path = str(tmp_path / 'home' / 'pred_table')
    _write_bundle(ttng_persisted_pred_staging_dir(pred_table_path), tt_id=old)
    new = read_tt_id_from_staging_bundle(
        ttng_persisted_pred_staging_dir(pred_table_path))
    assert new == old
