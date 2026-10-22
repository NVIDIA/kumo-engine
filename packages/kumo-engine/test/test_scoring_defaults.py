# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from kumo_core.artifacts import add_bundle_file, remove_bundle_file
from kumo_core.types import TaskType
from kumoapi.model_plan import ModelPlan

import kumo.engine as kumo
from kumo.engine.trainer._scoring import normalized_scoring_defaults


def _frame(trainer, table, snapshot, **kwargs):
    result = trainer.predict(table, snapshot=snapshot, **kwargs)
    return result.frame.sort_values('ENTITY').reset_index(drop=True)


def _digest(directory):
    return {
        str(path.relative_to(directory)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob('*')) if path.is_file()
    }


def _copy(trainer, destination):
    shutil.copytree(trainer._training_result.bundle_dir, destination)
    return destination


def test_derived_bundle_thresholds_unless_explicit(fitted):
    trainer, table, snapshot = fitted
    parent_dir = trainer._training_result.bundle_dir
    original = _digest(parent_dir)
    probs = _frame(trainer, table, snapshot,
                   binary_classification_threshold=.5)['True_PROB']
    threshold = float(probs.median())
    child = trainer.with_scoring_defaults(
        kumo.ScoringDefaults(binary_classification_threshold=threshold))

    implicit = _frame(child, table, snapshot)
    pd.testing.assert_frame_equal(
        implicit,
        _frame(trainer, table, snapshot,
               binary_classification_threshold=threshold))
    assert implicit['TARGET_PRED'].any()
    assert not _frame(child, table, snapshot,
                      binary_classification_threshold=1.0)['TARGET_PRED'].any()

    assert _digest(parent_dir) == original
    assert trainer.scoring_defaults == kumo.ScoringDefaults()
    assert child.bundle_id != trainer.bundle_id
    assert child.provenance.id == child.bundle_id
    assert trainer.provenance.id == trainer.bundle_id
    assert child.provenance.parents == replace(trainer.provenance.parents,
                                               derived_from=trainer.bundle_id)
    cleared = child.with_scoring_defaults(kumo.ScoringDefaults())
    assert cleared.bundle_id == trainer.bundle_id


def test_equal_defaults_give_equal_bundle_ids(fitted):
    trainer, *_ = fitted
    children = [
        trainer.with_scoring_defaults(
            kumo.ScoringDefaults(binary_classification_threshold=value))
        for value in (1, np.float32(1))
    ]
    assert children[0].bundle_id == children[1].bundle_id
    assert children[0].scoring_defaults.binary_classification_threshold == 1.0


def test_derivation_from_old_or_newer_provenance(fitted, tmp_path):
    trainer, *_ = fitted
    defaults = kumo.ScoringDefaults(binary_classification_threshold=.5)
    old = _copy(trainer, tmp_path / 'old')
    remove_bundle_file(old, 'provenance')
    child = kumo.Trainer.load(old).with_scoring_defaults(defaults)
    assert child.provenance.reconstructed is True
    assert child.provenance.id == child.bundle_id
    recorded = json.loads(
        (child._training_result.bundle_dir / 'provenance.json').read_text())
    assert recorded == {
        'kind': 'model_bundle', 'schema_version': 1, 'id': child.bundle_id,
        'parents': {'derived_from': trainer.bundle_id}, 'created_at': None,
        'environment': None, 'scheduler': None, 'reconstructed': True
    }

    newer = _copy(trainer, tmp_path / 'newer')
    record = json.dumps({'schema_version': 999, 'parents': {}})
    (newer / 'provenance.json').write_text(record)
    add_bundle_file(newer, 'provenance', 'provenance.json')
    with pytest.warns(UserWarning, match='left unchanged'):
        child = kumo.Trainer.load(newer).with_scoring_defaults(defaults)
    assert (child._training_result.bundle_dir /
            'provenance.json').read_text() == record


def test_defaults_survive_export_and_connector_load(fitted, tmp_path):
    trainer, *_ = fitted
    child = trainer.with_scoring_defaults(
        kumo.ScoringDefaults(binary_classification_threshold=.4))
    registry = kumo.FileConnector(root_dir=str(tmp_path))
    child.export(kumo.ModelOutputConfig(registry, path='m1'))
    loaded = kumo.Trainer.load_from_connector(registry, path='m1')
    assert loaded.scoring_defaults == child.scoring_defaults
    assert loaded.bundle_id == child.bundle_id


def test_defaults_are_validated_against_the_task(fitted):
    trainer, *_ = fitted
    with pytest.raises(ValueError, match='num_classes_to_return'):
        trainer.with_scoring_defaults(
            kumo.ScoringDefaults(num_classes_to_return=2))
    with pytest.raises(RuntimeError, match='requires a prior'):
        kumo.Trainer(ModelPlan()).with_scoring_defaults(kumo.ScoringDefaults())
    assert normalized_scoring_defaults(
        kumo.ScoringDefaults(.3, np.int64(2)),
        TaskType.MULTILABEL_CLASSIFICATION) == kumo.ScoringDefaults(.3, 2)


def test_unlisted_or_unreadable_defaults_are_ignored(fitted, tmp_path):
    trainer, table, snapshot = fitted
    bundle = _copy(trainer, tmp_path / 'old')
    (bundle / 'scoring-defaults.json').write_text('not json')
    old = kumo.Trainer.load(bundle)
    assert old.scoring_defaults == kumo.ScoringDefaults()
    assert old.bundle_id == trainer.bundle_id

    add_bundle_file(bundle, 'scoring_defaults', 'scoring-defaults.json')
    with pytest.warns(UserWarning, match='Ignoring the scoring defaults'):
        ignored = kumo.Trainer.load(bundle)
    assert ignored.scoring_defaults == kumo.ScoringDefaults()
    with pytest.raises(ValueError, match='must be specified'):
        ignored.predict(table, snapshot=snapshot)
