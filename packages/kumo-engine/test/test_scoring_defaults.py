# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from kumo_core.artifacts import add_bundle_file, remove_bundle_file
from kumo_core.types import TaskType
from kumoapi.plugin import PublishedModel

import kumo.engine as kumo
from kumo.engine.trainer._scoring import normalized_scoring_defaults


def _frame(model, table, snapshot, **kwargs):
    result = model.predict(table, snapshot=snapshot, **kwargs)
    return result.frame.sort_values('ENTITY').reset_index(drop=True)


def _digest(directory):
    return {
        str(path.relative_to(directory)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob('*')) if path.is_file()
    }


def _copy(trainer, destination):
    shutil.copytree(trainer.result.bundle_dir, destination)
    return destination


def test_derived_model_thresholds_unless_explicit(fitted):
    trainer, table, snapshot = fitted
    parent = trainer.result
    original = _digest(parent.bundle_dir)
    probs = _frame(parent, table, snapshot,
                   binary_classification_threshold=.5)['True_PROB']
    threshold = float(probs.median())
    child = parent.with_scoring_defaults(
        kumo.ScoringDefaults(binary_classification_threshold=threshold))

    implicit = _frame(child, table, snapshot)
    pd.testing.assert_frame_equal(
        implicit,
        _frame(parent, table, snapshot,
               binary_classification_threshold=threshold))
    assert implicit['TARGET_PRED'].any()
    assert not _frame(child, table, snapshot,
                      binary_classification_threshold=1.0)['TARGET_PRED'].any()

    assert _digest(parent.bundle_dir) == original
    assert parent.scoring_defaults == kumo.ScoringDefaults()
    assert child.id != parent.id
    assert child.manifest.id == child.id
    assert child.manifest.parents.training_table is None
    derived_from = child.manifest.parents.derived_from
    assert derived_from.id == parent.id
    assert derived_from.parents.training_table == (
        parent.manifest.parents.training_table)
    cleared = child.with_scoring_defaults(kumo.ScoringDefaults())
    assert cleared.id == parent.id
    assert cleared.manifest.parents.derived_from.parents.derived_from == (
        kumo.ModelManifest(id=parent.id, reconstructed=True))


def test_equal_defaults_give_equal_bundle_ids(fitted):
    trainer, *_ = fitted
    children = [
        trainer.result.with_scoring_defaults(
            kumo.ScoringDefaults(binary_classification_threshold=value))
        for value in (1, np.float32(1))
    ]
    assert children[0].id == children[1].id
    assert children[0].scoring_defaults.binary_classification_threshold == 1.0


def test_parent_without_a_manifest_reads_as_a_stub(fitted, tmp_path):
    trainer, *_ = fitted
    old = _copy(trainer, tmp_path / 'old')
    remove_bundle_file(old, 'provenance')
    child = kumo.Trainer.load(old).result.with_scoring_defaults(
        kumo.ScoringDefaults(binary_classification_threshold=.5))
    assert child.manifest.parents.derived_from == kumo.ModelManifest(
        id=trainer.result.id, reconstructed=True)


def test_defaults_survive_export_and_connector_load(fitted, tmp_path):
    trainer, *_ = fitted
    child = trainer.result.with_scoring_defaults(
        kumo.ScoringDefaults(binary_classification_threshold=.4))
    registry = kumo.FileConnector(root_dir=str(tmp_path))
    child._trainer.export(kumo.ModelOutputConfig(registry, path='m1'))
    loaded = kumo.Trainer.load_from_connector(registry, path='m1')
    assert loaded.result.scoring_defaults == child.scoring_defaults
    assert loaded.result.id == child.id


def test_derived_export_publishes_the_child_bundle_id(fitted):
    trainer, *_ = fitted
    child = trainer.result.with_scoring_defaults(
        kumo.ScoringDefaults(binary_classification_threshold=.4))
    calls = []
    sink = SimpleNamespace(publish_model=lambda *a, **k: calls.append(k) or
                           PublishedModel(name='m', version='1', uri='u'))
    child._trainer.export(
        kumo.ModelOutputConfig(SimpleNamespace(sink_backend=sink), path='m'))
    assert calls[0]['metadata']['kumo_bundle_id'] == child.id


def test_defaults_are_validated_against_the_task(fitted):
    trainer, *_ = fitted
    with pytest.raises(ValueError, match='num_classes_to_return'):
        trainer.result.with_scoring_defaults(
            kumo.ScoringDefaults(num_classes_to_return=2))
    assert normalized_scoring_defaults(
        kumo.ScoringDefaults(.3, np.int64(2)),
        TaskType.MULTILABEL_CLASSIFICATION) == kumo.ScoringDefaults(.3, 2)


def test_unlisted_or_unreadable_defaults_are_ignored(fitted, tmp_path):
    trainer, table, snapshot = fitted
    bundle = _copy(trainer, tmp_path / 'old')
    (bundle / 'scoring-defaults.json').write_text('not json')
    old = kumo.Trainer.load(bundle)
    assert old.result.scoring_defaults == kumo.ScoringDefaults()
    assert old.result.id == trainer.result.id

    add_bundle_file(bundle, 'scoring_defaults', 'scoring-defaults.json')
    with pytest.warns(UserWarning, match='Ignoring the scoring defaults'):
        ignored = kumo.Trainer.load(bundle)
    assert ignored.result.scoring_defaults == kumo.ScoringDefaults()
    with pytest.raises(ValueError, match='must be specified'):
        ignored.predict(table, snapshot=snapshot)
