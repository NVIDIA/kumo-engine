# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from kumo_core.artifacts import add_bundle_file, remove_bundle_file
from kumo_core.types import TaskType
from kumoapi.model_plan import ModelPlan

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.trainer._scoring import validate_scoring_defaults
from kumo.engine.trainer._spec import MANIFEST_FILE


def _frame(trainer, table, snapshot, **kwargs):
    result = trainer.predict(table, snapshot=snapshot, **kwargs)
    return result.frame.sort_values('ENTITY').reset_index(drop=True)


def _digest(directory):
    return {
        str(path.relative_to(directory)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob('*')) if path.is_file()
    }


@pytest.fixture(scope='module')
def derived(fitted):
    trainer, table, snapshot = fitted
    original = _digest(trainer._training_result.bundle_dir)
    probs = _frame(trainer, table, snapshot,
                   binary_classification_threshold=.5)['True_PROB']
    defaults = kumo.ScoringDefaults(
        binary_classification_threshold=float(probs.median()))
    return trainer.with_scoring_defaults(defaults), original


def test_binary_predict_uses_bundle_threshold_unless_explicit(
        fitted, derived):
    trainer, table, snapshot = fitted
    child, _ = derived
    threshold = child.scoring_defaults.binary_classification_threshold
    assert trainer.scoring_defaults == kumo.ScoringDefaults()

    implicit = _frame(child, table, snapshot)
    pd.testing.assert_frame_equal(
        implicit,
        _frame(trainer, table, snapshot,
               binary_classification_threshold=threshold))
    explicit = _frame(child, table, snapshot,
                      binary_classification_threshold=1.0)
    assert not explicit['TARGET_PRED'].any()
    assert implicit['TARGET_PRED'].any()


def test_derived_bundle_is_a_new_child_of_an_untouched_parent(
        fitted, derived):
    trainer, *_ = fitted
    child, original = derived
    parent_dir = trainer._training_result.bundle_dir
    assert _digest(parent_dir) == original
    assert kumo.Trainer.load(parent_dir).bundle_id == trainer.bundle_id
    assert child.bundle_id != trainer.bundle_id
    assert child._training_result.bundle_dir.name == 'derived-model'

    provenance = child._training_result.provenance
    assert provenance['parents']['derived_from'] == trainer.bundle_id
    assert (provenance['parents']['snapshot'] ==
            trainer._training_result.provenance['parents']['snapshot'])
    manifest = json.loads(
        (child._training_result.bundle_dir / MANIFEST_FILE).read_text())
    assert {
        entry['role']: entry['relative_path']
        for entry in manifest['artifacts']
    }['scoring_defaults'] == 'scoring-defaults.json'
    assert 'scoring_defaults' not in child._training_result.artifact_paths


def test_empty_defaults_restore_the_no_defaults_bundle_id(fitted, derived):
    trainer, *_ = fitted
    child, _ = derived
    for source in (trainer, child):
        cleared = source.with_scoring_defaults(kumo.ScoringDefaults())
        assert cleared.bundle_id == trainer.bundle_id
        assert cleared.scoring_defaults == kumo.ScoringDefaults()
        assert not (cleared._training_result.bundle_dir /
                    'scoring-defaults.json').exists()


def test_derivation_without_parent_provenance_is_marked_reconstructed(
        fitted, tmp_path):
    trainer, *_ = fitted
    bundle = tmp_path / 'old'
    shutil.copytree(trainer._training_result.bundle_dir, bundle)
    remove_bundle_file(bundle, 'provenance')
    child = kumo.Trainer.load(bundle).with_scoring_defaults(
        kumo.ScoringDefaults(binary_classification_threshold=.5))
    provenance = child._training_result.provenance
    assert provenance['reconstructed'] is True
    assert provenance['parents'] == {'derived_from': trainer.bundle_id}
    assert provenance['created_at'] is None
    assert provenance['environment'] is None


def test_defaults_survive_export_and_connector_load(derived, tmp_path):
    child, _ = derived
    registry = kumo.FileConnector(root_dir=str(tmp_path))
    published = child.export(kumo.ModelOutputConfig(registry, path='m1'))
    assert published is not None
    loaded = kumo.Trainer.load_from_connector(registry, path='m1')
    assert loaded.scoring_defaults == child.scoring_defaults
    assert loaded.bundle_id == child.bundle_id


@pytest.mark.parametrize('defaults, task_type, match', [
    (kumo.ScoringDefaults(binary_classification_threshold=1.5),
     TaskType.BINARY_CLASSIFICATION, 'between 0 and 1'),
    (kumo.ScoringDefaults(num_classes_to_return=3),
     TaskType.BINARY_CLASSIFICATION, 'num_classes_to_return'),
    (kumo.ScoringDefaults(binary_classification_threshold=.5),
     TaskType.MULTILABEL_CLASSIFICATION, 'only to binary'),
    (kumo.ScoringDefaults(binary_classification_threshold=.5),
     TaskType.REGRESSION, 'only to binary'),
    (kumo.ScoringDefaults(num_classes_to_return=501),
     TaskType.STATIC_LINK_PREDICTION, '500'),
])
def test_invalid_defaults_raise(defaults, task_type, match):
    with pytest.raises(ValueError, match=match):
        validate_scoring_defaults(defaults, task_type)


def test_with_scoring_defaults_validates_against_the_bundle(fitted):
    trainer, *_ = fitted
    with pytest.raises(ValueError, match='num_classes_to_return'):
        trainer.with_scoring_defaults(
            kumo.ScoringDefaults(num_classes_to_return=2))
    with pytest.raises(RuntimeError, match='requires a prior'):
        kumo.Trainer(ModelPlan()).with_scoring_defaults(
            kumo.ScoringDefaults())
    with pytest.raises(RuntimeError, match='requires a prior'):
        kumo.Trainer(ModelPlan()).export(
            kumo.ModelOutputConfig(kumo.FileConnector(root_dir='.'),
                                   path='m'))


def test_bundle_without_listed_defaults_keeps_todays_behaviour(
        fitted, tmp_path):
    trainer, table, snapshot = fitted
    bundle = tmp_path / 'old'
    shutil.copytree(trainer._training_result.bundle_dir, bundle)
    (bundle / 'scoring-defaults.json').write_text(
        json.dumps({'binary_classification_threshold': .5}))
    old = kumo.Trainer.load(bundle)
    assert old.scoring_defaults == kumo.ScoringDefaults()
    assert old.bundle_id == trainer.bundle_id
    with pytest.raises(ValueError, match='must be specified'):
        old.predict(table, snapshot=snapshot)

    for content in ('not json', json.dumps({'num_classes_to_return': 2})):
        (bundle / 'scoring-defaults.json').write_text(content)
        add_bundle_file(bundle, 'scoring_defaults', 'scoring-defaults.json')
        with pytest.warns(UserWarning, match='Ignoring the scoring defaults'):
            ignored = kumo.Trainer.load(bundle)
        assert ignored.scoring_defaults == kumo.ScoringDefaults()
    with pytest.raises(ValueError, match='must be specified'):
        ignored.predict(table, snapshot=snapshot)


def test_multilabel_count_default_yields_to_explicit_and_top_k(tmp_path):
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        data = tmp_path / 'data'
        data.mkdir()
        pd.DataFrame({
            'ID': range(120),
            'VALUE': np.arange(120) / 120,
            'TAGS': [['a', 'b'], ['b', 'c'], ['a', 'c']] * 33 +
            [['a', 'b']] + [None] * 20,
        }).to_parquet(data / 'users.parquet')
        connector = kumo.FileConnector(root_dir=str(data))
        graph = kumo.Graph({
            'users': kumo.Table.from_source_table(connector['users'],
                                                  primary_key='ID'),
        })
        snapshot = graph.snapshot()
        plan = ModelPlan()
        plan.training_job.disable_explain = True
        plan.optimization.max_epochs = 1
        plan.optimization.max_steps_per_epoch = 2
        plan.optimization.batch_size = [32]
        plan.model_architecture.channels = [16]
        trainer = kumo.Trainer(plan)
        trainer.fit(
            kumo.PredictiveQuery(
                graph, 'PREDICT users.TAGS CLASSIFY TOP 3 FOR EACH users.ID').
            generate_training_table(snapshot), snapshot=snapshot)
        child = trainer.with_scoring_defaults(
            kumo.ScoringDefaults(num_classes_to_return=2))

        def rows(model, query, **kwargs):
            table = kumo.PredictiveQuery(
                graph, query).generate_prediction_table(snapshot)
            return model.predict(table, snapshot=snapshot,
                                 **kwargs).frame.groupby('ENTITY').size()

        without_top = 'PREDICT users.TAGS CLASSIFY FOR EACH users.ID'
        with_top = 'PREDICT users.TAGS CLASSIFY TOP 3 FOR EACH users.ID'
        assert set(rows(child, without_top)) == {2}
        assert set(rows(child, without_top, num_classes_to_return=1)) == {1}
        assert set(rows(child, with_top)) == {3}
        assert set(rows(trainer, without_top)) == set(
            rows(child, without_top, num_classes_to_return=3))
    finally:
        kumo.configure(scratch_dir=previous)
