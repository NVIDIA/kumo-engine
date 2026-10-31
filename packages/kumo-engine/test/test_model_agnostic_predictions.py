# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prediction runs and monitoring work for any model's output files, with a
model manifest that records no fit."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from kumo_core.score_counts import Counts
from kumo_core.score_drift import score_drift as exact_drift
from kumo_core.summaries import Scalar
from kumo_core.types import TaskType

from kumo.engine._manifest import (
    ModelManifest,
    PredictionTableRecord,
    RunScoring,
    stub,
)
from kumo.engine._monitoring import score_drift
from kumo.engine._tracking import prediction_record
from kumo.engine.trainer._run_manifest import _record_predictions

MODEL = ModelManifest(id='toy-model')


def _run(tmp_path, name, seed):
    scores = np.random.default_rng(seed).uniform(size=200)
    path = tmp_path / f'{name}.parquet'
    pd.DataFrame({'ENTITY': range(200), 'True_PROB': scores,
                  'TARGET_PRED': scores > .5}).to_parquet(path)
    manifest = _record_predictions(
        name, {'predictions': (200, lambda: [str(path)])}, model=MODEL,
        snapshot=stub('snapshot', 'g@d'),
        task_type=TaskType.BINARY_CLASSIFICATION,
        table=PredictionTableRecord(query_sha256='q', source='entities',
                                    anchors=[], rows=200,
                                    requested_rows=205, scored_rows=200,
                                    skipped_rows=5),
        scoring=RunScoring(binary_classification_threshold=.5,
                           threshold_source='argument',
                           num_classes_to_return=None, count_source='none',
                           num_workers=1),
        started_at='2026-10-28T00:00:00+00:00')
    return SimpleNamespace(id=name, manifest=manifest, bundle_id=MODEL.id)


def test_any_models_prediction_run_is_recorded_and_monitored(tmp_path):
    run = _run(tmp_path, 'run-2', 1)
    stats = run.manifest.stats['predictions']
    assert stats.rows == 200
    assert Counts.from_dict(stats.counts).arrays['True_PROB'].sum() == 200
    assert run.manifest.parents.model == MODEL

    record = prediction_record(run.manifest)
    assert Scalar('pred/skipped_share', 5 / 205) in record.summaries
    assert record.tags['kumo.bundle_id'] == 'toy-model'

    previous = _run(tmp_path, 'run-1', 2)
    report = score_drift(SimpleNamespace(result=SimpleNamespace(
        manifest=MODEL, id=MODEL.id)), run, previous=previous, snapshot=None,
                         training_snapshot=None)
    assert report.reasons['holdout'] == (
        "the model's holdout records no score counts")
    assert report.psi is None and report.bundle_matches is True
    assert (report.previous_psi, report.previous_ks,
            report.previous_wasserstein) == tuple(exact_drift(*(
                Counts.from_dict(r.manifest.stats['predictions'].counts)
                for r in (run, previous))).values())
