# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Engine-local ``matmul_precision`` option.

``None`` keeps Engine's ``torch.set_float32_matmul_precision('medium')`` and
every default record byte for byte; a chosen value replaces it for every
Lightning trainer of fit (training, validation, post-fit evaluation) and
predict, is recorded in ``training-spec.json`` and the prediction manifest,
and is never adopted from a loaded bundle.
"""
import contextlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from kumo_core.explain.walk_graph import (
    mkldnn_matmul_fp32_precision,
    normalized_mkldnn,
)
from kumo_core.train.trainer import MATMUL_PRECISIONS
from kumo_core.train.trainer import Trainer as CoreTrainer
from kumo_core.train.trainer import validate_matmul_precision
from kumo_core.training.orchestration import CallableTrainingOptions
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan
from walk_binary_fixtures import (
    BINARY_QUERY,
    binary_graph,
    binary_plan,
    write_binary_tables,
)

import kumo.engine as kumo
from kumo.engine._manifest import RunScoring
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.trainer._spec import (
    SPEC_FILE,
    TrainingSpec,
    read_training_spec,
    write_training_spec,
)

INVALID = ['low', 'HIGHEST', '', ' highest', 1, 1.0, True, b'highest',
           ['highest']]
#: ``RunScoring`` fields of a default prediction manifest at 321ba862.
SCORING_KEYS = {'binary_classification_threshold', 'threshold_source',
                'num_classes_to_return', 'count_source', 'num_workers'}


# API ##########################################################################


@pytest.mark.parametrize('precision', [None, *MATMUL_PRECISIONS])
def test_trainer_accepts_the_torch_precisions(precision):
    assert MATMUL_PRECISIONS == ('highest', 'high', 'medium')
    trainer = kumo.Trainer(ModelPlan(), matmul_precision=precision)
    assert trainer.matmul_precision == precision
    assert validate_matmul_precision(precision) == precision
    assert CallableTrainingOptions(
        matmul_precision=precision).matmul_precision == precision


@pytest.mark.parametrize('precision', INVALID)
def test_other_values_raise_before_any_work(precision, tmp_path):
    with pytest.raises(ValueError, match='matmul_precision'):
        kumo.Trainer(ModelPlan(), matmul_precision=precision)
    with pytest.raises(ValueError, match='matmul_precision'):
        CallableTrainingOptions(matmul_precision=precision)
    # Validation precedes reading or fetching the bundle.
    with pytest.raises(ValueError, match='matmul_precision'):
        kumo.Trainer.load(tmp_path / 'missing', matmul_precision=precision)
    with pytest.raises(ValueError, match='matmul_precision'):
        kumo.Trainer.load_from_connector(cast(Any, None), path='missing',
                                         matmul_precision=precision)


# Records ######################################################################


def _spec_bytes_at_321ba862(**fields) -> bytes:
    """``write_training_spec`` as of 321ba862, which had no precision key."""
    from kumo_core.io.json import to_json_dict

    data = {
        'spec_version': 1,
        'predictive_query': fields['predictive_query'],
        'model_plan': to_json_dict(fields['model_plan']),
        'snapshot_id': fields['snapshot_id'],
        'training_table_generation': to_json_dict(
            fields['training_table_generation']),
        'graph_config': fields.get('graph_config'),
        'timestamp_precision': fields.get('timestamp_precision', 'seconds'),
    }
    if fields.get('bias_options') is not None:
        data['bias_options'] = fields['bias_options']
    return (json.dumps(data, indent=2, sort_keys=True) + '\n').encode()


def _write_spec(bundle: Path, **options: Any) -> bytes:
    bundle.mkdir(parents=True)
    (bundle / 'publication-manifest.json').write_text('{"artifacts": []}')
    fields: dict[str, Any] = dict(
        predictive_query=BINARY_QUERY, model_plan=ModelPlan(),
        snapshot_id='snap',
        training_table_generation=TrainingTableGenerationPlan(), **options)
    write_training_spec(bundle, **fields)
    return _spec_bytes_at_321ba862(**fields)


@pytest.mark.parametrize('bias', [None, {'gnn': False, 'head': True}])
def test_default_spec_keeps_its_bytes(tmp_path, bias):
    expected = _write_spec(tmp_path / 'bundle', bias_options=bias)
    assert (tmp_path / 'bundle' / SPEC_FILE).read_bytes() == expected
    assert read_training_spec(tmp_path / 'bundle').matmul_precision is None


@pytest.mark.parametrize('precision', MATMUL_PRECISIONS)
def test_spec_records_a_chosen_precision(tmp_path, precision):
    _write_spec(tmp_path / 'bundle', matmul_precision=precision)
    data = json.loads((tmp_path / 'bundle' / SPEC_FILE).read_text())
    assert data['matmul_precision'] == precision
    assert read_training_spec(tmp_path / 'bundle').matmul_precision == (
        precision)
    # A reader that predates the key still reads the spec: it takes only
    # its known keys (unknown keys are ignored).
    data['future_option'] = {'any': 1}
    (tmp_path / 'bundle' / SPEC_FILE).write_text(json.dumps(data))
    spec = read_training_spec(tmp_path / 'bundle')
    assert set(asdict(spec)) == {f for f in TrainingSpec.__dataclass_fields__}
    assert spec.matmul_precision == precision


def test_run_scoring_omits_the_default_precision():
    fields: dict[str, Any] = dict(binary_classification_threshold=None,
                                  threshold_source='default',
                                  num_classes_to_return=None,
                                  count_source='query', num_workers=1)
    assert set(RunScoring(**fields).to_dict()) == SCORING_KEYS
    assert set(RunScoring(**fields,
                          matmul_precision=None).to_dict()) == SCORING_KEYS
    chosen = RunScoring(**fields, matmul_precision='highest')
    assert chosen.to_dict()['matmul_precision'] == 'highest'
    assert RunScoring.from_dict(chosen.to_dict()) == chosen
    assert RunScoring.from_dict(RunScoring(**fields).to_dict()) == RunScoring(
        **fields)


# End to end ###################################################################


class _Observer(pl.Callback):
    """Record the float32 matmul settings around every Lightning step.

    The setting is process-global and only ``create_pl_trainer`` writes it, so
    a value seen at a step's start and end is the value inside the step.
    """
    def __init__(self, seen: list) -> None:
        self.seen = seen

    def _record(self, stage: str) -> None:
        self.seen.append((stage, torch.get_float32_matmul_precision(),
                          normalized_mkldnn(mkldnn_matmul_fp32_precision())))

    def on_train_batch_start(self, *args, **kwargs):
        self._record('train')

    def on_train_batch_end(self, *args, **kwargs):
        self._record('train')

    def on_validation_batch_start(self, *args, **kwargs):
        self._record('validation')

    def on_test_batch_start(self, *args, **kwargs):
        self._record('test')

    def on_predict_batch_start(self, *args, **kwargs):
        self._record('predict')

    def on_predict_batch_end(self, *args, **kwargs):
        self._record('predict')


@contextlib.contextmanager
def _observed(monkeypatch):
    """Attach an :class:`_Observer` to every Lightning trainer Engine builds,
    and restore the process precision settings afterwards."""
    seen: list = []
    create = CoreTrainer.create_pl_trainer

    def create_and_observe(self):
        create(self)
        self.pl_trainer.callbacks.append(_Observer(seen))

    with monkeypatch.context() as patch, _restored_settings():
        patch.setattr(CoreTrainer, 'create_pl_trainer', create_and_observe)
        yield seen


@contextlib.contextmanager
def _restored_settings():
    """Restore the process float32 matmul settings on exit."""
    previous = torch.get_float32_matmul_precision()
    previous_mkldnn = mkldnn_matmul_fp32_precision()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
        if previous_mkldnn is not None:
            matmul = getattr(torch.backends.mkldnn, 'matmul')
            matmul.fp32_precision = previous_mkldnn


def _settings(seen, stage):
    return {(precision, mkldnn) for name, precision, mkldnn in seen
            if name == stage}


def _torch_mapping(precision: str) -> tuple[str, Optional[str]]:
    """The settings torch itself derives from ``precision`` on this host
    (``medium`` selects oneDNN bfloat16 matmuls where the CPU supports
    them)."""
    with _restored_settings():
        torch.set_float32_matmul_precision(precision)
        return (torch.get_float32_matmul_precision(),
                normalized_mkldnn(mkldnn_matmul_fp32_precision()))


@pytest.fixture(scope='module')
def binary(tmp_path_factory):
    root = tmp_path_factory.mktemp('matmul-precision')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    try:
        data = root / 'data'
        data.mkdir()
        write_binary_tables(data)
        pd.DataFrame({'ENTITY1': np.arange(6)}).to_parquet(data /
                                                           'score.parquet')
        connector = kumo.FileConnector(root_dir=str(data))
        graph = binary_graph(kumo, connector)
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph=graph, query=BINARY_QUERY)
        yield (snapshot, query.generate_training_table(snapshot),
               connector, root)
    finally:
        kumo.configure(scratch_dir=previous)


@pytest.fixture(scope='module')
def fits(binary):
    """One default fit and one ``'highest'`` fit, each with its settings."""
    snapshot, table, _, _ = binary
    out = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        for precision in (None, 'highest'):
            with _observed(monkeypatch) as seen:
                trainer = kumo.Trainer(binary_plan(epochs=1, steps=2),
                                       accelerator='cpu',
                                       matmul_precision=precision)
                trainer.fit(table, snapshot=snapshot)
            out[precision] = (trainer, list(seen))
    return out


def _predict(trainer, binary, monkeypatch, **options):
    snapshot, _, connector, _ = binary
    with _observed(monkeypatch) as seen:
        result = trainer.predict(snapshot=snapshot,
                                 entities=connector['score.parquet'],
                                 **options)
    manifest = json.loads(result._manifest_path.read_text())
    return result, list(seen), manifest


def _bundle(trainer) -> Path:
    assert trainer.result is not None
    return Path(trainer.result.bundle_dir)


def test_default_fit_keeps_medium_and_its_records(fits):
    trainer, seen = fits[None]
    for stage in ('train', 'validation', 'test'):
        assert _settings(seen, stage) == {_torch_mapping('medium')}, stage
    bundle = _bundle(trainer)
    spec = json.loads((bundle / SPEC_FILE).read_text())
    assert 'matmul_precision' not in spec
    assert 'matmul_precision' not in (bundle / 'model' /
                                      'config.json').read_text()


def test_highest_fit_runs_every_stage_in_full_float32(fits):
    assert _torch_mapping('highest') == ('highest', 'ieee')
    trainer, seen = fits['highest']
    for stage in ('train', 'validation', 'test'):
        assert _settings(seen, stage) == {('highest', 'ieee')}, stage
    bundle = _bundle(trainer)
    spec = json.loads((bundle / SPEC_FILE).read_text())
    assert spec['matmul_precision'] == 'highest'
    # The option is not part of the trainer config: both bundles' configs
    # agree apart from paths and versions.
    assert 'matmul_precision' not in (bundle / 'model' /
                                      'config.json').read_text()
    default = json.loads((_bundle(fits[None][0]) / SPEC_FILE).read_text())
    assert set(spec) - set(default) == {'matmul_precision'}


def test_predict_follows_the_loaded_option_not_the_bundle(
        fits, binary, monkeypatch):
    bundle = _bundle(fits['highest'][0])
    plain = kumo.Trainer.load(bundle, accelerator='cpu')
    assert plain.matmul_precision is None
    result, seen, manifest = _predict(plain, binary, monkeypatch,
                                      batch_size=2)
    assert _settings(seen, 'predict') == {_torch_mapping('medium')}
    assert set(manifest['scoring']) == SCORING_KEYS
    assert 'matmul_precision' not in json.dumps(manifest)

    chosen = kumo.Trainer.load(bundle, accelerator='cpu',
                               matmul_precision='highest')
    assert chosen.matmul_precision == 'highest'
    chosen_result, seen, manifest = _predict(chosen, binary, monkeypatch,
                                             batch_size=2)
    assert _settings(seen, 'predict') == {('highest', 'ieee')}
    assert manifest['scoring']['matmul_precision'] == 'highest'
    assert set(manifest['scoring']) == SCORING_KEYS | {'matmul_precision'}
    # Same parameters, other numerics: identical rows and columns.
    left = result.frame.sort_values('ENTITY', ignore_index=True)
    right = chosen_result.frame.sort_values('ENTITY', ignore_index=True)
    assert list(left.columns) == list(right.columns)
    assert left.ENTITY.tolist() == right.ENTITY.tolist()


def test_fitted_trainer_predicts_under_its_own_option(fits, binary,
                                                      monkeypatch):
    trainer = fits['highest'][0]
    _, seen, manifest = _predict(trainer, binary, monkeypatch)
    assert _settings(seen, 'predict') == {('highest', 'ieee')}
    assert manifest['scoring']['matmul_precision'] == 'highest'
    default = fits[None][0]
    _, seen, manifest = _predict(default, binary, monkeypatch)
    assert _settings(seen, 'predict') == {_torch_mapping('medium')}
    assert set(manifest['scoring']) == SCORING_KEYS
