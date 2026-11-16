# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import weakref

import numpy as np
import pandas as pd
import pytest
from kumo_core import embedding_models
from kumo_core.automl.auto_trainer import AutoTrainer
from kumo_core.embedding_models import (
    EmbeddingModelNotFoundError,
    resolve_encoder,
)
from kumo_core.encoders.config import EncodersConfig
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.model_plan import ModelPlan, ModelPlanValueError
from kumo_core.train.trainer import Trainer, load_encoders
from kumoapi.encoder import Index, TextEmbedding

from kumo.automl.runtime import MonolithAutoTrainingRuntime
from kumo.utils.embedding_models import (
    embedding_model_store,
    validate_text_embedding_models,
)


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    monkeypatch.setattr(embedding_models, 'entry_points', lambda group: [])
    monkeypatch.setattr(embedding_models, '_INSTALLED', None)
    embedding_model_store.cache_clear()
    yield
    embedding_model_store.cache_clear()


def test_the_baked_directory_is_registered_under_default_names(
        tmp_path, monkeypatch):
    pd.DataFrame({
        'index': ['hello', 'world']
    }).to_parquet(tmp_path / 'glove_6B_index.parquet')
    for dim in (50, 100):
        np.save(tmp_path / f'glove_6B_{dim}d_emb.npy',
                np.ones((2, dim), dtype=np.float32))
    monkeypatch.setenv('KUMO_GLOVE_PATH', str(tmp_path))

    store = embedding_model_store()

    assert store.names == ['glove.6B-100d', 'glove.6B-50d']
    assert store.get('glove.6B-100d').dim == 100
    assert embedding_model_store() is store

    with pytest.raises(EmbeddingModelNotFoundError) as error:
        resolve_encoder(TextEmbedding('glove.42B-300d'), store, ['a.x'],
                        'override')
    assert str(error.value) == (
        "Column 'a.x' uses the embedding model 'glove.42B-300d', which this "
        "deployment's image does not include (it has 'glove.6B-100d', "
        "'glove.6B-50d'). Use one of those, or bake the model into the image "
        "(BAKE_GLOVE_LEVEL=all).")


@pytest.mark.parametrize('path', [None, 'missing', 'empty'])
def test_no_baked_directory_leaves_the_store_empty(path, tmp_path,
                                                   monkeypatch):
    monkeypatch.delenv('KUMO_GLOVE_PATH', raising=False)
    if path is not None:
        (tmp_path / 'empty').mkdir()
        monkeypatch.setenv('KUMO_GLOVE_PATH', str(tmp_path / path))
    assert embedding_model_store().names == []


def test_plans_may_only_name_official_glove_models():
    plan = ModelPlan()
    plan.column_processing.encoder_overrides = {
        'a.x': TextEmbedding(),
        'a.y': 'TextEmbedding(model="glove.840B-300d")',
        'a.z': Index(),
    }
    validate_text_embedding_models(plan)

    plan.column_processing.encoder_overrides['b.x'] = TextEmbedding('desc')
    with pytest.raises(ModelPlanValueError) as error:
        validate_text_embedding_models(plan)
    assert "'b.x' uses 'desc'" in str(error.value)
    assert 'glove.42B-300d, glove.6B-100d' in str(error.value)


def test_hosted_training_resolves_from_the_baked_store(tmp_path, monkeypatch):
    baked = tmp_path / 'baked'
    baked.mkdir()
    pd.DataFrame({
        'index': ['hello', 'world']
    }).to_parquet(baked / 'glove_6B_index.parquet')
    np.save(baked / 'glove_6B_50d_emb.npy', np.ones((2, 50), np.float32))
    monkeypatch.setenv('KUMO_GLOVE_PATH', str(baked))
    # Not the tests' default store: only what the monolith builds.
    monkeypatch.setattr(embedding_models, '_default_store', None)
    monkeypatch.setattr(embedding_models, '_RESOLVED',
                        weakref.WeakValueDictionary())
    with pytest.raises(EmbeddingModelNotFoundError):
        resolve_encoder(TextEmbedding())

    log_dir = str(tmp_path / 'automl')
    monkeypatch.setattr(AutoTrainer, 'log_dir', log_dir)
    monkeypatch.setattr(
        AutoTrainer, '_resolve_encoders', lambda self: EncodersConfig(
            {'users': {
                'bio': TextEmbedding()
            }}))
    auto_trainer = AutoTrainer.__new__(AutoTrainer)
    auto_trainer.runtime = MonolithAutoTrainingRuntime()
    encoders = auto_trainer.encoders
    trainer = Trainer.__new__(Trainer)
    trainer._embedding_models = (
        auto_trainer.runtime.trainer_runtime_kwargs()['embedding_models'])
    trainer.encoders = encoders
    mapper = TensorMapper.from_encoder(encoders.encoders['users']['bio'])
    assert mapper.model.path == baked

    # Prediction, in another process, loads the saved encoders.
    monkeypatch.setattr(embedding_models, '_RESOLVED',
                        weakref.WeakValueDictionary())
    predictor = Trainer.__new__(Trainer)
    predictor._embedding_models = embedding_model_store()
    predictor.encoders = load_encoders(log_dir)
    encoder = predictor.encoders.encoders['users']['bio']
    assert encoder._identity['label'] == 'glove.6B 50d'
    assert TensorMapper.from_encoder(encoder).model.path == baked
