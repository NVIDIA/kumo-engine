# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import numpy as np
import pandas as pd
import pytest
from kumoapi.encoder import TextEmbedding

from kumo_core.embedding_models import EmbeddingModelStore
from kumo_core.encoders.config import EncodersConfig
from kumo_core.encoders.fused.text import FusedTextEncoder
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.glove import TokenEmbeddingTable, VocabularyIndex
from kumo_core.train.trainer import Trainer


class _Words:
    __slots__: tuple[str, ...] = ()
    family = "lifetime-words/1"
    label = None
    dim = 4

    def __init__(self) -> None:
        self.words = pd.Index(["red", "shoe", "sock"])

    def fingerprint(self) -> str:
        return "lifetime-words-fingerprint"

    def tokenize(self, texts):
        return VocabularyIndex(self.words).tokenize(texts)

    def vectors(self) -> np.ndarray:
        return np.ones((len(self.words), self.dim), dtype=np.float32)

    def vocabulary(self) -> pd.Index:
        return self.words


class WordsModel(_Words):
    pass


class SlotsModel(_Words):
    __slots__ = ("words",)


def _use(model: WordsModel):
    store = EmbeddingModelStore()
    store.add("mine", model)
    trainer = Trainer.__new__(Trainer)
    trainer._embedding_models = store
    trainer.encoders = EncodersConfig({"users": {"bio": TextEmbedding("mine")}})
    encoder = trainer.encoders.encoders["users"]["bio"]
    mapper = TensorMapper.from_encoder(encoder)
    mapper(pd.Series(["red shoe"]))
    module = FusedTextEncoder({"bio": encoder}, [], 8, "fuse")
    return store, trainer, encoder, mapper, module


def test_caches_do_not_keep_a_dropped_model_alive() -> None:
    model = WordsModel()
    store, trainer, encoder, mapper, module = _use(model)
    assert mapper.model is model
    table = module.tables[0]
    model_ref, table_ref = weakref.ref(model), weakref.ref(table)

    del model, store, trainer, encoder, mapper, module, table
    gc.collect()

    assert model_ref() is None
    assert table_ref() is None


def test_live_users_share_one_table() -> None:
    first, second = WordsModel(), WordsModel()
    *_, module = _use(first)
    *_, other = _use(second)

    assert other.tables[0] is module.tables[0]
    assert TokenEmbeddingTable.instance(second) is module.tables[0]


def test_models_must_be_weakly_referenceable() -> None:
    store = EmbeddingModelStore()
    with pytest.raises(TypeError, match="add \"__weakref__\" to the __slots__"):
        store.add("mine", SlotsModel())
