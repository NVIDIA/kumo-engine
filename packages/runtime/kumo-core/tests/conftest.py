# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest

from tests.support.embedding_models import TinyModel
from tests.support.fixtures import (
    large_user_store_graph,
    user_store_graph,
)
from kumo_core import embedding_models
from kumo_core.embedding_models import EmbeddingModelStore, GloVeVectors
from kumo_core.glove import TokenEmbeddingTable


@pytest.fixture(scope="session", autouse=True)
def configure_glove_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None, None, None]:
    artifacts = tmp_path_factory.mktemp("glove")
    tokens = [
        "get",
        "old",
        "hello",
        "world",
        "a",
        "test",
        "welcome",
        "goodbye",
        "this",
        "is",
        "text",
        "graph",
        "network",
        "greetings",
        "from",
        "europe",
        "wow",
        "cool",
    ]
    pd.DataFrame({"index": tokens}).to_parquet(artifacts / "glove_6B_index.parquet")
    np.save(
        artifacts / "glove_6B_50d_emb.npy",
        np.ones((len(tokens), 50), dtype=np.float32),
    )
    store = EmbeddingModelStore()
    store.add("glove.6B-50d", GloVeVectors(artifacts, cache_dir=artifacts / "cache"))
    store.add("glove.test-10d", TinyModel(10))
    previous = embedding_models.set_default_store(store)
    yield
    embedding_models.set_default_store(previous)
    TokenEmbeddingTable.instances.clear()


@pytest.fixture
def teardown_glove_helper() -> Generator:
    yield
    for table in TokenEmbeddingTable.instances.values():
        table.to('cpu')


__all__ = [
    'user_store_graph',
    'large_user_store_graph',
]
