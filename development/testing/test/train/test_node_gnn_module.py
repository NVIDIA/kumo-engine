# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from kumo_core.train.node_gnn_module import NodeGNNLightningModule


class _Batch:
    def __init__(self) -> None:
        self.tf_dict = {}
        self.batch_dict = {}
        self._values = {
            "entity":
            SimpleNamespace(
                batch_size=2,
                pkey=np.array(["cust_1", "cust_2"], dtype=object),
            ),
            "train_table": {
                "time": torch.tensor([1, 2]),
            },
        }

    def __getitem__(self, key: str):
        return self._values[key]

    def collect(self, *args, **kwargs):
        return {}


def test_predict_step_emits_embeddings_only_when_ttng_flag_is_set() -> None:
    embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    class _Stub:
        entity_table_name = "entity"
        train_table_name = "train_table"

        def __init__(self, return_embeddings: bool) -> None:
            self._kumo_return_predict_embeddings = return_embeddings

        def common_step(self, batch, return_embeddings: bool = False):
            assert return_embeddings is False
            return torch.tensor([[0.1], [0.2]])

        def __call__(self, *args, **kwargs):
            assert kwargs["return_embeddings"] is True
            return embedding

        def head(self, x, seed_time):
            assert x is embedding
            return torch.tensor([[0.1], [0.2]])

    default_out = NodeGNNLightningModule.predict_step(_Stub(False), _Batch(), 0)
    embedding_out = NodeGNNLightningModule.predict_step(_Stub(True), _Batch(),
                                                        0)

    assert "embedding" not in default_out
    assert torch.equal(embedding_out["embedding"], embedding)


def test_common_step_return_embeddings_keeps_head_output_contract() -> None:
    output = torch.tensor([[0.1], [0.2]])

    class _Stub:
        train_table_name = "train_table"

        def __call__(self, *args, **kwargs):
            assert "return_embeddings" not in kwargs
            return output

    out = NodeGNNLightningModule.common_step(_Stub(), _Batch(),
                                             return_embeddings=True)

    assert out is output


def test_output_dtype_supports_distribution_output_dicts() -> None:
    fallback = torch.ones(1, dtype=torch.float32)
    tensor_out = torch.ones(1, dtype=torch.float16)
    dict_out = {
        "loc": torch.ones(1, dtype=torch.float64),
        "scale": torch.ones(1, dtype=torch.float32),
    }

    assert NodeGNNLightningModule._output_dtype(tensor_out,
                                                fallback) == torch.float16
    assert NodeGNNLightningModule._output_dtype(dict_out,
                                                fallback) == torch.float64
    assert NodeGNNLightningModule._output_dtype({}, fallback) == torch.float32
