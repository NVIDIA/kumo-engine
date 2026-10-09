# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pathlib
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch
from kumoapi.encoder import Encoder
from kumoapi.explain import GraphGradientScore
from kumoapi.task import TaskType
from kumoapi.train import TrainingStage
from torch import Tensor
from torch_geometric.testing import withCUDA

import kumo_core.modeling.context as modeling_context
from kumo_core.io.json import load_json
from kumo_core.features.stores.in_memory_feature import InMemoryFeatureStore
from tests.support.in_memory_graph import InMemoryGraphStore
from tests.support.fixtures import user_store_batch
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.explain.gradient_explainer import GradientExplainer


@withCUDA
def test_gradient(
    tmp_path: pathlib.Path,
    user_store_graph: tuple[
        InMemoryFeatureStore,
        InMemoryGraphStore,
        dict[str, dict[str, Encoder]],
    ],
    device: torch.device,
) -> None:

    feature_store, graph_store, table_encoder_dict = user_store_graph
    loader = SimpleNamespace(data=(feature_store, graph_store))
    batch = user_store_batch(feature_store).to(device)

    class FakeModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = HeteroTensorFrameEncoder(
                table_encoder_dict,
                time_col_dict=feature_store.time_col_dict,
                out_channels=32,
                exclude_cols_dict={},
            )

        def forward(self, *args: Any, **kwargs: Any) -> Tensor:
            out_dict = self.encoder(*args, **kwargs)
            return torch.cat(list(out_dict.values()), dim=0)

    module = FakeModule().to(device).eval()

    explainer = GradientExplainer(
        task_type=TaskType.BINARY_CLASSIFICATION,
        root_dir=str(tmp_path),
        formats=['json', 'csv'],
    )
    assert str(explainer) == 'GradientExplainer()'

    with torch.no_grad():
        explainer.explain_start(
            module=module,
            datamodule=loader,
            stage=TrainingStage.VAL,
        )
        explainer.explain_pre_step(
            batch=batch,
            module=module,
            datamodule=loader,
            stage=TrainingStage.VAL,
            batch_idx=0,
        )
        with (modeling_context.explaining(), torch.enable_grad()):
            step_output = {}
            step_output['logits'] = module(
                batch.tf_dict,
                batch.batch_dict,
                batch.time_dict,
                batch['USERS'].seed_time,
            )
        explainer.explain_post_step(
            batch=batch,
            module=module,
            datamodule=loader,
            step_output=step_output,
            stage=TrainingStage.VAL,
            batch_idx=0,
        )
        explainer.explain_end(
            module=module,
            datamodule=loader,
            outputs=step_output,
            stage=TrainingStage.VAL,
        )

    obj = load_json(str(tmp_path / 'scores.json'), GraphGradientScore)
    assert len(obj.tables) == len(batch.node_types)
    df = pd.read_csv(str(tmp_path / 'scores.csv'))
    assert df.columns.tolist() == ['table', 'column', 'score']
    for table_name, tf in batch.tf_dict.items():
        col_names = [
            col_name for col_names in tf.col_names_dict.values()
            for col_name in col_names
        ]
        assert len(obj.tables[table_name].columns) == len(col_names)
        for col_name in col_names:
            assert obj.tables[table_name].columns[col_name] >= 0
            mask = (df['table'] == table_name) & (df['column'] == col_name)
            assert mask.sum() == 1
            assert float(df[mask]['score'].iloc[0]) >= 0


@withCUDA
def test_gradient_forecasting(
    tmp_path: pathlib.Path,
    user_store_graph: tuple[
        InMemoryFeatureStore,
        InMemoryGraphStore,
        dict[str, dict[str, Encoder]],
    ],
    device: torch.device,
) -> None:
    feature_store, graph_store, table_encoder_dict = user_store_graph
    loader = SimpleNamespace(data=(feature_store, graph_store))
    batch = user_store_batch(feature_store).to(device)

    forecast_length = 3

    class FakeModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = HeteroTensorFrameEncoder(
                table_encoder_dict,
                time_col_dict=feature_store.time_col_dict,
                out_channels=32,
                exclude_cols_dict={},
            )
            self.head = torch.nn.Linear(32, forecast_length)

        def forward(self, *args: Any, **kwargs: Any) -> Tensor:
            out_dict = self.encoder(*args, **kwargs)
            x = torch.cat(list(out_dict.values()), dim=0)
            # Simulate forecasting head: [num_nodes, forecast_length, 1]
            return self.head(x).unsqueeze(-1)

    module = FakeModule().to(device).eval()

    explainer = GradientExplainer(
        task_type=TaskType.FORECASTING,
        root_dir=str(tmp_path),
        formats=['json', 'csv'],
    )

    with torch.no_grad():
        explainer.explain_start(
            module=module,
            datamodule=loader,
            stage=TrainingStage.VAL,
        )
        explainer.explain_pre_step(
            batch=batch,
            module=module,
            datamodule=loader,
            stage=TrainingStage.VAL,
            batch_idx=0,
        )
        with (modeling_context.explaining(), torch.enable_grad()):
            step_output: dict[str, Tensor] = {}
            step_output['logits'] = module(
                batch.tf_dict,
                batch.batch_dict,
                batch.time_dict,
                batch['USERS'].seed_time,
            )
        explainer.explain_post_step(
            batch=batch,
            module=module,
            datamodule=loader,
            step_output=step_output,
            stage=TrainingStage.VAL,
            batch_idx=0,
        )
        explainer.explain_end(
            module=module,
            datamodule=loader,
            outputs=step_output,
            stage=TrainingStage.VAL,
        )

    obj = load_json(str(tmp_path / 'scores.json'), GraphGradientScore)
    assert len(obj.tables) == len(batch.node_types)
    df = pd.read_csv(str(tmp_path / 'scores.csv'))
    assert df.columns.tolist() == ['table', 'column', 'score']
    for table_name, tf in batch.tf_dict.items():
        col_names = [
            col_name for col_names in tf.col_names_dict.values()
            for col_name in col_names
        ]
        assert len(obj.tables[table_name].columns) == len(col_names)
        for col_name in col_names:
            assert obj.tables[table_name].columns[col_name] >= 0
            mask = (df['table'] == table_name) & (df['column'] == col_name)
            assert mask.sum() == 1
            assert float(df[mask]['score'].iloc[0]) >= 0
