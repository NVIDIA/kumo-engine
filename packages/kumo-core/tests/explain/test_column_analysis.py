# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pathlib
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Literal

import pytest
import torch
from kumoapi.encoder import Encoder
from kumoapi.explain import ColumnAnalysisOutput
from kumoapi.task import TaskType
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype
from torch import Tensor
from torch_geometric.testing import withCUDA

from kumo_core.io.json import load_json
from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper
from kumo_core.features.stores.in_memory_feature import InMemoryFeatureStore
from tests.support.in_memory_graph import InMemoryGraphStore
from tests.support.fixtures import user_store_batch
from kumo_core.explain.column_analysis import ColumnAnalysis
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.encoders.target.categorical import CategoricalTargetEncoder


@withCUDA
def test_column_analysis(
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
            self.target_encoder = CategoricalTargetEncoder(
                task_type=TaskType.BINARY_CLASSIFICATION,
                tensor_mapper=CategoricalTensorMapper([[False, 2], [True, 2]]),
            )

        def forward(self, *args: Any, **kwargs: Any) -> Tensor:
            out_dict = self.encoder(*args, **kwargs)
            return torch.cat(list(out_dict.values()), dim=0)

    module = FakeModule().to(device).eval()

    explainer = ColumnAnalysis(
        task_type=TaskType.BINARY_CLASSIFICATION,
        root_dir=str(tmp_path),
    )
    assert str(explainer) == 'ColumnAnalysis()'

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
    step_output = {
        'logits': torch.randn(4, 1, device=device),
        'target': torch.randint(0, 2, (4, 1), device=device),
    }
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

    out = load_json(
        path=str(tmp_path / 'val' / 'column_analysis.json'),
        cls=ColumnAnalysisOutput,
    )
    assert out is not None
    assert out.task_type == TaskType.BINARY_CLASSIFICATION

    cohort = out['USERS', 0, 'AGE']
    assert cohort.stype == Stype.numerical
    assert cohort.cohorts == ['[20 - 22]', '(22 - 30]', '(30 - 40]']
    assert cohort.populations == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert len(cohort.predictions) == 3
    assert len(cohort.targets) == 3
    assert cohort.prediction_score >= 0.0
    assert cohort.target_score >= 0.0

    cohort = out['ORDERS', 1, 'TIME']
    assert cohort.stype == Stype.timestamp
    assert cohort.cohorts == [
        '[0 days, 1 days)',
        '[4 days, 5 days)',
        '[5 days, 6 days)',
        '[6 days, 7 days)',
        '[7 days, 8 days)',
    ]
    assert len(cohort.populations) == 5
    assert sum(cohort.populations) == pytest.approx(1)
    assert len(cohort.predictions) == 5
    assert len(cohort.targets) == 5
    assert cohort.prediction_score >= 0.0
    assert cohort.target_score >= 0.0

    cohort = out['STORES', 2, 'CAT']
    assert cohort.stype == Stype.categorical
    assert cohort.cohorts == ['burger', 'pizza']
    assert len(cohort.populations) == 2
    assert sum(cohort.populations) == pytest.approx(1)
    assert len(cohort.predictions) == 2
    assert len(cohort.targets) == 2
    assert cohort.prediction_score >= 0.0
    assert cohort.target_score >= 0.0


@pytest.mark.parametrize(
    'method,torch_fn',
    [('var', torch.var), ('std', torch.std)],
)
def test_get_score(method: Literal['var', 'std'], torch_fn: Callable) -> None:
    inputs = torch.tensor([2.0, 2.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0])
    expected = float(torch_fn(inputs))

    inputs = torch.tensor([2.0, 4.0, 5.0])
    weight = torch.tensor([2, 2, 4])
    out = ColumnAnalysis.get_score(inputs, weight, method)
    assert out == pytest.approx(expected)

    inputs = torch.tensor([[2.0, 3.0], [4.0, 5.0], [5.0, 1.0], [5.0, 1.0]])
    expected = float(torch_fn(inputs, dim=0).mean())

    inputs = torch.tensor([[2.0, 3.0], [4.0, 5.0], [5.0, 1.0]])
    weight = torch.tensor([1, 1, 2])
    out = ColumnAnalysis.get_score(inputs, weight, method)
    assert out == pytest.approx(expected)

    inputs = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    weight = torch.tensor([2])
    out = ColumnAnalysis.get_score(inputs, weight, method)
    assert out == 0.0

    assert ColumnAnalysis.get_score(
        torch.tensor([]),
        torch.tensor([]),
        method,
    ) == 0.0

    assert ColumnAnalysis.get_score(
        torch.tensor([1]),
        torch.tensor([0]),
        method,
    ) == 0.0

    assert ColumnAnalysis.get_score(
        torch.tensor([1]),
        torch.tensor([1]),
        method,
    ) == 0.0

    assert ColumnAnalysis.get_score(
        torch.tensor([1]),
        torch.tensor([2]),
        method,
    ) == 0.0
