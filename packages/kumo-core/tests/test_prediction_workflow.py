# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from kumo_core.model_plan import ModelPlan, OptimizationPlan
from kumo_core.training.workflow import (
    GNNPredictionResult,
    GNNPredictionWorkflow,
)


class Backend:
    def __init__(self):
        self.request = None

    def predict(self, **request):
        self.request = SimpleNamespace(**request)
        return GNNPredictionResult(
            run_id=request["run_id"],
            status="DONE",
            output_table=request["output_table"],
            row_count=17,
        )


def workflow(backend):
    return GNNPredictionWorkflow(
        backend=backend,
        graph={
            "name": "events",
            "tables": []
        },
        model_plan=ModelPlan(optimization=OptimizationPlan(batch_size=[64]), ),
        task_type="binary_classification",
        prediction_timestamps=(123, ),
        top_k=3,
        return_embeddings=True,
        write_mode="overwrite",
    )


def test_predict_gnn_delegates_exact_configured_request():
    backend = Backend()

    result = workflow(backend).predict_gnn(
        "PREDICT events.label FOR EACH events.id",
        "models:/main.s.events/2",
        "main.s.predictions",
        run_id="prediction-1",
    )

    assert result.to_dict() == {
        "run_id": "prediction-1",
        "status": "DONE",
        "output_table": "main.s.predictions",
        "row_count": 17,
    }
    assert backend.request.query == ("PREDICT events.label FOR EACH events.id")
    assert backend.request.checkpoint_uri == "models:/main.s.events/2"
    assert backend.request.model_plan.optimization.batch_size == [64]
    assert backend.request.task_type == "binary_classification"
    assert backend.request.prediction_timestamps == (123, )
    assert backend.request.top_k == 3
    assert backend.request.return_embeddings is True
    assert backend.request.write_mode == "overwrite"


@pytest.mark.parametrize(
    ("query", "checkpoint_uri", "output_table", "message"),
    (
        ("", "models:/main.s.events/2", "main.s.out", "query"),
        ("PREDICT x.y FOR EACH x.id", "", "main.s.out", "checkpoint_uri"),
        ("PREDICT x.y FOR EACH x.id", "models:/main.s.events/2", "", "output"),
    ),
)
def test_predict_gnn_rejects_missing_required_inputs(
    query,
    checkpoint_uri,
    output_table,
    message,
):
    with pytest.raises(ValueError, match=message):
        workflow(Backend()).predict_gnn(
            query,
            checkpoint_uri,
            output_table,
        )


def test_predict_gnn_requires_exact_backend_result():
    class WrongBackend:
        def predict(self, **request):
            del request
            return {"status": "DONE"}

    with pytest.raises(TypeError, match="GNNPredictionResult"):
        workflow(WrongBackend()).predict_gnn(
            "PREDICT x.y FOR EACH x.id",
            "models:/main.s.events/2",
            "main.s.out",
        )
