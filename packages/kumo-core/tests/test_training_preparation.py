# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local and remote fits share validation, dataset planning and ingestion."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumo_core.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumo_core.training import preparation, workflow
from kumo_core.training.custom_table import contract
from kumo_core.training.search import AutoMLSearchOptions
from kumo_core.training.table import GeneratedTrainTableResult
from kumo_core.types import TaskType


@pytest.fixture
def fit_preparation(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Isolate files and native work while retaining the real preparation flow."""
    request = Mock(
        spec=workflow.TrainingWorkflowRequest,
        query=SimpleNamespace(
            entity_table="users",
            rhs_node_type="products",
            task_type="binary_classification",
            query_type="static",
        ),
        graph=Mock(),
        model_plan=ModelPlan(),
        training_table_generation_plan=TrainingTableGenerationPlan(
            lag_timesteps=3, year_over_year=True
        ),
        prepared_graph=workflow.PreparedGraph("snapshot", "/graph"),
        run_id="fit",
        artifact_dir="/fit",
        search=AutoMLSearchOptions(),
    )
    dataset = SimpleNamespace(feature_schema={"node_types": {}})
    generated = GeneratedTrainTableResult(
        "/staged/tt.parquet", "/staged/manifest.json", "/staged"
    )
    operations = Mock()
    validate = Mock()
    build = Mock(return_value=dataset)
    features = Mock()
    mkdir = Mock()
    materialize = Mock(
        return_value=Mock(
            spec=workflow.MaterializedTrainingTask,
            task=SimpleNamespace(
                split=SimpleNamespace(stage_counts={"test": 1}),
                task_type=TaskType.BINARY_CLASSIFICATION,
            ),
        )
    )
    for name, method in (
        ("validate", validate),
        ("build", build),
        ("features", features),
        ("mkdir", mkdir),
        ("materialize", materialize),
    ):
        operations.attach_mock(method, name)
    monkeypatch.setattr(preparation, "read_spec", Mock(return_value={}))
    monkeypatch.setattr(
        preparation, "stage_parts", Mock(return_value={"train": ["rows.parquet"]})
    )
    monkeypatch.setattr(
        preparation.parquet,
        "read_schema",
        Mock(return_value=SimpleNamespace(names=[contract()["weight_output_column"]])),
    )
    monkeypatch.setattr(preparation, "query_is_temporal", Mock(return_value=False))
    monkeypatch.setattr(preparation, "target_to_entity_paths", Mock(return_value={}))
    monkeypatch.setattr(preparation, "to_plan_graph", Mock(return_value="graph"))
    monkeypatch.setattr(workflow, "resolve_model_plan", validate)
    monkeypatch.setattr(preparation, "build_generated_training_task_dataset", build)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(workflow, "materialize_generated_training_task", materialize)
    return SimpleNamespace(
        request=request,
        generated=generated,
        dataset=dataset,
        operations=operations,
        validate=validate,
        build=build,
        features=features,
        mkdir=mkdir,
        materialize=materialize,
    )


@pytest.mark.parametrize("with_features", [False, True])
def test_shared_fit_prepares_before_ingestion(
    fit_preparation: SimpleNamespace, with_features: bool
) -> None:
    """Both execution locations validate and compile the same fit specification."""
    state = fit_preparation

    result = preparation.prepare_training_task(
        state.request,
        state.generated,
        ("train", "val"),
        state.features if with_features else None,
    )

    expected = (
        ["validate", "build"]
        + (["features"] if with_features else [])
        + ["mkdir", "materialize"]
    )
    assert [call[0] for call in state.operations.mock_calls] == expected
    assert result is state.materialize.return_value
    facts = state.validate.call_args.kwargs
    assert facts["has_train_table_weight_col"] is True
    assert facts["entity_table_names"] == ["users", "products"]
    assert state.build.call_args.kwargs["lag_timesteps"] == 3
    assert state.build.call_args.kwargs["year_over_year"] is True
    assert state.build.call_args.kwargs["required_stages"] == ("train", "val")
    state.materialize.assert_called_once_with(
        state.request, state.dataset, state.generated, tt_id="tt-fit"
    )


@pytest.mark.parametrize("failure", ["validate", "features"])
def test_shared_fit_rejects_before_ingestion(
    fit_preparation: SimpleNamespace, failure: str
) -> None:
    """Invalid plans and missing training resources fail before native work."""
    state = fit_preparation
    getattr(state, failure).side_effect = ValueError("invalid fit")

    with pytest.raises(ValueError, match="invalid fit"):
        preparation.prepare_training_task(
            state.request, state.generated, ("train",), state.features
        )

    state.mkdir.assert_not_called()
    state.materialize.assert_not_called()


@pytest.mark.parametrize(
    "task_type", [TaskType.BINARY_CLASSIFICATION, TaskType.LINK_PREDICTION]
)
@pytest.mark.parametrize("test_rows", [0, 1, None])
def test_shared_preparation_checks_test_split(
    fit_preparation: SimpleNamespace,
    task_type: TaskType,
    test_rows: int | None,
) -> None:
    """Local and remote preparation apply the same empty-test policy."""
    state = fit_preparation
    result = state.materialize.return_value
    result.task.task_type = task_type
    result.task.split.stage_counts = {} if test_rows is None else {"test": test_rows}
    if test_rows == 0 and not task_type.is_link_pred:
        with pytest.raises(ValueError, match="The test split is empty"):
            preparation.prepare_training_task(
                state.request, state.generated, ("train",)
            )
    else:
        assert (
            preparation.prepare_training_task(
                state.request, state.generated, ("train",)
            )
            is result
        )
    state.materialize.assert_called_once()


@pytest.mark.parametrize("overridden", [None, frozenset({"users.bio"})])
@pytest.mark.parametrize("fails", [False, True])
def test_encoder_warning_filter_restores_after_preparation(
    caplog: pytest.LogCaptureFixture,
    overridden: frozenset[str] | None,
    fails: bool,
) -> None:
    """Selected warnings are suppressed only while inference uses the context."""
    import logging

    from kumo_core.encoder_selection import no_encoder_inference_warnings

    logger = logging.getLogger("kumo_core.encoder_selection")
    outcome = (
        pytest.raises(ValueError, match="inference failed") if fails else nullcontext()
    )
    with (
        caplog.at_level(logging.WARNING),
        outcome,
        no_encoder_inference_warnings(overridden),
    ):
        logger.warning("%s inferred", "users.bio")
        logger.warning("%s inferred", "users.age")
        if fails:
            raise ValueError("inference failed")
    logger.warning("%s restored", "users.bio")
    assert "users.bio inferred" not in caplog.text
    assert ("users.age inferred" in caplog.text) is (overridden is not None)
    assert "users.bio restored" in caplog.text
