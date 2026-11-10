# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed generation checks the existing SDK table ID and portable export."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumo_core.model_plan import TrainingTableGenerationPlan

import kumo.engine as kumo
from kumo.engine._scratch import scoped_scratch_dir
from kumo.engine.cluster import _TrainingTablePublication
from kumo.engine.pquery.training_table import _table_id


@pytest.mark.parametrize("seed", [None, 0, 123])
def test_description_uses_existing_table_identity_without_generation(
    snapshot: kumo.GraphSnapshot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: int | None,
) -> None:
    """Planning computes the standard table ID before generating rows."""
    workflow = SimpleNamespace(
        query=SimpleNamespace(query_type="static", task_type="binary_classification")
    )
    task_plan = SimpleNamespace(
        task_dataset=SimpleNamespace(required_stages=("train", "val", "test")),
        train_table=SimpleNamespace(
            split_family="RandomSplit", split_config={"rnd_seed": seed}
        ),
    )
    monkeypatch.setattr(
        kumo.PredictiveQuery,
        "_plan_generation",
        Mock(return_value=(workflow, task_plan)),
    )
    monkeypatch.setattr(
        kumo.PredictiveQuery,
        "_generation_options",
        Mock(return_value=TrainingTableGenerationPlan()),
    )
    generate = Mock(side_effect=AssertionError("Planning attempted to generate rows"))
    monkeypatch.setattr(
        "kumo.engine.pquery.predictive_query.generate_train_table", generate
    )
    query = kumo.PredictiveQuery._from_snapshot(
        snapshot, "PREDICT users.label FOR EACH users.id"
    )
    with scoped_scratch_dir(tmp_path):
        table = query._describe_training_table(snapshot, None)
    assert table.id == _table_id(table, snapshot, seed or 0, None, custom=False)
    assert table.staging_dir == tmp_path / "train_tables" / table.id
    assert not table.staging_dir.exists()
    generate.assert_not_called()


@pytest.mark.parametrize(
    "scenario",
    [
        (True, [False, False], 0),
        (False, [False, True], 1),
        (False, [False, False], 0),
    ],
)
def test_coordinator_checks_uploaded_table_id_before_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot: kumo.GraphSnapshot,
    scenario: tuple[bool, list[bool], int],
) -> None:
    """An existing portable export skips generation without a separate cache index."""
    import ray

    uploaded, cached, owner_index = scenario
    cluster = kumo.ComputeCluster(
        kumo.ComputeClusterConfig(), kumo.ArtifactRoot(str(tmp_path))
    )
    actors = [Mock(), Mock()]
    cluster._actors, cluster._active = actors, True
    snapshot = replace(snapshot, _compute=cluster)
    table = kumo.TrainingTable(
        "PREDICT users.label FOR EACH users.id",
        tmp_path / "stage/tt.parquet",
        tmp_path / "stage/manifest.json",
        tmp_path / "stage",
        ("train", "val", "test"),
        TrainingTableGenerationPlan(),
        id="tt-existing-id",
    )
    cluster._snapshots[snapshot.id] = actors[0]
    publication = _TrainingTablePublication(table, snapshot.id, str(tmp_path), None)
    actors[0].describe_training_table.remote.return_value = publication
    monkeypatch.setattr(ray, "get", lambda value: value)
    if uploaded:
        exported = tmp_path / "training_tables" / table.id
        exported.mkdir(parents=True)
        (exported / "_SUCCESS").write_bytes(b"")
    for index, actor in enumerate(actors):
        actor.has_training_table.remote.return_value = cached[index]
        actor.training_table_generate.remote.return_value = table
        actor.download_training_table.remote.return_value = table
    result = cluster._training_table_generate(Mock(query=table.query), snapshot, None)
    assert result.id == table.id and result._compute is cluster
    assert cluster._table_owners[table.id] is actors[owner_index]
    for index, actor in enumerate(actors):
        if uploaded:
            actor.has_training_table.remote.assert_not_called()
            actor.training_table_generate.remote.assert_not_called()
            actor.download_training_table.remote.assert_called_once_with(
                publication, cluster._settings
            )
        else:
            actor.has_training_table.remote.assert_called_once_with(publication)
            assert actor.training_table_generate.remote.call_count == int(
                index == owner_index
            )
    assert not (tmp_path / "training_tables/cache").exists()


def test_every_fit_requests_fresh_ingestion(
    snapshot: kumo.GraphSnapshot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two fits of the same table receive distinct run and extension identities."""
    cluster = kumo.ComputeCluster(
        kumo.ComputeClusterConfig(), kumo.ArtifactRoot(str(tmp_path))
    )
    owner = Mock()
    cluster._actors, cluster._active = [owner], True
    snapshot = replace(snapshot, _compute=cluster)
    table = kumo.TrainingTable(
        "query",
        tmp_path,
        tmp_path,
        tmp_path,
        (),
        TrainingTableGenerationPlan(),
        id="table",
        _compute=cluster,
    )
    cluster._tables[table.id] = snapshot.id
    cluster._table_owners[table.id] = owner
    call = Mock()
    monkeypatch.setattr(cluster, "_call", call)
    cluster.prepare_training(table, snapshot, Mock())
    first = call.call_args.args[2]
    cluster.prepare_training(table, snapshot, Mock())
    second = call.call_args.args[2]
    assert first[:2] == second[:2] == (snapshot.id, table.id)
    assert first[2] != second[2]
    assert call.call_count == 2
