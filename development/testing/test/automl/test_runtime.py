# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from kumo_core.types import TrainTableType

from kumo.automl.runtime import infer_authoring_max_target_neighbors_per_entity

EDGE = ('events', 'USER_ID', 'users')


def authoring_task(train_table_stats=None, **rows) -> SimpleNamespace:
    """A task whose row counts record every read the inference makes."""
    asked: list[str] = []

    def table(name: str) -> SimpleNamespace:
        def num_rows(compute_table_stats_if_missing: bool) -> int:
            asked.append(name)
            return rows[name]

        return SimpleNamespace(num_rows=num_rows)

    task = SimpleNamespace(
        target_to_entity_paths=[[EDGE]],
        get_train_table_stats=lambda after_split: train_table_stats,
        graph={name: table(name)
               for name in rows},
    )
    task.asked = asked
    return task


def test_authoring_neighbor_inference_stays_at_monolith_boundary() -> None:
    # `Task` and `Graph` never reach `kumo_core`: the degree policy is
    # shared, the two lookups that need the monolith are not.
    task = authoring_task(events=100, users=10)

    assert infer_authoring_max_target_neighbors_per_entity(task) == 64


def test_authoring_neighbor_inference_applies_the_shared_degree_formula(
) -> None:
    # 40000 / 100 -> degree 400 -> ceil(sqrt(400) * 8) == 160, the same
    # number `kumo_core.automl.target_fanout` produces for that degree.
    task = authoring_task(events=40_000, users=100)

    assert infer_authoring_max_target_neighbors_per_entity(task) == 160


def test_authoring_neighbor_inference_prefers_measured_node_degrees() -> None:
    task = authoring_task(
        train_table_stats=SimpleNamespace(
            stats_type=TrainTableType.TEMPORAL_LINK_PRED,
            node_degree_stats=SimpleNamespace(avg_entity_degree=400),
        ),
        events=1,
        users=1,
    )

    assert infer_authoring_max_target_neighbors_per_entity(task) == 160
    # `num_rows(compute_table_stats_if_missing=True)` can trigger a full
    # table-statistics computation, so the branch that does not need the row
    # counts must not read them.
    assert task.asked == []


def test_authoring_neighbor_inference_ignores_statistics_without_degrees(
) -> None:
    task = authoring_task(
        train_table_stats=SimpleNamespace(
            stats_type=TrainTableType.TEMPORAL_LINK_PRED,
            node_degree_stats=None,
        ),
        events=40_000,
        users=100,
    )

    assert infer_authoring_max_target_neighbors_per_entity(task) == 160
    assert task.asked == ['events', 'users']


def test_authoring_neighbor_inference_reads_nothing_without_a_path() -> None:
    task = authoring_task(events=40_000, users=100)
    task.target_to_entity_paths = []

    assert infer_authoring_max_target_neighbors_per_entity(task) is None
    assert task.asked == []
