# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from pathlib import Path

from pool.activities import SnapshotActivities
from rest.backends.data_backend.activities.custom_table_validation import (
    validate_custom_train_table_activity,
)
from rest.backends.data_backend.activities.custom_train_table import (
    validate_weighted_train_table,
)
from rest.backends.ml_backend.activities.materialize_training_task import (
    _ttng_train_table_parse_target,
    materialize_training_task,
)
from rest.backends.ml_backend.workflows.custom_train_table_workflow import (
    CustomTrainTableWorkflow,
)
from rest.backends.ml_backend.workflows.materialize_task_workflows import (
    MaterializeTrainingTaskWorkflow,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_train_table_path_contract_has_no_version_dispatch():
    forbidden = {
        'train_table_artifact_version',
        'original_train_table_artifact_version',
        'resolve_published_train_table_data_path',
    }
    for package in ('kumo', 'pool', 'rest'):
        for path in (_REPO_ROOT / package).rglob('*.py'):
            source = path.read_text()
            for name in forbidden:
                assert name not in source, (path.relative_to(_REPO_ROOT), name)


def test_train_table_consumers_accept_one_physical_path_contract():
    consumers = (
        SnapshotActivities.build_ttng_custom_train_table_bundle,
        SnapshotActivities.materialize_training_task,
        validate_custom_train_table_activity,
        validate_weighted_train_table,
        _ttng_train_table_parse_target,
        materialize_training_task,
        CustomTrainTableWorkflow.run,
        MaterializeTrainingTaskWorkflow.run,
    )
    for consumer in consumers:
        parameters = inspect.signature(consumer).parameters
        assert 'train_table_artifact_version' not in parameters
        assert 'original_train_table_artifact_version' not in parameters
