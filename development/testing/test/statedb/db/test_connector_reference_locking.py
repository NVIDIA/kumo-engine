# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import pytest_asyncio

import statedb.db.ml_jobs as ml_jobs_module
from statedb.config.common import ConnectorID, ProjectID
from statedb.db import MLJobService, StateDB

PROJECT_A = ProjectID.generate()
PROJECT_B = ProjectID.generate()


@pytest_asyncio.fixture
async def ml_jobs(tmp_path) -> MLJobService:
    state_db = await StateDB.open_sqlite(str(tmp_path / 'state.sqlite'))
    try:
        yield MLJobService(state_db)
    finally:
        await state_db.close()


@pytest.mark.asyncio
async def test_custom_training_connector_uses_shared_admission_boundary(
        ml_jobs: MLJobService, monkeypatch) -> None:
    connector_id = ConnectorID.generate()
    events = []

    async def record_admission(tx, project_id, connector_ids) -> None:
        events.append(('admit', project_id, list(connector_ids)))

    monkeypatch.setattr(ml_jobs_module, 'require_connectors_for_admission',
                        record_admission)
    reference = SimpleNamespace(
        project_id=PROJECT_A,
        connector_id=connector_id,
    )

    async with ml_jobs._state_db.write_tx() as tx:
        await ml_jobs._require_connector_ref_in_project(
            tx, reference, PROJECT_A)

    assert events == [
        ('admit', PROJECT_A, [connector_id]),
    ]


@pytest.mark.asyncio
async def test_batch_outputs_delegate_sorted_unique_connectors_to_admission(
        ml_jobs: MLJobService, monkeypatch) -> None:
    first, second = sorted(
        [ConnectorID.generate(), ConnectorID.generate()], key=str)
    events = []

    async def record_admission(tx, project_id, connector_ids) -> None:
        events.append(('admit', project_id, list(connector_ids)))

    monkeypatch.setattr(ml_jobs_module, 'require_connectors_for_admission',
                        record_admission)
    config = SimpleNamespace(
        outputs=[
            SimpleNamespace(connector_id=second),
            SimpleNamespace(connector_id=first),
            SimpleNamespace(connector_id=second),
            SimpleNamespace(),
        ],
        custom_pred_table_source=None,
    )

    async with ml_jobs._state_db.write_tx() as tx:
        await ml_jobs._require_batch_output_connectors_in_project(
            tx, config, PROJECT_A)

    assert events == [
        ('admit', PROJECT_A, [first, second]),
    ]


@pytest.mark.asyncio
async def test_connector_reference_locking_preserves_validation_errors(
        ml_jobs: MLJobService, monkeypatch) -> None:
    connector_id = ConnectorID.generate()
    locked = []

    async def reject_admission(tx, project_id, connector_ids) -> None:
        connector_ids = list(connector_ids)
        locked.append((project_id, connector_ids))
        raise ValueError(
            f"Connector '{connector_ids[0]}' is not active in project "
            f"'{project_id}'")

    monkeypatch.setattr(ml_jobs_module, 'require_connectors_for_admission',
                        reject_admission)

    async with ml_jobs._state_db.write_tx() as tx:
        with pytest.raises(ValueError, match=(
                f"Connector '{connector_id}' belongs to another project")):
            await ml_jobs._require_connector_ref_in_project(
                tx,
                SimpleNamespace(
                    project_id=PROJECT_B,
                    connector_id=connector_id,
                ),
                PROJECT_A,
            )
        assert locked == []

        with pytest.raises(ValueError, match=(
                f"Connector '{connector_id}' is not active in project "
                f"'{PROJECT_A}'")):
            await ml_jobs._require_connector_ref_in_project(
                tx,
                SimpleNamespace(
                    project_id=PROJECT_A,
                    connector_id=connector_id,
                ),
                PROJECT_A,
            )

        with pytest.raises(ValueError, match=(
                f"Connector '{connector_id}' is not active in project "
                f"'{PROJECT_A}'")):
            await ml_jobs._require_batch_output_connectors_in_project(
                tx,
                SimpleNamespace(
                    outputs=[SimpleNamespace(connector_id=connector_id)],
                    custom_pred_table_source=None,
                ),
                PROJECT_A,
            )

    assert locked == [
        (PROJECT_A, [connector_id]),
        (PROJECT_A, [connector_id]),
    ]
