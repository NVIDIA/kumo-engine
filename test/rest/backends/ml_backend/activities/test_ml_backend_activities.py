# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from io_utils import fsutil
from io_utils.json import load_json
from kumo.home import HomeDirManager
from rest.backends.ml_backend.activities.ml_backend_activities import (
    MLBackendActivities,
    _load_persisted_gen_pred_table_input,
    persist_job_inputs,
)
from rest.config.batch_prediction_job import BatchPredictionJobInput
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    ProjectID,
)
from statedb.db import MLJobService

GEN_JOB_ID = GeneratePredictionTableJobID("gen-predtable-job-xyz")


@dataclass
class _PersistedPredictionInput:
    id: str
    graph_data_compute_input: dict
    graph_version_info: dict


def _bound_input_harness(monkeypatch, *, invalid=None, connector_error=None):
    project_id = ProjectID.generate()
    bp_id = BatchPredictionJobID.make_random_id()
    gen_id = GeneratePredictionTableJobID.as_child_of(bp_id)
    other_bp_id = BatchPredictionJobID.make_random_id()
    other_gen_id = GeneratePredictionTableJobID.as_child_of(other_bp_id)
    bp_input = SimpleNamespace(
        id=bp_id,
        project_id=project_id,
        validate_connector_execution_refs=MagicMock(),
    )
    bp_row = SimpleNamespace(
        id=other_bp_id if invalid == 'wrong_bp_row_id' else bp_id,
        project_id=project_id,
        config=SimpleNamespace(generate_pred_table_job_id=(
            other_gen_id if invalid == 'wrong_bp_reference' else gen_id)),
    )
    persisted = SimpleNamespace(
        id=other_bp_id if invalid == 'wrong_persisted_id' else bp_id,
        project_id=(ProjectID.generate()
                    if invalid == 'wrong_persisted_project' else project_id),
        generate_pred_table_job_id=(
            other_gen_id if invalid == 'wrong_persisted_gen_id' else gen_id),
        validate_connector_execution_refs=MagicMock(
            side_effect=(ValueError(connector_error)
                         if connector_error else None)),
    )
    service = SimpleNamespace(
        get_batch_prediction_rows=AsyncMock(
            return_value=[] if invalid == 'missing_bp_row' else [bp_row]),
        get_generate_pred_table_jobs=AsyncMock(
            return_value=[] if invalid == 'missing_gen_row' else [
                SimpleNamespace(
                    id=(other_gen_id
                        if invalid == 'wrong_gen_row_id' else gen_id),
                    project_id=(ProjectID.generate()
                                if invalid == 'wrong_gen_row_project' else
                                project_id),
                )
            ]),
    )
    activities = object.__new__(MLBackendActivities)
    activities._ml_job_service = cast(MLJobService, service)
    monkeypatch.setattr(
        'rest.backends.ml_backend.activities.ml_backend_activities.'
        '_load_persisted_gen_pred_table_input',
        lambda _bp_id, _gen_id: (bp_id, persisted),
    )
    return SimpleNamespace(
        activities=activities,
        bp_input=bp_input,
        bp_id=bp_id,
        gen_id=gen_id,
        persisted=persisted,
        project_id=project_id,
        service=service,
    )


@pytest.mark.parametrize("coupled", [True, False])
@patch("rest.backends.ml_backend.activities.ml_backend_activities.load_json")
def test_load_persisted_gen_pred_table_input_uses_relationship_owned_home(
        mock_load_json, monkeypatch, coupled):
    bp_id = BatchPredictionJobID.make_random_id()
    gen_id = (GeneratePredictionTableJobID.as_child_of(bp_id) if coupled else
              GeneratePredictionTableJobID.make_random_id())
    expected_input_id = (bp_id if coupled else
                         BatchPredictionJobID(str(gen_id)))
    gen_input = MagicMock()
    gen_input.graph_data_compute_input = MagicMock()
    gen_input.graph_version_info = MagicMock()
    mock_load_json.return_value = gen_input
    monkeypatch.setattr(
        "rest.backends.ml_backend.activities.ml_backend_activities."
        "get_job_base_dir", lambda: "/data/jobs")

    input_id, loaded = _load_persisted_gen_pred_table_input(bp_id, gen_id)

    assert input_id == expected_input_id
    assert loaded is gen_input
    mock_load_json.assert_called_once_with(
        HomeDirManager(f"/data/jobs/{expected_input_id}").
        prediction_job_input_file,
        BatchPredictionJobInput,
    )


@patch("rest.backends.ml_backend.activities.ml_backend_activities.load_json")
def test_load_persisted_gen_pred_table_input_missing_fails_closed(
        mock_load_json):
    bp_id = BatchPredictionJobID.make_random_id()
    gen_id = GeneratePredictionTableJobID.as_child_of(bp_id)
    mock_load_json.side_effect = FileNotFoundError

    input_id, loaded = _load_persisted_gen_pred_table_input(bp_id, gen_id)

    assert input_id == bp_id
    assert loaded is None


@pytest.mark.parametrize("coupled", [True, False])
def test_default_persistence_and_locator_share_relationship_owned_home(
        tmp_path, monkeypatch, coupled):
    jobs_dir = tmp_path / "jobs"
    bp_id = BatchPredictionJobID.make_random_id()
    gen_id = (GeneratePredictionTableJobID.as_child_of(bp_id) if coupled else
              GeneratePredictionTableJobID.make_random_id())
    input_id = (bp_id if coupled else BatchPredictionJobID(str(gen_id)))
    other_owner = gen_id if coupled else bp_id
    job_input = _PersistedPredictionInput(
        id=str(input_id),
        graph_data_compute_input={"graph": "pinned"},
        graph_version_info={"version": "v1"},
    )
    monkeypatch.setattr(
        "rest.backends.ml_backend.activities.ml_backend_activities."
        "get_job_base_dir", lambda: str(jobs_dir))
    monkeypatch.setattr(
        "rest.backends.ml_backend.activities.ml_backend_activities."
        "BatchPredictionJobInput", _PersistedPredictionInput)

    # No explicit home: this is the production persistence rule used by the
    # coupled and standalone generate-prediction-table workflows.
    persist_job_inputs(job_input, {"setting": "value"})
    expected_input_id, loaded = _load_persisted_gen_pred_table_input(
        bp_id, gen_id)

    expected_path = HomeDirManager(
        str(jobs_dir / str(input_id))).prediction_job_input_file
    other_path = HomeDirManager(
        str(jobs_dir / str(other_owner))).prediction_job_input_file
    assert expected_input_id == input_id
    assert loaded == job_input
    assert os.path.isfile(expected_path)
    assert not os.path.exists(other_path)


@pytest.mark.asyncio
async def test_bound_gen_pred_input_rechecks_durable_project_identity(
        monkeypatch):
    case = _bound_input_harness(monkeypatch)

    assert await case.activities._load_project_bound_gen_pred_table_input(
        cast(BatchPredictionJobInput, case.bp_input),
        case.gen_id) is case.persisted
    case.service.get_batch_prediction_rows.assert_awaited_once_with(
        [case.bp_id], project_ids={case.project_id})
    case.service.get_generate_pred_table_jobs.assert_awaited_once_with(
        [case.gen_id], project_ids={case.project_id})


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_boundary", [
    "missing_bp_row",
    "wrong_bp_row_id",
    "wrong_bp_reference",
    "missing_gen_row",
    "wrong_gen_row_id",
    "wrong_gen_row_project",
    "wrong_persisted_project",
    "wrong_persisted_id",
    "wrong_persisted_gen_id",
])
async def test_bound_gen_pred_input_fails_closed_on_identity_mismatch(
        monkeypatch, invalid_boundary):
    case = _bound_input_harness(monkeypatch, invalid=invalid_boundary)

    with pytest.raises(ValueError):
        await case.activities._load_project_bound_gen_pred_table_input(
            cast(BatchPredictionJobInput, case.bp_input), case.gen_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["wrong_project_ref", "extra_output_ref"])
async def test_bound_gen_pred_input_rejects_tampered_connector_refs(
        monkeypatch, tamper):
    case = _bound_input_harness(monkeypatch, connector_error=tamper)

    with pytest.raises(ValueError, match=tamper):
        await case.activities._load_project_bound_gen_pred_table_input(
            cast(BatchPredictionJobInput, case.bp_input), case.gen_id)

    case.persisted.validate_connector_execution_refs.assert_called_once_with()


@patch("rest.backends.ml_backend.activities.ml_backend_activities.save_json_atomic")
@patch("rest.backends.ml_backend.activities.ml_backend_activities.fsutil")
def test_persist_job_inputs_overwrites_without_deleting_published_inputs(
        mock_fsutil, mock_save_json, tmp_path):
    job_input = MagicMock()
    job_input.id = GEN_JOB_ID
    ml_job_settings = MagicMock()
    home_dir = str(tmp_path / 'gen-home')
    # `fsutil` is mocked wholesale, so `makedirs` creates nothing; report the
    # directory as non-local rather than letting a truthy Mock send `chmod`
    # at a path that does not exist.
    mock_fsutil.is_local.return_value = False

    # Generate-table workflows publish once at startup and again after graph
    # refresh. The second publication must not make the first one disappear.
    persist_job_inputs(job_input, ml_job_settings, home_dir)
    persist_job_inputs(job_input, ml_job_settings, home_dir)

    mock_fsutil.rm.assert_not_called()
    mock_fsutil.makedirs.assert_called_with(
        f'{home_dir}/inputs', exist_ok=True)
    assert mock_save_json.call_count == 4
    assert mock_save_json.call_args_list == [
        call(ml_job_settings,
             f'{home_dir}/inputs/ml_job_settings.json'),
        call(job_input,
             f'{home_dir}/inputs/prediction_job_input.json'),
        call(ml_job_settings,
             f'{home_dir}/inputs/ml_job_settings.json'),
        call(job_input,
             f'{home_dir}/inputs/prediction_job_input.json'),
    ]


@dataclass
class _FakePredictionJobInput:
    id: str
    generation: int


def test_persist_job_inputs_keeps_previous_input_visible_until_publish(
        tmp_path, monkeypatch):
    home_dir = str(tmp_path / 'gen-home')
    manager = HomeDirManager(home_dir)
    first = _FakePredictionJobInput('gen-predtable-job', 1)
    second = _FakePredictionJobInput('gen-predtable-job', 2)
    persist_job_inputs(first, {'generation': 1}, home_dir)

    replace_ready = threading.Event()
    allow_replace = threading.Event()
    original_replace = os.replace

    def paused_input_replace(src, dst):
        if dst == manager.prediction_job_input_file:
            replace_ready.set()
            assert allow_replace.wait(timeout=5)
        original_replace(src, dst)

    monkeypatch.setattr('io_utils.json.os.replace', paused_input_replace)
    writer = threading.Thread(
        target=persist_job_inputs,
        args=(second, {'generation': 2}, home_dir),
    )
    writer.start()
    assert replace_ready.wait(timeout=5)
    assert load_json(manager.prediction_job_input_file,
                     dict)['generation'] == 1
    allow_replace.set()
    writer.join(timeout=5)
    assert not writer.is_alive()
    assert load_json(manager.prediction_job_input_file,
                     dict)['generation'] == 2
    assert stat.S_IMODE(os.stat(manager.inputs_dir).st_mode) == 0o700
    assert stat.S_IMODE(
        os.stat(manager.prediction_job_input_file).st_mode) == 0o600


@pytest.mark.parametrize('scheme', ['abfs', 'abfss'])
def test_persist_job_inputs_uses_atomic_adls_publication_in_marker_order(
        tmp_path, monkeypatch, scheme):
    home_dir = str(tmp_path / 'gen-home')
    job_input = _FakePredictionJobInput('gen-predtable-job', 1)
    settings = {'generation': 1}
    legacy_save = MagicMock()
    atomic_save = MagicMock()
    monkeypatch.setattr(fsutil, 'get_scheme', lambda _path: scheme)
    monkeypatch.setattr(
        'rest.backends.ml_backend.activities.ml_backend_activities.save_json',
        legacy_save)
    monkeypatch.setattr(
        'rest.backends.ml_backend.activities.ml_backend_activities.'
        'save_json_atomic', atomic_save)

    persist_job_inputs(job_input, settings, home_dir)

    manager = HomeDirManager(home_dir)
    assert atomic_save.call_args_list == [
        call(settings, manager.ml_job_settings_file),
        call(job_input, manager.prediction_job_input_file),
    ]
    legacy_save.assert_not_called()


def test_persist_job_inputs_preserves_legacy_provider_with_explicit_warning(
        tmp_path, monkeypatch, caplog):
    scheme = 'dbfs'
    home_dir = str(tmp_path / 'gen-home')
    job_input = _FakePredictionJobInput('gen-predtable-job', 1)
    settings = {'generation': 1}
    legacy_save = MagicMock()
    atomic_save = MagicMock()
    monkeypatch.setattr(fsutil, 'get_scheme', lambda _path: scheme)
    monkeypatch.setattr(
        'rest.backends.ml_backend.activities.ml_backend_activities.save_json',
        legacy_save)
    monkeypatch.setattr(
        'rest.backends.ml_backend.activities.ml_backend_activities.'
        'save_json_atomic', atomic_save)

    with caplog.at_level('WARNING'):
        persist_job_inputs(job_input, settings, home_dir)

    manager = HomeDirManager(home_dir)
    assert legacy_save.call_args_list == [
        call(settings, manager.ml_job_settings_file),
        call(job_input, manager.prediction_job_input_file),
    ]
    atomic_save.assert_not_called()
    warnings = [
        record.message for record in caplog.records
        if 'legacy non-atomic overwrite' in record.message
    ]
    assert len(warnings) == 1
    assert scheme in warnings[0]
