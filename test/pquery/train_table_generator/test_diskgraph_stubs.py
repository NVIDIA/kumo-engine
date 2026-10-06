# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for narrow DG parity guard stubs in train_table_generator.

Each stub corresponds to a capability the Spark path handles but DG Rust
does not. The guards fire only on the specific trigger so unrelated DG
jobs (e.g., node-pred regression in test_diskgraph_e2e.py) keep passing.
"""
from unittest.mock import MagicMock

import pytest
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType

from kumo.pquery.train_table_generator import (
    _DISKGRAPH_ZERO_ROWS_MARKERS,
    _check_diskgraph_downsampling_parity,
    _check_diskgraph_train_table_parity,
    _diskgraph_empty_train_table_error_message,
    _invoke_diskgraph_train_table_seam,
    _raise_training_table_generation_error_for_diskgraph_empty_result,
)
from kumo.pquery.weight_spec import WeightSpec
from kumo.task.split import TimeRangeSplit
from kumo.task.splits.time_split import TimeSplit
from kumo.validation.train_table_validation import TrainingTableGenerationError
from statedb.utils.errors import UnsupportedInDiskgraphError


def _supported_combo():
    return dict(
        weight_spec=None,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
        split=MagicMock(spec=[]),
    )


def test_train_table_parity_no_raise_for_supported_combo():
    _check_diskgraph_train_table_parity(**_supported_combo())


def test_train_table_parity_raises_on_weight_spec():
    combo = _supported_combo()
    combo['weight_spec'] = MagicMock(spec=WeightSpec)
    with pytest.raises(UnsupportedInDiskgraphError, match='Weighted'):
        _check_diskgraph_train_table_parity(**combo)


def test_train_table_parity_allows_forecasting():
    combo = _supported_combo()
    combo['task_type'] = TaskType.FORECASTING
    _check_diskgraph_train_table_parity(**combo)


def test_train_table_parity_allows_static_node_pred_with_time_range_split():
    combo = _supported_combo()
    combo['task_type'] = TaskType.REGRESSION
    combo['query_type'] = QueryType.STATIC
    combo['split'] = MagicMock(spec=TimeRangeSplit)
    _check_diskgraph_train_table_parity(**combo)


def test_train_table_parity_raises_on_static_node_pred_with_time_split():
    combo = _supported_combo()
    combo['task_type'] = TaskType.REGRESSION
    combo['query_type'] = QueryType.STATIC
    combo['split'] = MagicMock(spec=TimeSplit)
    with pytest.raises(UnsupportedInDiskgraphError,
                       match='static split filtering'):
        _check_diskgraph_train_table_parity(**combo)


def test_train_table_parity_allows_static_node_pred_without_time_split():
    combo = _supported_combo()
    combo['task_type'] = TaskType.REGRESSION
    combo['query_type'] = QueryType.STATIC
    combo['split'] = MagicMock(spec=[])
    _check_diskgraph_train_table_parity(**combo)


def test_train_table_parity_allows_temporal_link_pred_with_time_split():
    combo = _supported_combo()
    combo['task_type'] = TaskType.TEMPORAL_LINK_PREDICTION
    combo['query_type'] = QueryType.TEMPORAL
    combo['split'] = MagicMock(spec=TimeSplit)
    _check_diskgraph_train_table_parity(**combo)


def test_downsampling_parity_no_raise_when_disabled():
    _check_diskgraph_downsampling_parity(False)


def test_downsampling_parity_raises_when_enabled():
    with pytest.raises(UnsupportedInDiskgraphError,
                       match='DISKGRAPH_PRODUCER_DOWNSAMPLING_ENABLED'):
        _check_diskgraph_downsampling_parity(True)


class _FakeDiskgraphDataError(Exception):
    pass


class _FakeDiskgraphSystemError(Exception):
    pass


class _FakeDiskgraphTargetCategoryLimitError(_FakeDiskgraphDataError):
    pass


def _patch_diskgraph_errors(monkeypatch):
    monkeypatch.setattr(
        "kumo.pquery.train_table_generator.DiskgraphDataError",
        _FakeDiskgraphDataError,
    )
    monkeypatch.setattr(
        "kumo.pquery.train_table_generator.DiskgraphSystemError",
        _FakeDiskgraphSystemError,
    )
    monkeypatch.setattr(
        "kumo.pquery.train_table_generator.DiskgraphTargetCategoryLimitError",
        _FakeDiskgraphTargetCategoryLimitError,
    )


@pytest.mark.parametrize("error_cls",
                         [_FakeDiskgraphDataError, _FakeDiskgraphSystemError])
@pytest.mark.parametrize("marker", _DISKGRAPH_ZERO_ROWS_MARKERS)
def test_diskgraph_zero_row_error_raises_training_table_generation_error(
        monkeypatch, error_cls, marker):
    # Both the legacy ``execute_tt_plan returned zero rows`` signal and the
    # TT-NG streaming ``execute_tt_gen`` 0-byte-parquet EOF signal must
    # reclassify to the graceful "Empty training table" validation error.
    _patch_diskgraph_errors(monkeypatch)

    with pytest.raises(TrainingTableGenerationError) as exc_info:
        _raise_training_table_generation_error_for_diskgraph_empty_result(
            error_cls(marker),
            entity_table="patients",
            timestamp_count=1296,
        )

    error = exc_info.value.as_error()
    assert error.title == "Empty training table"
    assert "zero training rows" in error.message
    assert "entity table 'patients'" in error.message
    assert "1296 anchor timestamps" in error.message
    assert "ingested graph edges" in error.message


def test_diskgraph_zero_row_message_handles_static_query_context(monkeypatch):
    _patch_diskgraph_errors(monkeypatch)

    message = _diskgraph_empty_train_table_error_message(
        _FakeDiskgraphDataError(_DISKGRAPH_ZERO_ROWS_MARKERS[0]),
        entity_table="users",
        timestamp_count=0,
    )

    assert message is not None
    assert "without temporal anchor timestamps" in message
    assert "across 0 anchor timestamps" not in message


def test_diskgraph_non_zero_row_error_is_not_reclassified(monkeypatch):
    _patch_diskgraph_errors(monkeypatch)

    result = _raise_training_table_generation_error_for_diskgraph_empty_result(
        _FakeDiskgraphDataError("column missing"),
        entity_table="patients",
        timestamp_count=1296,
    )

    assert result is None


# ---------------------------------------------------------------------------
# Integration coverage: exercise the real seam used by
# ``TrainingTableGeneratorV2.generate_table`` so a zero-row DiskGraph error
# raised from inside the production call path is observably wrapped as a
# ``TrainingTableGenerationError`` (not just inspected via the helper return
# value). The DiskGraph executor itself isn't importable from a unit test, so
# we drive the same ``_invoke_diskgraph_train_table_seam`` wrapper that the
# production code calls and assert the wrapping behavior at that seam.


def test_seam_wraps_diskgraph_zero_row_error_as_training_table_error(
        monkeypatch):
    _patch_diskgraph_errors(monkeypatch)

    def fake_generate_train_table():
        raise _FakeDiskgraphDataError(
            f"{_DISKGRAPH_ZERO_ROWS_MARKERS[0]} for query foo")

    with pytest.raises(TrainingTableGenerationError) as exc_info:
        _invoke_diskgraph_train_table_seam(
            fake_generate_train_table,
            entity_table="patients",
            timestamp_count=42,
        )

    error = exc_info.value.as_error()
    assert error.title == "Empty training table"
    assert "entity table 'patients'" in error.message
    assert "42 anchor timestamps" in error.message
    # The original DiskGraph error must be preserved as the cause so
    # downstream loggers / Sentry see the full traceback.
    assert isinstance(exc_info.value.__cause__, _FakeDiskgraphDataError)


def test_seam_classifies_target_category_limit_as_terminal_validation(
        monkeypatch):
    _patch_diskgraph_errors(monkeypatch)

    def fake_generate_train_table():
        raise _FakeDiskgraphTargetCategoryLimitError(
            'TARGET1 exact category limit exceeded')

    with pytest.raises(TrainingTableGenerationError) as exc_info:
        _invoke_diskgraph_train_table_seam(
            fake_generate_train_table,
            entity_table='patients',
            timestamp_count=42,
        )

    error = exc_info.value.as_error()
    assert error.title == 'Train-table target category limit exceeded'
    assert 'no usable train table statistics artifact was published' in error.message
    assert 'Retrying with unchanged data cannot succeed' in error.message
    assert isinstance(exc_info.value.__cause__,
                      _FakeDiskgraphTargetCategoryLimitError)


def test_seam_does_not_classify_system_error_by_message(monkeypatch):
    _patch_diskgraph_errors(monkeypatch)

    def fake_generate_train_table():
        raise _FakeDiskgraphSystemError(
            'cannot produce complete train-table column stats')

    with pytest.raises(_FakeDiskgraphSystemError,
                       match='cannot produce complete'):
        _invoke_diskgraph_train_table_seam(
            fake_generate_train_table,
            entity_table='patients',
            timestamp_count=42,
        )


def test_seam_passes_through_unrelated_diskgraph_error(monkeypatch):
    _patch_diskgraph_errors(monkeypatch)

    def fake_generate_train_table():
        raise _FakeDiskgraphDataError("schema mismatch on edge table")

    with pytest.raises(_FakeDiskgraphDataError, match="schema mismatch"):
        _invoke_diskgraph_train_table_seam(
            fake_generate_train_table,
            entity_table="patients",
            timestamp_count=42,
        )


def test_seam_returns_result_when_call_succeeds(monkeypatch):
    _patch_diskgraph_errors(monkeypatch)

    sentinel = ("output_path", "manifest_path")

    result = _invoke_diskgraph_train_table_seam(
        lambda: sentinel,
        entity_table="patients",
        timestamp_count=42,
    )

    assert result is sentinel
