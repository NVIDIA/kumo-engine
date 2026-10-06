# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
from typing import Any, Optional, Tuple

from kumo_core.model_plan import TrainingTableGenerationPlan
from kumo_core.utils.time import now_utc
from kumoapi.train import TrainingStage
from temporalio import activity

from io_utils import kumo_dataclass, logging
from io_utils.json import load_json, save_json
from kumo.config.task import TrainingTableConfig
from kumo.diskgraph import diskgraph_native_tt_artifacts, get_service
from kumo.home import HomeDirManager
from kumo.stats.train_table_stats import SplitTableStats, SplitTrainTableStats
from kumo.task.split import Split
from kumo.task.training_table import TrainingTable
from kumo.validation.train_table_validation import TrainingTableSplitError
from kumo.validation.validation import ValidationResponse

from ...data_backend import connector_activity_registry

# isort: off
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.training_job import TrainingSpec
from kumo_core.types.schema import FPID
from kumo_core.types import TaskType
from statedb.config.common import (
    ConnectorID,
    GenerateTrainTableJobID,
    JobEventLogEntry,
    JobStatus,
    SplitJobID,
)

log = logging.getLogger(__name__)

# TODO(frg100): Can we remove the ConfigMixin to prevent this?
SplitConfigSerializable = dict


@kumo_dataclass
class SplitWorkflowStatus:
    status: JobStatus
    log: JobEventLogEntry
    validation: ValidationResponse
    materialized_path: Optional[str] = None


@kumo_dataclass
class SplitWorkflowInput:
    split_job_id: SplitJobID
    generate_train_table_job_id: GenerateTrainTableJobID
    split_config: SplitConfigSerializable
    materialized_train_table_data_path: str
    task_name: str
    training_spec: TrainingSpec
    connector_refs: dict[ConnectorID, ConnectorExecutionRef]
    graph_key: Optional[str] = None
    graph_data_version: Optional[str] = None
    diskgraph_manifest_path: Optional[str] = None


@activity.defn
async def save_split_workflow_status(
    status_path: str,
    status: SplitWorkflowStatus,
):
    await save_json.coroutine(status, status_path)


def generate_split_fpid(split_config: Any) -> FPID:
    split = Split.from_config(split_config)
    assert issubclass(type(split), Split)
    return split.fpid


def construct_split_workflow_input(
    generate_train_table_job_id: GenerateTrainTableJobID,
    training_table_generation_plan: TrainingTableGenerationPlan,
    materialized_train_table_data_path: str,
    task_name: str,
    training_spec: TrainingSpec,
    connector_refs: dict[ConnectorID, ConnectorExecutionRef],
    graph_key: Optional[str] = None,
    graph_data_version: Optional[str] = None,
    diskgraph_manifest_path: Optional[str] = None,
) -> Tuple[SplitWorkflowInput, str]:
    split_str = training_table_generation_plan.split
    assert isinstance(split_str, str)
    split: Optional[Split] = (
        training_table_generation_plan.get_split(split_str))
    assert isinstance(split, Split)
    split_job_id = SplitJobID.as_child_of(
        parent_job_id=generate_train_table_job_id,
        split_fpid=split.fpid,
    )
    split_wf_input = SplitWorkflowInput(
        split_job_id=split_job_id,
        generate_train_table_job_id=generate_train_table_job_id,
        split_config=asdict(split.config()),
        materialized_train_table_data_path=materialized_train_table_data_path,
        task_name=task_name,
        training_spec=training_spec,
        connector_refs=connector_refs,
        graph_key=graph_key,
        graph_data_version=graph_data_version,
        diskgraph_manifest_path=diskgraph_manifest_path,
    )
    return split_wf_input, split_job_id.wf_id


def _validate_native_split_manifest(
    manifest_path: Optional[str],
) -> tuple[dict[str, Any], dict[str, Any], int, bytes]:
    """Validate the DiskGraph manifest and extract the parsed split payload.

    Returns ``(manifest, parsed_split, row_count, stage_ids)``. Raises
    ``RuntimeError`` if the manifest is missing, non-authoritative, or carries a
    malformed parsed-split payload.
    """
    if not manifest_path:
        raise RuntimeError(
            'diskgraph path requires diskgraph manifest path but none was provided'
        )

    manifest = get_service().validate_manifest(manifest_path)
    split_meta = manifest.get('split')
    parsed_split = manifest.get('parsed_split')
    if not isinstance(split_meta, dict) or not isinstance(parsed_split, dict):
        raise RuntimeError('diskgraph manifest missing parsed split metadata')

    authoritative = bool(split_meta.get('authoritative', False))
    if not authoritative:
        raise RuntimeError('DiskGraph split artifact is non-authoritative')

    row_count = parsed_split.get('row_count')
    stage_ids = parsed_split.get('stage_ids')
    if not isinstance(row_count, int) or not isinstance(stage_ids,
                                                        (bytes, bytearray)):
        raise RuntimeError('diskgraph parsed split payload malformed')

    return manifest, parsed_split, row_count, bytes(stage_ids)


def _apply_native_split_stats(
    training_table: TrainingTable,
    manifest_path: Optional[str],
    prevalidated_manifest: Optional[dict[str, Any]] = None,
) -> None:
    if prevalidated_manifest is not None:
        manifest = prevalidated_manifest
    else:
        if not manifest_path:
            raise RuntimeError(
                'diskgraph path requires diskgraph manifest path for split stats'
            )
        manifest = get_service().validate_manifest(manifest_path)
    parsed = manifest.get('parsed_split_stats')
    if not isinstance(parsed, dict):
        raise RuntimeError(
            'diskgraph path requires parsed split stats artifact in manifest')

    _apply_native_split_stats_payload(training_table, parsed)


def _split_train_table_stats_from_payload(
    parsed: dict[str, Any], ) -> SplitTrainTableStats:
    """Build a :class:`SplitTrainTableStats` from a validated native
    split-stats sidecar payload (``split_stats.json``)."""
    def _to_stage(stage_payload: Any) -> Optional[SplitTableStats]:
        if stage_payload is None:
            return None
        if not isinstance(stage_payload, dict):
            raise RuntimeError(
                'DiskGraph split-stats stage payload must be dict or null')
        payload = dict(stage_payload)
        for k in ('min_date', 'max_date', 'anchor_date'):
            if k in payload:
                payload[k] = (diskgraph_native_tt_artifacts.
                              normalize_timestamp_like_value(payload.get(k)))
        return SplitTableStats(**payload)

    train_stage_stats = _to_stage(parsed.get('train_table_stats'))
    if train_stage_stats is None:
        raise RuntimeError(
            'DiskGraph split-stats payload missing train_table_stats')

    return SplitTrainTableStats(
        train_table_stats=train_stage_stats,
        val_table_stats=_to_stage(parsed.get('val_table_stats')),
        test_table_stats=_to_stage(parsed.get('test_table_stats')),
    )


def _apply_native_split_stats_payload(
    training_table: TrainingTable,
    parsed: dict[str, Any],
) -> None:
    """Apply an already-validated native split-stats sidecar payload.

    TT-NG uses this after ``execute_tt_ingest`` from the published
    ``split_stats.json`` sidecar. Legacy callers enter through
    :func:`_apply_native_split_stats`, which validates a producer manifest
    first and then delegates here.
    """
    split_stats = _split_train_table_stats_from_payload(parsed)
    _assert_native_split_stats_contract(training_table.task_type, split_stats)

    training_table._split_train_table_stats = split_stats
    training_table._save_split_train_table_stats_to_job_dir(split_stats)
    training_table._update_and_save_stats_cache(split_stats)
    log.info('Applied native diskgraph split stats artifact directly')


def _assert_native_split_stats_contract(
    task_type: Optional[TaskType],
    split_stats: SplitTrainTableStats,
) -> None:
    """Assert DiskGraph split stats satisfy Kumo's runtime contract.

    This is the handoff point after DiskGraph has generated native split stats
    and before Kumo saves/caches them. Add future producer/consumer contract
    assertions here so malformed native stats fail before entering Kumo state.
    """
    if task_type is None or task_type.is_link_pred:
        return
    for stage in TrainingStage.list_stages():
        stage_stats = split_stats.get_stage_table_stats(stage)
        if stage_stats is None:
            continue
        if stage_stats.target_distribution is None:
            raise RuntimeError(
                'DiskGraph split-stats contract violation: '
                f'{stage.value} missing target_distribution for {task_type}')


@connector_activity_registry.connector_authorized_activity
def execute_split(
    job_input: SplitWorkflowInput,
    materialized_split_root_dir: str,
    home_dir: str,
) -> SplitWorkflowStatus:
    # Retained for compatibility with serialized workflow payloads.
    del materialized_split_root_dir
    # This activity runs on the data-backend thread-pool worker, not the
    # compute process pool. It must NOT set up per-job engine/data backends
    # here: that path is gated to process-pool activities and would raise on
    # a thread. The split path needs no per-job engine state anyway -- the
    # manifest is validated from the explicit diskgraph_manifest_path,
    # get_service() is built from the deploy-wide data_compute config, and the
    # stats cache is keyed by per-job job_id -- so avoiding that setup keeps the
    # activity thread-safe (mirrors the materialize_training_table migration).
    if not job_input.graph_key or not job_input.graph_data_version:
        raise RuntimeError(
            "DISKGRAPH mode requires graph_key and graph_data_version "
            "in SplitWorkflowInput")

    home_dir_mgr = HomeDirManager(home_dir)
    training_table_config = load_json(
        home_dir_mgr.materialized_train_table_config_path,
        TrainingTableConfig,
    )
    training_table: TrainingTable = TrainingTable.from_config(
        training_table_config, )

    # DG already produced per-stage partitioned train-table parquet + split
    # metadata at generation; TT-NG training reads the DG bundle, never the
    # re-materialized split parquet (materialize_training_task creates an empty
    # TableData and never load_data's the split path). So the load -> stamp
    # masks -> save_data cycle is pure wasted I/O here. Validate the manifest +
    # emit the manifest-sourced split stats (needed by regr/fcst/node-pred; LP
    # is exempt) and skip the heavy load/save entirely.
    try:
        manifest, _, _, _ = _validate_native_split_manifest(
            job_input.diskgraph_manifest_path)
        _apply_native_split_stats(
            training_table,
            job_input.diskgraph_manifest_path,
            prevalidated_manifest=manifest,
        )
    except TrainingTableSplitError as e:
        return SplitWorkflowStatus(
            status=JobStatus.FAILED,
            log=JobEventLogEntry(
                'Splitting Training Table',
                now_utc(),
                'Encountered an error when splitting the Training Table.',
            ),
            validation=ValidationResponse(errors=[e.as_error()]),
        )
    except Exception as e:
        log.exception('TT-NG native split application failed')
        err = TrainingTableSplitError(
            title='TT-NG native split application failed',
            message=str(e),
        )
        return SplitWorkflowStatus(
            status=JobStatus.FAILED,
            log=JobEventLogEntry(
                'Splitting Training Table',
                now_utc(),
                'Encountered an error when splitting the Training Table.',
            ),
            validation=ValidationResponse(errors=[err.as_error()]),
        )
    return SplitWorkflowStatus(
        status=JobStatus.DONE,
        log=JobEventLogEntry(
            'Splitting Training Table',
            now_utc(),
            'Applied native diskgraph split (no in-memory materialization).',
        ),
        validation=ValidationResponse(),
        # None = "DG owns the split; no re-materialized parquet". Safe on the
        # default (coupled-splits) TT-NG path: training reads the DG bundle,
        # not this path. NOTE: the off-by-default USE_DECOUPLED_SPLITS path
        # treats materialized_path is None as failure
        # (generate_table_workflows.py); TT-NG + decoupled-splits is not a
        # supported combination today -- revisit if it becomes one.
        materialized_path=None,
    )
