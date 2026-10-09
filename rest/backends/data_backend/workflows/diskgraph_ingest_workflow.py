# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from datetime import timedelta

from kumo_core.diskgraph.snapshot import DiskgraphIngestResult
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

from io_utils import logging
from pool.activities import SnapshotActivities
from pool.dataclasses import ResourceHint, SnapshotActivity
from pool.diskgraph_args import DiskgraphIngestArgs
from pool.errors import (
    NO_SPACE_ERROR_TYPE,
    DiskgraphCapacityError,
    is_no_space_activity_error,
)
from pool.wf_coordinator import snapshot_placement_session_ctx
from rest.backends import DATA_BACKEND_DISKGRAPH_WORKFLOW_TIMEOUT_HOURS
from rest.backends.data_backend import diskgraph_stats_bridge
from rest.backends.data_backend.activities import diskgraph_ingest_activity
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.graph_v2 import GraphDefinition
from rest.config.table_v2 import TableDefinition
from statedb.config._graph_key_norm import _graph_key_to_v2
from statedb.config.common import DataSnapshot

log = logging.getLogger(__name__)


class _LocalNoSpaceAtTier(Exception):
    def __init__(self, cause: ActivityError, tier_ssd_gib: int):
        super().__init__(str(cause))
        self.cause = cause
        self.tier_ssd_gib = tier_ssd_gib


@workflow.defn
class DiskgraphIngestWorkflow:
    """Deterministic ``compose_wf_id`` lets concurrent ``GraphRefreshWorkflow``
    instances share a single ingest via ``start_child_data_wf``.
    """
    @staticmethod
    def compose_wf_id(graph_key: str, graph_data_version: str) -> str:
        return f'dg-ingest:{graph_key}@ver={graph_data_version}'

    @workflow.run
    async def run(
        self,
        graph_definition: GraphDefinition,
        graph_display_name: str,
        data_persist_dir: str,
        table_data_versions: dict[str, str],
        graph_key: str,
        graph_data_version: str,
        all_graph_tables: dict[str, TableDefinition],
        connector_specs: dict[str, ConnectorExecutionRef],
        source_size_bytes: int,
        resource_hint: ResourceHint | None,
        source_size_bytes_by_table: dict[str, int],
    ) -> DiskgraphIngestResult:
        snapshot_id = DataSnapshot.GRAPH.canonicalize(
            f'{_graph_key_to_v2(graph_key)}@{graph_data_version}')

        current_resource_hint = resource_hint or ResourceHint()
        previous_failed_tier = 0
        while True:
            try:
                return await self._ingest_once(
                    snapshot_id=snapshot_id,
                    graph_definition=graph_definition,
                    graph_display_name=graph_display_name,
                    data_persist_dir=data_persist_dir,
                    table_data_versions=table_data_versions,
                    graph_key=graph_key,
                    graph_data_version=graph_data_version,
                    all_graph_tables=all_graph_tables,
                    connector_specs=connector_specs,
                    source_size_bytes=source_size_bytes,
                    source_size_bytes_by_table=source_size_bytes_by_table,
                    resource_hint=current_resource_hint,
                )
            except _LocalNoSpaceAtTier as failure:
                failed_tier = failure.tier_ssd_gib
                if failed_tier <= 0:
                    raise DiskgraphCapacityError(
                        'DiskGraph ingest ran out of local disk on a fixed '
                        'or unreported host whose SSD tier is unknown. This '
                        'deployment cannot automatically re-place the '
                        'snapshot on a larger host.') from failure.cause
                if failed_tier <= previous_failed_tier:
                    raise DiskgraphCapacityError(
                        f'DiskGraph placement did not increase after local '
                        f'disk exhaustion: previous failed tier '
                        f'{previous_failed_tier} GiB, current failed tier '
                        f'{failed_tier} GiB.') from failure.cause
                previous_failed_tier = failed_tier
                current_resource_hint = dataclasses.replace(
                    current_resource_hint,
                    min_ssd_gib=failed_tier + 1,
                )
                workflow.logger.info(
                    'Ingest of %s filled the %d GiB local SSD tier; '
                    're-placing with min_ssd_gib=%d.',
                    snapshot_id,
                    failed_tier,
                    current_resource_hint.min_ssd_gib,
                )

    async def _ingest_once(
        self,
        *,
        snapshot_id: str,
        graph_definition: GraphDefinition,
        graph_display_name: str,
        data_persist_dir: str,
        table_data_versions: dict[str, str],
        graph_key: str,
        graph_data_version: str,
        all_graph_tables: dict[str, TableDefinition],
        connector_specs: dict[str, ConnectorExecutionRef],
        source_size_bytes: int,
        source_size_bytes_by_table: dict[str, int],
        resource_hint: ResourceHint | None,
    ) -> DiskgraphIngestResult:
        # The controller computes the on-SSD estimate
        # (source × ingest_space_multiplier + buffer) from its own config,
        # keeping deploy_env() access out of workflow code.
        async with snapshot_placement_session_ctx(
                snapshot_id=snapshot_id,
                activity=SnapshotActivity.ENSURE_SNAPSHOT,
                source_size_bytes=source_size_bytes,
                resource_hint=resource_hint,
        ) as placement:
            pool_task_queue = placement.task_queue
            min_ssd_gib = (resource_hint.min_ssd_gib
                           if resource_hint is not None else 0)
            if min_ssd_gib > 0 and placement.tier_ssd_gib < min_ssd_gib:
                raise DiskgraphCapacityError(
                    f'DiskGraph placement returned '
                    f'{placement.tier_ssd_gib} GiB for '
                    f'min_ssd_gib={min_ssd_gib}; refusing to repeat ingest '
                    f'on the failed capacity tier.')
            try:
                try:
                    result: DiskgraphIngestResult = (
                        await workflow.execute_activity(
                            SnapshotActivities.ensure_snapshot,
                            args=[
                                snapshot_id,
                                DiskgraphIngestArgs(
                                    graph_definition=graph_definition,
                                    graph_display_name=graph_display_name,
                                    data_persist_dir=data_persist_dir,
                                    table_data_versions=table_data_versions,
                                    connector_specs=connector_specs,
                                    all_graph_tables=all_graph_tables,
                                    source_size_bytes_by_table=(
                                        source_size_bytes_by_table),
                                ),
                            ],
                            schedule_to_close_timeout=timedelta(
                                hours=(
                                    DATA_BACKEND_DISKGRAPH_WORKFLOW_TIMEOUT_HOURS
                                )),
                            heartbeat_timeout=timedelta(minutes=5),
                            # A same-placement retry cannot recover local
                            # ENOSPC. Surface it so run() requests a larger
                            # physical tier.
                            retry_policy=RetryPolicy(
                                maximum_attempts=3,
                                non_retryable_error_types=[NO_SPACE_ERROR_TYPE],
                            ),
                            task_queue=pool_task_queue,
                        ))
                except ActivityError as err:
                    if is_no_space_activity_error(err):
                        raise _LocalNoSpaceAtTier(
                            err, placement.tier_ssd_gib) from err
                    raise
            finally:
                # Synchronous: cleanup is just rmtree, fast enough that
                # blocking stats below isn't worth optimizing around.
                # 1h schedule_to_close absorbs a daemon crash-restart
                # with EBS reattach across a couple retries.
                await workflow.execute_activity(
                    diskgraph_ingest_activity.cleanup_source_files,
                    args=[snapshot_id],
                    schedule_to_close_timeout=timedelta(hours=1),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=2),
                        backoff_coefficient=2.0,
                        maximum_interval=timedelta(seconds=30),
                        maximum_attempts=3,
                    ),
                    task_queue=pool_task_queue,
                )

            pkey_duplicate_counts = await workflow.execute_activity(
                diskgraph_stats_bridge.diskgraph_ingest_stats_activity,
                args=[
                    result.index_path,
                    graph_key,
                    graph_data_version,
                    all_graph_tables,
                    table_data_versions,
                    bool(graph_definition.col_groups),
                    result.ingest_stats_path,
                ],
                schedule_to_close_timeout=timedelta(hours=1),
                heartbeat_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=3),
                task_queue=pool_task_queue,
            )

        return dataclasses.replace(
            result,
            pkey_duplicate_counts=pkey_duplicate_counts,
        )
