# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from temporalio.worker import Worker

from io_utils import logging
from kumo.data import get_persist_dir
from kumo.home import set_home_dir
from pool.activities import (
    get_pool_stats_activity,
    place_snapshot_activity,
    release_snapshot_activity,
)
from rest.backends.common.activities import io_activities
from rest.backends.data_backend.activities import (
    DataBackendActivities,
    diskgraph_ingest_stats_activity,
    ingest_custom_train_table,
    validate_weighted_train_table,
)
from rest.backends.data_backend.activities.connector_authorization import (
    delegate_connector_authorizations,
    revoke_connector_authorizations,
)
from rest.backends.data_backend.activities.custom_table_validation import (
    validate_custom_train_table_activity,
)
from rest.backends.data_backend.activities.graph_config_activity import (
    build_graph_config_activity,
)
from rest.backends.data_backend.activities.identity_management import (
    create_project_connector_activity,
    validate_identity_credentials_activity,
)
from rest.backends.data_backend.activities.ingest_table_data_activity_v2 import (
    ingest_table_data_activity_v2,
)
from rest.backends.data_backend.activities.prediction_output_validation import (
    validate_object_prediction_output,
    validate_warehouse_prediction_output,
)
from rest.backends.data_backend.connector_activity_registry import (
    validate_connector_activity_registration,
)
from rest.backends.data_backend.workflows import (
    GraphRefreshWorkflow,
    TableIngestionWorkflow,
    TableRefreshWorkflow,
)
from rest.backends.ml_backend.activities import (
    generate_prediction_config,
    generate_training_config,
    materialize_training_table,
    stage_custom_prediction_table,
)
from rest.backends.ml_backend.activities.split import (
    execute_split,
    save_split_workflow_status,
)
from rest.constants import DATA_BACKEND_WORKER_NAME
from rest.driver.connector_api_driver import (
    get_connector_sample_rows,
    get_connector_source_columns,
    get_connector_source_table_file_type,
    is_connector_source_table_view,
    list_connector_source_tables,
    stage_connector_source_download,
    validate_connector_reference,
    validate_connector_source_table_exists,
)
from rest.driver.graph_logic import (
    infer_graph_links_activity,
    validate_graph_definition_activity,
)
from rest.driver.pquery_logic import validate_predictive_query_activity
from rest.driver.table_logic import (
    get_table_col_stats_and_sample_rows_activity,
    infer_table_metadata_activity,
    validate_custom_sql,
    validate_table_definition_activity,
)
from service_framework.deploy_env_config import deploy_env
from temporalio_commons import WorkerEnv
from temporalio_commons.worker_registry import register_bundled_workers

from rest.backends.data_backend.workflows.diskgraph_ingest_workflow import (  # isort:skip
    DiskgraphIngestWorkflow,
)

log = logging.getLogger(__name__)


@register_bundled_workers(worker_name=DATA_BACKEND_WORKER_NAME)
async def run_worker(worker_env: WorkerEnv) -> List[Worker]:
    # The data persist dir is derived from deploy_env() on demand
    # (kumo.data.get_persist_dir); no process-global state to prime here.
    log.info('Data backend worker initialized with persist_dir=%s',
             get_persist_dir())

    assert worker_env.options.max_worker_threads is not None

    # workflow worker uses a thread pool executor.
    # DiskgraphIngestWorkflow dispatches SnapshotActivities.ensure_snapshot
    # to the pool task queue, which owns the per-snapshot lock + dataplane
    # ingest semaphore.
    data_backend_activities = [
            DataBackendActivities.wait_on_workflow,
            DataBackendActivities.start_bp_stats_activity,
            DataBackendActivities.compute_standalone_table_col_stats,
            DataBackendActivities.validate_graph_table_stats,
            DataBackendActivities.validate_table_ingest,
            DataBackendActivities.validate_table_col_stats,
            DataBackendActivities.validate_graph_edge_stats,
            DataBackendActivities.create_table_cache_dir,
            DataBackendActivities.save_table_version_activity,
            DataBackendActivities.save_table_refresh_result,
            DataBackendActivities.save_colstat_task_result,
            DataBackendActivities.copy_colstat_to_job_dir,
            DataBackendActivities.save_source_table_ingest_result,
            DataBackendActivities.refresh_graph_data_version,
            DataBackendActivities.get_connector_table_data_version,
            DataBackendActivities.save_graph_data_version,
            DataBackendActivities.update_table_timestamp_units,
            build_graph_config_activity,
            validate_custom_train_table_activity,
            validate_object_prediction_output,
            validate_warehouse_prediction_output,
            create_project_connector_activity,
            validate_identity_credentials_activity,
            delegate_connector_authorizations,
            revoke_connector_authorizations,
            list_connector_source_tables,
            get_connector_source_columns,
            get_connector_source_table_file_type,
            stage_connector_source_download,
            validate_connector_source_table_exists,
            get_connector_sample_rows,
            is_connector_source_table_view,
            validate_connector_reference,
            infer_graph_links_activity,
            validate_graph_definition_activity,
            validate_predictive_query_activity,
            infer_table_metadata_activity,
            validate_table_definition_activity,
            validate_custom_sql,
            get_table_col_stats_and_sample_rows_activity,
            # Both modes dispatch this — pool path runs the
            # stats-bridge step on the pool queue after ensure_snapshot.
            diskgraph_ingest_stats_activity,
            io_activities.save_validation_response_activity,
            # Pure-PyArrow validation with no process-wide global state, so it
            # runs safely on the thread executor.
            validate_weighted_train_table,
            # I/O-bound connector unload + parquet copy with no process-wide
            # global state, so it runs on the shared thread pool. The trade-off
            # is that a long warehouse unload holds one of the workflow_worker
            # threads (max_worker_threads, 20) for its whole duration, sharing
            # that pool with the lightweight validate_*/save_* activities.
            ingest_custom_train_table,
            # I/O-bound connector export with no engine/per-job global state, so
            # it runs on the shared thread pool; a blocking warehouse unload
            # holds one of those threads for its whole duration.
            ingest_table_data_activity_v2,
            # Computes the prediction config from explicit per-job inputs
            # (home_dir param, settings, job_input). Its only global reads are
            # deploy-wide constants (engine backend, data-backend type) or
            # vestigial-in-DG (edge_persist_paths, which the DG short-circuit
            # never reads), so concurrent jobs on a shared thread can't collide.
            generate_prediction_config,
            # Computes the training config from explicit per-job inputs. The one
            # per-job value it produces (the diskgraph index path) is threaded
            # explicitly into save_trainer_job_config, the home dir is never
            # read on this path, and its remaining global reads are deploy-wide
            # (set at worker startup), so concurrent jobs on a shared thread
            # can't collide.
            generate_training_config,
            # Materializes the train table from explicit per-job values
            # (home_dir, diskgraph handle/index_path, output paths) derived
            # locally per call and passed to callees, touching no per-job
            # process globals, so concurrent jobs on a shared thread can't
            # collide.
            materialize_training_table,
            # Connector-authorized custom PREDICT source localization. The
            # DiskGraph bundle build runs on the snapshot-resident pool worker.
            stage_custom_prediction_table,
            # Splits a materialized train table. Nothing on the split path reads
            # per-job process globals (manifest validated from the explicit
            # path, get_service() built from deploy-wide config, stats cache
            # keyed by job_id), so concurrent jobs on a shared thread can't
            # collide.
            execute_split,
            # Async status-write activity hosted here so the compute queue stays
            # heavy-compute only.
            save_split_workflow_status,
            # Pool integration: workflow-side wrappers that bridge
            # Placement coordinators to PoolController
            # via AutoRPC.
            place_snapshot_activity,
            release_snapshot_activity,
            get_pool_stats_activity,
        ]
    validate_connector_activity_registration(data_backend_activities)
    workflow_worker = await worker_env.build_worker(
        activities=data_backend_activities,
        workflows=[
            TableIngestionWorkflow,
            GraphRefreshWorkflow,
            DiskgraphIngestWorkflow,
            TableRefreshWorkflow,
        ],
    )

    # The compute process-pool worker has been retired: every activity that
    # used to run there is now de-globalized and registered on the thread-pool
    # workflow_worker above. No data-compute-task-queue worker is started.

    # set home dir to match the base dir used for data persist dir
    set_home_dir(deploy_env().data_plane_storage_root)

    return [workflow_worker]
