# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import shutil
from typing import Any, List, Optional

from io_utils import fsutil, logging
from kumo.config.predictive_query import PredictiveQueryV2Config
from kumo.data import get_active_cache
from kumo.data.backend_cache import BackendCacheManager
from kumo.pquery.train_table_generator import TrainingTableGeneratorV2
from kumo.stats.stats import Stats

logger = logging.getLogger(__name__)


class ReportProgressClass:
    r"""
    Class that tracks progress of the training table generation.
    Contains a single method `report_progress_callback` that is used as a
    callback function for `generator.generate_train_table`.
    Args:
        job_id (str): Job id for indexing stats cache.
    """
    def __init__(self, job_id: str):
        self.iteration_stats: List[Any] = []
        self.job_id = job_id

    def report_progress_callback(self, total_number: int, iteration_data: Any):
        r"""Callback function for `generator.generate_train_table()`"""
        from kumo.stats.train_table_stats import (
            ITERATION_DATA,
            NUM_EXPECTED_ITERATIONS,
        )
        cache: BackendCacheManager = get_active_cache()
        self.iteration_stats.append(iteration_data)
        stored_stats = {
            NUM_EXPECTED_ITERATIONS: Stats(total_number, approx=False),
            ITERATION_DATA: Stats(self.iteration_stats, approx=False),
        }
        cache.pquery_table_progress_cache.put(self.job_id, stored_stats)


def generate_train_table_generic(
    predictive_query_config: PredictiveQueryV2Config,
    output_file_path: str,
    destination: str,
    train_table_validation_response_path: Optional[str] = None,
    job_id: Optional[str] = None,
    diskgraph_index_path: Optional[str] = None,
) -> Optional[str]:
    """
    Generic sql implementation of generate train table
    Args:
        predictive_query_config (PredictiveQueryV2Config):
            PredictiveQueryV2Config for which train table needs to be generated
        output_file_path (str): Output file path where the generated train
            table needs to be stored.
        train_table_validation_response_path (str, optional): Output file path
            where the validation response generated during train table
            generation needs to be stored.
        job_id (str, optional): Job id for indexing stats cache.
            (default: :obj:`None`)
    """
    from kumo.pquery import PredictiveQuery, PredictiveQueryV2
    from kumo.validation import ValidationResponse
    from kumo.validation.train_table_validation import (
        TRAIN_TABLE_VALIDATION_RESPONSE_NAME,
        TrainingTableGenerationError,
    )

    table_details = [(table.table_name, table.data_version)
                     for table in predictive_query_config.graph.tables]
    logger.info(f'Table Name and versions : {table_details}')

    predictive_query: PredictiveQuery = PredictiveQueryV2.from_config(
        predictive_query_config)
    table_generator = predictive_query.get_table_generator()

    # Track validation errors and warnings in train table generation due to
    # predictive query parameters entered by customer
    response = ValidationResponse()
    if job_id is not None:
        report_progress_class = ReportProgressClass(job_id)
        progress_callback = report_progress_class.report_progress_callback
    else:
        progress_callback = None
    published_train_table_path: Optional[str] = None
    try:
        # ``PredictiveQueryV2.get_table_generator`` always returns a
        # ``TrainingTableGeneratorV2`` on DG; the legacy non-V2 branch
        # was retired in the DG cutover.
        assert isinstance(table_generator, TrainingTableGeneratorV2)
        assert isinstance(predictive_query, PredictiveQueryV2)
        table_generator.generate_table(
            job_id=job_id,
            report_progress=progress_callback,
            graph_config=predictive_query_config.graph,
            diskgraph_index_path=diskgraph_index_path,
            list_distinct_target_is_array=(
                predictive_query.list_distinct_target_is_array),
            destination=destination,
        )
        # DiskGraph emits one ordered parquet dataset inside a complete native
        # artifact bundle. Publish that bundle once, then return the selected
        # immutable dataset path to every downstream consumer. On TT-NG the
        # producer does NOT materialize a
        # physical ``arange`` row-index column (the row-index is the implicit
        # contiguous row position). The per-stage train/val/test masks come
        # from the DiskGraph-native split (stamped on ``table.data``); the
        # legacy whole-table ``Split.get_masks`` machinery is dead on TT-NG
        # (see kumo/task/training_table.py::execute_split). The DG
        # short-circuit contract is enforced inside ``generate_table`` which
        # raises ``UnsupportedInDiskgraphError`` if ``diskgraph_index_path``
        # is missing — so by this point the short-circuit always holds.
        source_path = table_generator._diskgraph_last_output_path
        if not source_path:
            raise RuntimeError(
                'diskgraph train table path missing on table generator')

        if not fsutil.isdir(source_path):
            raise RuntimeError(
                'diskgraph producer contract violated: expected parquet '
                f'dataset directory at {source_path}')

        try:
            manifest_src = getattr(
                table_generator,
                '_diskgraph_last_manifest_path',
                None,
            )
            if not isinstance(manifest_src,
                              str) or not fsutil.exists(manifest_src):
                raise RuntimeError(
                    'diskgraph producer manifest missing at '
                    f'{manifest_src!r}; cannot persist TT bundle')

            from kumo.diskgraph.diskgraph_native_tt_artifacts import (
                artifact_file_size,
                validate_artifact_bundle,
            )
            source_manifest = validate_artifact_bundle(manifest_src)
            source_train_table = source_manifest['train_table']
            manifest_train_table_path = source_train_table['path']
            if (osp.normpath(manifest_train_table_path)
                    != osp.normpath(source_path)):
                raise RuntimeError(
                    'diskgraph train-table source is not bound to its '
                    f'manifest: source={source_path!r} '
                    f'manifest={manifest_train_table_path!r}')
            source_size = sum(
                artifact_file_size(part['path'])
                for part in source_train_table['parts']
            )
            if source_size != int(source_train_table['size_bytes']):
                raise RuntimeError(
                    'diskgraph train-table source size changed after '
                    'manifest validation')

            from kumo.data.generic.diskgraph_train_table import (
                TT_PARQUET_FILENAME,
                persist_ttng_bundle,
                ttng_persisted_staging_dir,
            )
            staging_src = osp.dirname(manifest_src)
            staging_dst = ttng_persisted_staging_dir(output_file_path)
            logger.info(
                'Publishing diskgraph producer bundle once to immutable storage: '
                '%s -> %s', staging_src, staging_dst)
            published_bundle = persist_ttng_bundle(
                staging_src, staging_dst)
            published_train_table_path = osp.join(
                published_bundle, TT_PARQUET_FILENAME)
        finally:
            # Clean up local staging directory (created when
            # get_home_dir() returned a remote path).
            staging_dir = getattr(table_generator, '_diskgraph_staging_dir',
                                  None)
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
                table_generator._diskgraph_staging_dir = None

    except TrainingTableGenerationError as e:
        response.errors.append(e.as_error())

    finally:
        response.merge(table_generator.get_validate_train_table_response())
        if train_table_validation_response_path:
            logger.info(f"Save generate train table validation responses to "
                        f"{train_table_validation_response_path}.")
            response.save(train_table_validation_response_path,
                          TRAIN_TABLE_VALIDATION_RESPONSE_NAME)
    return published_train_table_path
