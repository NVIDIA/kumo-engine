# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from kumoapi.typing import Dtype

from io_utils import logging
from kumo.task import Task
from service_framework.deploy_env_config import deploy_env
from statedb.config.common import JobID

logger = logging.getLogger(__name__)


def get_job_dir(job_base_dir: str, job_id: JobID) -> str:
    r"""Get the job directory for a given customer and a job id."""
    if isinstance(job_base_dir, list) or isinstance(job_id, list):
        raise RuntimeError(
            f'Unexpected job dir args: {job_base_dir} or {job_id}')
    return os.path.join(job_base_dir, job_id)


def get_job_base_dir() -> str:
    r"""Get the base directory for all jobs under the data-plane root."""
    return os.path.join(deploy_env().data_plane_storage_root, 'jobs')


def get_max_partition_size_with_top_k(max_partition_size: int,
                                      max_rows_per_partition: int, top_k: int,
                                      task: Task) -> int:
    """Get the max parititon size based on top_k and task type.
    """
    max_partition_size = min(max_partition_size,
                             int(max_rows_per_partition / top_k))

    if (task.task_type.is_link_pred and len(task.target_to_entity_paths) == 2
            and len(task.target_to_entity_paths[1]) == 1
            and len(task.target_to_entity_paths[1][0]) == 3):
        rhs_table_name = str(task.target_to_entity_paths[1][0][2])
        rhs_pkey = task.graph.table(rhs_table_name).pkey
        if rhs_pkey is not None and rhs_pkey.dtype == Dtype.string:
            logger.info(f"Reducing max_partitoin_size for link prediction "
                        f"task {task.name=} for string key types.")
            # In case of string ids, we limit the pkey length to be at most 64
            # characters so that we divide the partition size by 16 (*4=64).
            # TODO(dong) Add validation for max pkey length.
            max_partition_size = int(max_partition_size / 16)

    return max_partition_size
