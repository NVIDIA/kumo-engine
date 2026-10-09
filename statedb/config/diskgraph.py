# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp

from service_framework._raw_deploy_config import DiskgraphConfig
from service_framework.deploy_env_config import deploy_env
from statedb.config._graph_key_norm import _graph_key_to_v2


def get_diskgraph_data_root() -> str:
    """Return the diskgraph data root directory from deploy config."""
    dc = deploy_env().data_compute
    assert isinstance(dc, DiskgraphConfig)
    return dc.data_root


def get_diskgraph_index_path_for_job(job_input) -> str:
    """Derive the diskgraph index directory for a job's graph binding.

    Works with both ``TrainingJobInput`` and ``BatchPredictionJobInput``.
    ``graph_key`` is normalized to its ``GraphIDv2`` form so the reader-side
    path matches the writer-side path produced by
    ``pool/activities.py:_ensure_snapshot_local``.
    """
    if job_input.graph_data_compute_input is None:
        raise RuntimeError("DISKGRAPH mode requires graph_data_compute_input")
    if job_input.graph_version_info is None:
        raise RuntimeError("DISKGRAPH mode requires graph_version_info")

    dc = deploy_env().data_compute
    assert isinstance(dc, DiskgraphConfig)
    return osp.join(
        dc.data_root,
        _graph_key_to_v2(job_input.graph_data_compute_input.graph_key),
        job_input.graph_version_info.graph_data_version,
    )


def get_diskgraph_local_ingest_root() -> str:
    """Return the local root directory for diskgraph ingest parquet files.

    Falls back to ``{data_root}/local-ingest`` when
    ``DiskgraphConfig.local_ingest_root`` is unset.
    """
    dc = deploy_env().data_compute
    assert isinstance(dc, DiskgraphConfig)
    return dc.local_ingest_root or os.path.join(dc.data_root, 'local-ingest')
