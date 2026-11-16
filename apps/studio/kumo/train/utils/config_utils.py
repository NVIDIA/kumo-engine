# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.job_config import JobConfig
from kumo_core.model_plan import ModelPlan

from io_utils import fsutil, logging
from io_utils.json import save_json
from kumo.graph import Graph
from kumo.home import HomeDirManager

logger = logging.getLogger(__name__)


def save_trainer_job_config(
    home_dir: str,
    task_name: str,
    graph: Graph,
    space: ModelPlan,
    job_id: str,
    diskgraph_index_path: str | None,
    disable_explain: bool = False,
    warm_start_trainer_dir: str | None = None,
) -> str:
    home_dir_mgr = HomeDirManager(home_dir)
    # Save the job configuration to the home directory.
    job_dir = home_dir_mgr.jobs_dir
    fsutil.makedirs(job_dir, exist_ok=True)

    automl_cfg_path = home_dir_mgr.job_automl_model_plan(job_id)
    save_json(space, automl_cfg_path)

    # kumo-auto worker subprocess consumes a raw filesystem path, so the caller
    # resolves the per-job diskgraph index path and threads it in explicitly via
    # JobConfig (no reliance on process-global engine state).
    job_config = JobConfig(
        home_dir=home_dir,
        task_name=task_name,
        automl_cfg_path=automl_cfg_path,
        tables_to_load=graph.table_names,
        nodes_to_load=graph.table_names,
        disable_explain=disable_explain,
        training_job_id=job_id,
        warm_start_trainer_dir=warm_start_trainer_dir,
        diskgraph_index_path=diskgraph_index_path,
    )
    config_path = home_dir_mgr.job_config_path(job_id)
    save_json(job_config, config_path)
    return config_path
