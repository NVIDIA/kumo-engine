# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.io.json import save_json
from kumo_core.job_config import JobConfig


def test_job_config_loads_from_remote_filesystem() -> None:
    path = "memory://jobs/training.json"
    save_json(JobConfig(home_dir="/tmp/job", task_name="task"), path)

    loaded = JobConfig.load(path)

    assert loaded.home_dir == "/tmp/job"
    assert loaded.task_name == "task"
