# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from typing import Any, List

from temporalio import activity, workflow

from io_utils import fsutil
from io_utils.json import save_json
from kumo.home import HomeDirManager
from kumo.validation import ValidationResponse


@activity.defn
async def save_validation_response_activity(
    response: ValidationResponse,
    root_path: str,
    unique_id: str,
):
    # Mirrors ValidationResponse.save, but offloads the JSON write off the
    # activity thread pool via save_json.coroutine so this runs on the worker
    # event loop instead of holding a scarce activity-executor thread.
    path = HomeDirManager(root_path).validation_path(unique_id)
    await save_json.coroutine(response, path)


@activity.defn
def load_validation_response_activity(
    root_path: str,
    unique_id: str,
) -> ValidationResponse:
    """Load existing ValidationResponse, or return empty if not found."""
    return ValidationResponse.load_or_create(root_path, unique_id)


async def save_validation_response(response: ValidationResponse, home_dir: str,
                                   job_id: str) -> None:
    """Save the validation response to the job directory."""
    await workflow.execute_activity(
        save_validation_response_activity,
        args=[response, home_dir, job_id],
        schedule_to_close_timeout=timedelta(minutes=20),
    )


@activity.defn
def save_json_activity(pydantic_obj: Any, path: str) -> None:
    """Save a JSON object to a file."""
    save_json(pydantic_obj, path)


@activity.defn
def get_missing_files(file_paths: List[str]) -> List[str]:
    missing_files = []
    for file_path in file_paths:
        size = fsutil.filesize(file_path)
        if size is None or size == 0:
            missing_files.append(file_path)
    return missing_files
