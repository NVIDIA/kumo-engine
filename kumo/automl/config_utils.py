# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

from async_utils.sync import blocking_func
from io_utils import fsutil, logging
from kumo.config.mixin import construct_object_string_from_config
from kumo.pquery.predictive_query import PredictiveQuery
from kumo.task import Task

log = logging.getLogger(__name__)


@blocking_func()
def maybe_get_file_from_s3(path: str) -> str | None:
    try:
        return fsutil.read_text(path)
    except Exception as e:
        log.exception(e)
        return None


@blocking_func()
def maybe_get_image_from_s3(path: str) -> str | None:
    try:
        raw = fsutil.read_binary(path)
        base64_utf8_str = base64.b64encode(raw).decode('utf-8')
        ext = path.split('.')[-1]
        return f'data:image/{ext};base64,{base64_utf8_str}'
    except Exception as e:
        log.exception(e)
        return None


@blocking_func()
def get_task(query_id: str, job_dir: str) -> Task | None:
    try:
        return Task.load(query_id, home_dir=job_dir)
    except Exception as e:
        log.exception(e)
        return None


def get_encoders_from_pquery(pquery: PredictiveQuery) -> dict[str, str]:
    encoders: dict[str, str] = {}
    for table in pquery.graph.tables:
        for col in table.cols:
            if col.encoder is not None:
                encoders[f"{table.name}.{col.name}"] = (
                    construct_object_string_from_config(col.encoder))

    return encoders
