# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe runtime: orchestration that wires data, model, training, and reports.

This is recipe glue, not library architecture. One lifecycle serves every
stack; the config's building-block slots decide what is built (building-block contracts).

- `tfm.runtime.data`: load, split, tokenize, and collate (slots 0 and 1).
- `tfm.runtime.pipeline`: setup, train, infer, evaluate, report, smoke.
- `tfm.runtime.evaluate`: probes and fine-tuning (slot 4).
"""

from tfm.runtime.common import (
    apply_device_override,
    default_run_dir,
    parse_duration,
    parse_timestamp,
    recipe_dir,
)
from tfm.runtime.data import (
    PreparedData,
    PretrainingData,
    prepare_data,
    prepare_pretraining,
    tokenizer_for,
)
from tfm.runtime.pipeline import (
    RunArtifacts,
    build_model,
    default_config_path,
    run_evaluate,
    run_infer,
    run_report,
    run_smoke,
    run_train,
    setup_run,
)
from tfm.training.device import resolve_device

__all__ = [
    "PreparedData",
    "PretrainingData",
    "RunArtifacts",
    "apply_device_override",
    "build_model",
    "default_config_path",
    "default_run_dir",
    "parse_duration",
    "parse_timestamp",
    "prepare_data",
    "prepare_pretraining",
    "recipe_dir",
    "resolve_device",
    "run_evaluate",
    "run_infer",
    "run_report",
    "run_smoke",
    "run_train",
    "setup_run",
    "tokenizer_for",
]
