# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve optional local dataset roots without committed symlinks."""

from __future__ import annotations

import os
from pathlib import Path

from tfm.config import DataConfig
from tfm.contracts.errors import SchemaError

ADAPTER_ENV: dict[str, str] = {
    "tabformer": "TFM_TABFORMER_ROOT",
    "pragmatiq": "TFM_PRAGMATIQ_ROOT",
}

EXAMPLE_PATHS: dict[str, str] = {
    "tabformer": "~/data/TabFormer (build it with `tfm prepare-tabformer --root ~/data/TabFormer`)",
    "pragmatiq": "../pragmatiq/data/synth_1m_seed0",
}


def entity_cap(config: DataConfig) -> int:
    if config.max_entities is not None:
        return int(config.max_entities)
    return int(config.n_entities)


def sample_seed(config: DataConfig, fallback: int) -> int:
    if config.sample_seed is not None:
        return int(config.sample_seed)
    return int(fallback)


def resolve_data_path(config: DataConfig) -> Path:
    adapter = config.adapter.strip().lower()
    if config.path:
        root = Path(config.path).expanduser()
    else:
        env_key = ADAPTER_ENV.get(adapter)
        env_val = os.environ.get(env_key, "") if env_key else ""
        if not env_val:
            example = EXAMPLE_PATHS.get(adapter, "<local-dataset-root>")
            env_hint = f" or set {env_key}" if env_key else ""
            raise SchemaError(
                f"Adapter {adapter!r} requires data.path{env_hint}. "
                f"Do not commit a symlink; a local example is {example}."
            )
        root = Path(env_val).expanduser()
    resolved = root.resolve()
    if not resolved.exists():
        raise SchemaError(f"Dataset path does not exist: {resolved}")
    return resolved
