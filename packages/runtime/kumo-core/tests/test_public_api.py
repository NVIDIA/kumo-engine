# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys


def test_public_objects_are_owned_by_explicit_modules() -> None:
    source = """
import kumo_core
import kumo_core.types

assert not hasattr(kumo_core, "Dtype")
assert not hasattr(kumo_core, "NativeRuntimeConfig")
assert not hasattr(kumo_core, "StageRunner")

# `Dtype` and `Stype` are wire contract, so kumoapi owns them outright. They
# are deliberately not re-exported here: one import path, so the two copies
# cannot drift again.
assert not hasattr(kumo_core.types, "Dtype")
assert not hasattr(kumo_core.types, "Stype")

from kumo_core.config import NativeRuntimeConfig
from kumo_core.pipeline import StageRunner
from kumo_core.types.enums import TaskType
from kumoapi.typing import Dtype, Stype

assert Dtype.__module__ == "kumoapi.typing"
assert Stype.__module__ == "kumoapi.typing"
assert TaskType.__module__ == "kumo_core.types.enums"
assert NativeRuntimeConfig.__module__ == "kumo_core.config"
assert StageRunner.__module__ == "kumo_core.pipeline"
"""

    subprocess.run([sys.executable, "-c", source], check=True)


def test_training_namespace_does_not_export_or_load_runtime() -> None:
    source = """
import sys
import kumo_core.training

assert not hasattr(kumo_core.training, "run_callable_training")
assert not hasattr(kumo_core.training, "LocalDiskgraphTrainingRuntime")
assert "torch" not in sys.modules
assert "torch_geometric" not in sys.modules
assert "pytorch_lightning" not in sys.modules

from kumo_core.training.orchestration import run_callable_training
from kumo_core.training.runtime import LocalDiskgraphTrainingRuntime

assert callable(run_callable_training)
assert LocalDiskgraphTrainingRuntime.__module__ == "kumo_core.training.runtime"
"""

    subprocess.run([sys.executable, "-c", source], check=True)
