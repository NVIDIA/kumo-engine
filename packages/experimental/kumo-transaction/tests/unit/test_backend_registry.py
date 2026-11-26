# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The training-backend registry (backend contracts)."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest

from tfm.contracts.capabilities import BackendCapabilities
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.training import backends
from tfm.training.backends import (
    advertised_backends,
    backend_capabilities,
    build_trainer,
    canonical_backend_name,
    register_backend,
)
from tfm.training.trainer import EagerTrainer


@pytest.fixture
def isolated_registry(monkeypatch):
    monkeypatch.setattr(backends, "_BACKENDS", dict(backends._BACKENDS))


def test_builtin_backends_are_registered() -> None:
    assert {"eager", "ddp", "fsdp2"} <= set(advertised_backends())


def test_names_are_normalized_and_cpu_aliases_eager() -> None:
    assert canonical_backend_name(" Eager ") == "eager"
    assert canonical_backend_name("cpu") == "eager"
    assert isinstance(build_trainer("cpu"), EagerTrainer)


def test_unknown_backend_lists_the_registered_ones() -> None:
    with pytest.raises(UnsupportedCombinationError, match="Registered backends: .*fsdp2"):
        build_trainer("deepspeed")


def test_capability_lookup_does_not_import_other_extras() -> None:
    script = (
        "import sys; from tfm.training.backends import backend_capabilities; "
        "backend_capabilities('eager'); "
        "assert 'tfm.extras.fsdp2' not in sys.modules"
    )
    subprocess.check_call([sys.executable, "-c", script])


def test_register_backend_requires_declared_capabilities(isolated_registry) -> None:
    with pytest.raises(UnsupportedCombinationError, match="must declare"):

        @register_backend("undeclared")
        class Undeclared:
            pass


def test_register_backend_rejects_capabilities_for_another_name(isolated_registry) -> None:
    with pytest.raises(UnsupportedCombinationError, match="declares capabilities for 'eager'"):

        @register_backend("copycat")
        class Copycat(EagerTrainer):
            pass


def test_a_registered_backend_is_buildable(isolated_registry, tmp_path) -> None:
    @register_backend("custom")
    class Custom(EagerTrainer):
        capabilities = replace(
            EagerTrainer.capabilities, name="custom", support_tier="experimental"
        )

    trainer = build_trainer("custom", checkpoint_dir=tmp_path)

    assert isinstance(trainer, Custom)
    assert trainer.checkpoint_dir == tmp_path
    assert backend_capabilities("custom").support_tier == "experimental"


def test_removed_nemo_backend_names_its_replacement() -> None:
    from pydantic import ValidationError

    from tfm.config import TrainingConfig

    with pytest.raises(ValidationError, match="Use training.backend: fsdp2"):
        TrainingConfig(backend="nemo")


def test_capabilities_are_immutable() -> None:
    capabilities = backend_capabilities("eager")
    assert isinstance(capabilities, BackendCapabilities)
    with pytest.raises(AttributeError):
        capabilities.requires_cuda = True  # type: ignore[misc]
