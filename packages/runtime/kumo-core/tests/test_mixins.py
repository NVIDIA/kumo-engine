# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from kumo_core.utils.mixins import CastMixin


@dataclass
class BaseConfig(CastMixin):
    name: str


@dataclass
class SpecializedConfig(BaseConfig):
    enabled: bool = True


def test_cast_preserves_other_cast_mixin_instances() -> None:
    config = SpecializedConfig(name="source")

    assert BaseConfig.cast(config) is config
