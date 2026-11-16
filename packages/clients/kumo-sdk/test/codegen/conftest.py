# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumoai_studio.codegen.registry import REG, Handler
from kumoai_studio.testing.codegen import MockSDKObject


@pytest.fixture(autouse=True)
def register_mock_handler():
    handler: Handler[MockSDKObject] = Handler(
        parents=lambda e, ctx: e.parents,
        required_imports=lambda e: e.imports,
        emit_lines=lambda e, var_name, context, codegen_ctx: [
            f"# Emitting {e.name}",
            f"{var_name} = '{e.name}'",
        ],
        detect_edits=None,
    )
    already_registered = MockSDKObject in REG
    if not already_registered:
        REG[MockSDKObject] = handler

    yield

    if not already_registered:
        del REG[MockSDKObject]
