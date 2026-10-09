# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CodegenContext:
    """Context for code generation containing shared state and mappings."""

    # Maps config IDs to shared parent data
    shared_parents: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Maps config IDs to variable names
    object_to_var: Dict[str, str] = field(default_factory=dict)

    # Execution environment for generated code
    execution_env: Dict[str, Any] = field(default_factory=dict)
