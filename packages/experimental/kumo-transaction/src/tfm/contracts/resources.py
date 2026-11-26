# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural ownership capability for resource-backed training inputs."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Closable(Protocol):
    """A caller-owned resource that releases its handles through close().

    For example, a custom minibatch sequence with a ``close()`` method can be
    consumed by the recipe runner without inheriting a loader implementation.
    """

    def close(self) -> None:
        """Release resources owned by this instance."""
        ...
